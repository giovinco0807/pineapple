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
from pathlib import Path

import pytest

from ofc_regular.cards import create_deck
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.state import Board
from ofc_regular import hu_m31_t3_step6d_fresh_quality_v1 as quality
from ofc_regular import hu_m31_t3_step6d_fresh_quality_transport_v1 as subject


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = (
    ROOT
    / "outputs/gcp_runs/regular-hu-m31-c02-full100-dev-20260717-001/"
    "package_src/native/candidate/release/libofc_hu_m3_engine.so"
)
FEATURE = (
    ROOT
    / "outputs/gcp_runs/regular-hu-m31-c02-full100-dev-20260717-001/"
    "package_src/target/release/libofc_stage3_feature_encoder.so"
)
STARTUP = ROOT / "scripts/startup_hu_m31_t3_step6d_fresh_quality_v1.sh"

_AUTHORIZATION = {
    "schema": quality.PERFORMANCE_RECEIPT_SCHEMA,
    "status": "qualified",
    "decision": quality.QUALIFIED_DECISION,
    "receipt_sha256": "a" * 64,
    "performance_lock_qualified": True,
    "quality_pilot_authorized": True,
    "performance_lock_finalized": True,
    "one_shot_lock_consumed": True,
    "current_profile_changed": False,
}


def _observations(pair_index: int, phase: str) -> tuple[ActorObservation, ...]:
    deck = create_deck(shuffle=False)
    if phase == quality.CONFIRMATION_PHASE:
        deck = list(reversed(deck))
    offset = pair_index % len(deck)
    deck = deck[offset:] + deck[:offset]
    first_board = Board.from_rows(
        top=deck[0:2], middle=deck[2:6], bottom=deck[6:9]
    )
    second_board = Board.from_rows(
        top=deck[9:11], middle=deck[11:15], bottom=deck[15:18]
    )
    first = ActorObservation(
        hero_board=first_board,
        opponent_public_board=second_board,
        dealt_cards=tuple(deck[20:23]),
        hero_private_discards=tuple(deck[18:20]),
        seat="first",
        street="T3",
        to_act_order="first",
    )
    first_after = Board.from_rows(
        top=first_board.top,
        middle=(*first_board.middle, deck[20]),
        bottom=(*first_board.bottom, deck[21]),
    )
    second = ActorObservation(
        hero_board=second_board,
        opponent_public_board=first_after,
        dealt_cards=tuple(deck[25:28]),
        hero_private_discards=tuple(deck[23:25]),
        seat="second",
        street="T3",
        to_act_order="second",
    )
    return first, second


@pytest.fixture()
def quality_package(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> dict[str, object]:
    monkeypatch.setattr(
        quality,
        "_load_performance_authorization",
        lambda _path: dict(_AUTHORIZATION),
    )
    plan = quality.build_fresh_quality_plan(
        performance_receipt_path=tmp_path / "performance.json"
    )
    plan_path = tmp_path / "plan.json"
    quality._write_once(plan_path, plan)
    roots = tmp_path / "roots"
    for row in quality.schedule_rows():
        value = quality._root_value(
            plan=plan,
            row=row,
            observations=_observations(row["pair_index"], row["phase"]),
        )
        quality._write_once(
            roots / quality._root_relative_path(row["phase"], row["pair_index"]),
            value,
        )
    materialization = quality.build_materialization_receipt(
        plan=plan, root_directory=roots
    )
    seal = quality.build_root_seal(plan=plan, materialization=materialization)
    materialization_path = tmp_path / "materialization.json"
    seal_path = tmp_path / "seal.json"
    quality._write_once(materialization_path, materialization)
    quality._write_once(seal_path, seal)
    package_dir = tmp_path / "package"
    archive = tmp_path / "fresh-quality.zip"
    result = quality.create_fresh_quality_package(
        plan_path=plan_path,
        materialization_path=materialization_path,
        seal_path=seal_path,
        performance_receipt_path=tmp_path / "performance.json",
        output_directory=package_dir,
        archive_path=archive,
    )
    return {
        "archive": archive,
        "sha256": result["archive_sha256"],
        "directory": package_dir,
        "plan": plan,
        "seal": seal,
    }


def _fake_row(
    root,
    record,
    observation,
    phase,
    _library,
    _library_sha,
):
    return {
        "phase": phase,
        "pair_index": root["pair_index"],
        "root_index": record["root_index"],
        "seat": observation.seat,
        "observation_fingerprint": observation.fingerprint(),
        "observation_sha256": record["observation_sha256"],
        "primary_wall_seconds": 1.0,
        "primary_decision": {"selected_action_key": "test-only"},
        "confirmation_wall_seconds": (
            2.0 if phase == quality.CONFIRMATION_PHASE else None
        ),
        "confirmation": (
            {"decision": {"selected_action_key": "test-only"}}
            if phase == quality.CONFIRMATION_PHASE
            else None
        ),
        "peak_rss_bytes": 123_456,
    }


def _patch_decision_oracle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subject.gate,
        "_validate_result_row",
        lambda row, *, job, root_lookup: {"row": deepcopy(dict(row))},
    )
    monkeypatch.setattr(
        subject.gate,
        "validate_job_result",
        lambda value, *, job, plan, seal, root_lookup: (
            deepcopy(dict(value)),
            [],
        ),
    )


def _job_sha(package_dir: Path, job_id: str) -> str:
    return subject.sha256_file(package_dir / "jobs" / f"{job_id}.json")


def test_portable_archive_replay_and_all_15_jobs(
    quality_package: dict[str, object], tmp_path: Path
) -> None:
    extracted = tmp_path / "extracted"
    package = subject.extract_and_validate_package(
        archive_path=quality_package["archive"],
        expected_archive_sha256=str(quality_package["sha256"]),
        extraction_directory=extracted,
    )
    expected_ids = {
        *(f"primary-{index:02d}" for index in range(10)),
        *(f"confirmation-{index:02d}" for index in range(5)),
    }
    assert {job["job_id"] for job in package["jobs"]} == expected_ids
    for job_id in sorted(expected_ids):
        selected = subject.validate_selected_job(
            package=package,
            job_id=job_id,
            expected_job_manifest_sha256=_job_sha(extracted, job_id),
        )
        assert selected["job_id"] == job_id
        assert len(selected["root_paths"]) == (
            5 if job_id.startswith("primary-") else 1
        )

    # Re-extraction is a byte-identity check, not an overwrite.
    before = {
        path.relative_to(extracted).as_posix(): path.read_bytes()
        for path in extracted.rglob("*")
        if path.is_file()
    }
    subject.extract_and_validate_package(
        archive_path=quality_package["archive"],
        expected_archive_sha256=str(quality_package["sha256"]),
        extraction_directory=extracted,
    )
    after = {
        path.relative_to(extracted).as_posix(): path.read_bytes()
        for path in extracted.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_root_lookup_returns_mapping_and_artifacts_for_primary01_global_roots(
    quality_package: dict[str, object], tmp_path: Path
) -> None:
    extracted = tmp_path / "primary01-root-lookup"
    package = subject.extract_and_validate_package(
        archive_path=quality_package["archive"],
        expected_archive_sha256=str(quality_package["sha256"]),
        extraction_directory=extracted,
    )
    selected = subject.validate_selected_job(
        package=package,
        job_id="primary-01",
        expected_job_manifest_sha256=_job_sha(extracted, "primary-01"),
    )

    actual = subject._root_lookup(extracted, selected, package["plan"])

    assert isinstance(actual, tuple)
    assert len(actual) == 2
    lookup, root_artifacts = actual
    assert isinstance(lookup, dict)
    assert list(lookup) == list(range(10, 20))
    assert [
        index
        for artifact in root_artifacts
        for index in artifact["root_indices"]
    ] == list(range(10, 20))


def test_portable_archive_does_not_reopen_source_materialization_root(
    monkeypatch: pytest.MonkeyPatch,
    quality_package: dict[str, object],
    tmp_path: Path,
) -> None:
    materialization = json.loads(
        (
            Path(quality_package["directory"])
            / "control"
            / "materialization.json"
        ).read_text("ascii")
    )
    source_root = Path(materialization["root_directory"])
    relocated_root = source_root.with_name(f"{source_root.name}-relocated")
    source_root.rename(relocated_root)

    # This is the exact regression boundary.  build_job_descriptors performs
    # source-side seal validation and dereferences materialization.root_directory.
    # Portable replay must validate the packaged job bytes directly instead.
    monkeypatch.setattr(
        quality,
        "build_job_descriptors",
        lambda **_kwargs: pytest.fail(
            "portable package replay reopened the source materialization path"
        ),
    )
    package = subject.extract_and_validate_package(
        archive_path=quality_package["archive"],
        expected_archive_sha256=str(quality_package["sha256"]),
        extraction_directory=tmp_path / "portable-worker",
    )

    assert package["materialization"]["root_directory"] == str(source_root.resolve())
    assert not source_root.exists()
    assert relocated_root.is_dir()
    assert len(package["jobs"]) == 15


def test_portable_replay_is_tamper_closed_without_source_root(
    monkeypatch: pytest.MonkeyPatch,
    quality_package: dict[str, object],
    tmp_path: Path,
) -> None:
    extracted = tmp_path / "tampered-portable-worker"
    subject.extract_and_validate_package(
        archive_path=quality_package["archive"],
        expected_archive_sha256=str(quality_package["sha256"]),
        extraction_directory=extracted,
    )
    materialization = json.loads(
        (extracted / "control/materialization.json").read_text("ascii")
    )
    source_root = Path(materialization["root_directory"])
    relocated_root = source_root.with_name(f"{source_root.name}-relocated")
    source_root.rename(relocated_root)
    monkeypatch.setattr(
        quality,
        "build_job_descriptors",
        lambda **_kwargs: pytest.fail(
            "tamper validation reopened the source materialization path"
        ),
    )

    # Make the outer package hashes internally consistent after changing one
    # frozen job.  Validation must still fail at the semantic job-grid replay,
    # rather than succeeding merely because the envelope was rehashed.
    job_path = extracted / "jobs/primary-00.json"
    job = json.loads(job_path.read_text("ascii"))
    job["pair_indices"] = [5, 6, 7, 8, 9]
    job["root_paths"] = [
        f"roots/primary/pair_{index:03d}.json"
        for index in job["pair_indices"]
    ]
    job_path.write_bytes(subject.canonical_bytes(job))

    manifest_path = extracted / "PACKAGE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text("ascii"))
    job_record = next(
        record
        for record in manifest["entries"]
        if record["path"] == "jobs/primary-00.json"
    )
    job_record["bytes"] = job_path.stat().st_size
    job_record["sha256"] = subject.sha256_file(job_path)
    manifest["entry_aggregate_sha256"] = quality.canonical_sha256(
        manifest["entries"]
    )
    manifest_path.write_bytes(subject.canonical_bytes(manifest))

    ready_path = extracted / "PACKAGE_READY.json"
    ready = json.loads(ready_path.read_text("ascii"))
    ready["package_manifest_sha256"] = quality.canonical_sha256(manifest)
    ready["package_file_sha256"] = subject.sha256_file(manifest_path)
    ready_path.write_bytes(subject.canonical_bytes(ready))

    with pytest.raises(ValueError, match="frozen shard grid"):
        subject._portable_package_validation(extracted)


def test_worker_resume_is_byte_identical_and_done_is_last(
    monkeypatch: pytest.MonkeyPatch,
    quality_package: dict[str, object],
    tmp_path: Path,
) -> None:
    _patch_decision_oracle(monkeypatch)
    output = tmp_path / "output"
    job_id = "primary-00"
    first = subject.run_job(
        package_archive=quality_package["archive"],
        package_archive_sha256=str(quality_package["sha256"]),
        job_id=job_id,
        expected_job_manifest_sha256=_job_sha(
            Path(quality_package["directory"]), job_id
        ),
        library_path=CANDIDATE,
        output_directory=output,
        stop_after_roots=1,
        solve_row=_fake_row,
    )
    assert first["status"] == "interrupted_for_resume"
    retained = (output / "tasks/root_000.json").read_bytes()
    complete = subject.run_job(
        package_archive=quality_package["archive"],
        package_archive_sha256=str(quality_package["sha256"]),
        job_id=job_id,
        expected_job_manifest_sha256=_job_sha(
            Path(quality_package["directory"]), job_id
        ),
        library_path=CANDIDATE,
        output_directory=output,
        solve_row=_fake_row,
    )
    assert complete["status"] == "complete_validated_quality_job"
    assert (output / "tasks/root_000.json").read_bytes() == retained
    snapshot = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    replay = subject.run_job(
        package_archive=quality_package["archive"],
        package_archive_sha256=str(quality_package["sha256"]),
        job_id=job_id,
        expected_job_manifest_sha256=_job_sha(
            Path(quality_package["directory"]), job_id
        ),
        library_path=CANDIDATE,
        output_directory=output,
        solve_row=lambda *_args: pytest.fail("completed resume must not solve"),
    )
    assert replay == complete
    assert {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    } == snapshot
    done_mtime = (output / "DONE.json").stat().st_mtime_ns
    assert done_mtime >= max(
        path.stat().st_mtime_ns
        for path in output.rglob("*")
        if path.is_file() and path.name != "DONE.json"
    )


def test_worker_rejects_hidden_field_and_real_actionkey_oracle_is_called(
    monkeypatch: pytest.MonkeyPatch,
    quality_package: dict[str, object],
    tmp_path: Path,
) -> None:
    hidden = _fake_row

    def hidden_row(*args):
        row = hidden(*args)
        row["opponent_private_discards"] = []
        return row

    monkeypatch.setattr(
        subject.gate,
        "_validate_result_row",
        lambda row, *, job, root_lookup: {"row": row},
    )
    with pytest.raises(ValueError, match="hidden"):
        subject.run_job(
            package_archive=quality_package["archive"],
            package_archive_sha256=str(quality_package["sha256"]),
            job_id="primary-00",
            expected_job_manifest_sha256=_job_sha(
                Path(quality_package["directory"]), "primary-00"
            ),
            library_path=CANDIDATE,
            output_directory=tmp_path / "hidden",
            stop_after_roots=1,
            solve_row=hidden_row,
        )

    called = {"count": 0}

    def rejecting_action_key(row, *, job, root_lookup):
        called["count"] += 1
        raise ValueError("ActionKey mapping changed")

    monkeypatch.setattr(subject.gate, "_validate_result_row", rejecting_action_key)
    with pytest.raises(ValueError, match="ActionKey"):
        subject.run_job(
            package_archive=quality_package["archive"],
            package_archive_sha256=str(quality_package["sha256"]),
            job_id="primary-00",
            expected_job_manifest_sha256=_job_sha(
                Path(quality_package["directory"]), "primary-00"
            ),
            library_path=CANDIDATE,
            output_directory=tmp_path / "action-key",
            stop_after_roots=1,
            solve_row=_fake_row,
        )
    assert called["count"] == 1


def _wheelhouse(staging: Path) -> tuple[Path, Path]:
    wheel = b"local-smoke-wheel"
    wheel_name = "fresh_quality_smoke-1.0-py3-none-any.whl"
    archive = staging / "wheelhouse.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zipped:
        zipped.writestr(wheel_name, wheel)
    entries = [
        {
            "filename": wheel_name,
            "sha256": hashlib.sha256(wheel).hexdigest(),
            "bytes": len(wheel),
            "distribution": "fresh-quality-smoke",
            "version": "1.0",
            "tags": ["py3-none-any"],
        }
    ]
    manifest = {
        "schema": "hu_m31_t3_step6d_perfdev_v2_wheelhouse_v1",
        "status": "complete_hash_pinned_offline_wheelhouse",
        "requirements_sha256": "0" * 64,
        "python_abi": "cp311",
        "target_os": "linux",
        "target_architecture": "x86_64",
        "network_install_allowed": False,
        "entries": entries,
        "entry_count": 1,
        "entries_sha256": hashlib.sha256(
            subject.canonical_bytes(entries) + b"\n"
        ).hexdigest(),
    }
    manifest_path = staging / "wheelhouse_manifest.json"
    # Reuse the newline-framed convention of the accepted Run009 wheelhouse.
    manifest_path.write_bytes(subject.canonical_bytes(manifest) + b"\n")
    return archive, manifest_path


def _minimal_runtime_source(staging: Path) -> tuple[Path, Path]:
    """Small transport-only source fixture; production builder is tested above."""

    candidate = CANDIDATE.read_bytes()
    feature = FEATURE.read_bytes()
    payloads = {
        "pyproject.toml": b"[build-system]\n",
        "configs/hu_joint_policy_m31_t3_step6d_contract.json": b"{}\n",
        "src/ofc_regular/__init__.py": b"",
        "src/ofc_regular/hu_m31_t3_step6d_fresh_quality_v1.py": b"# smoke\n",
        "src/ofc_regular/hu_m31_t3_step6d_fresh_quality_gate_v1.py": b"# smoke\n",
        "src/ofc_regular/hu_m31_t3_step6d_fresh_quality_transport_v1.py": (
            b"# smoke\n"
        ),
        "src/ofc_regular/run_hu_m31_t3_step6c_shard.py": b"# smoke\n",
        "src/ofc_regular/ai_profiles.py": (
            ROOT / "src/ofc_regular/ai_profiles.py"
        ).read_bytes(),
        subject.DEFAULT_CANDIDATE_ARCHIVE_PATH: candidate,
        subject.DEFAULT_FEATURE_ARCHIVE_PATH: feature,
    }
    archive = staging / subject.SOURCE_ARCHIVE_NAME
    subject._zip_payloads(payloads, archive)
    entries = {
        name: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        for name, raw in sorted(payloads.items())
    }
    manifest = {
        "schema": subject.SOURCE_MANIFEST_SCHEMA,
        "status": "complete_hash_pinned_runtime_source",
        "archive": {
            "path": archive.name,
            "sha256": subject.sha256_file(archive),
            "bytes": archive.stat().st_size,
        },
        "entries": entries,
        "entry_count": len(entries),
        "entry_aggregate_sha256": subject.canonical_sha256(entries),
        "candidate_library": {
            "path": subject.DEFAULT_CANDIDATE_ARCHIVE_PATH,
            **entries[subject.DEFAULT_CANDIDATE_ARCHIVE_PATH],
        },
        "feature_encoder": {
            "path": subject.DEFAULT_FEATURE_ARCHIVE_PATH,
            **entries[subject.DEFAULT_FEATURE_ARCHIVE_PATH],
        },
        "profile_registry_sha256": quality.CURRENT_PROFILE_REGISTRY_SHA256,
        "content_addressed": True,
        "cloud_execution_started": False,
        "training_eligible": False,
        "current_profile_changed": False,
    }
    manifest_path = staging / subject.SOURCE_MANIFEST_NAME
    manifest_path.write_bytes(subject.canonical_bytes(manifest))
    assert subject.validate_runtime_source_manifest(
        manifest, archive_path=archive
    ) == manifest
    return archive, manifest_path


def test_source_launch_is_create_only_8_plus_7_and_tamper_closed(
    quality_package: dict[str, object], tmp_path: Path
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    package = staging / "fresh-quality.zip"
    shutil.copyfile(quality_package["archive"], package)
    source = staging / subject.SOURCE_ARCHIVE_NAME
    source_manifest = staging / subject.SOURCE_MANIFEST_NAME
    built_source = subject.create_runtime_source_archive(
        repository_root=ROOT,
        candidate_library_path=CANDIDATE,
        feature_encoder_path=FEATURE,
        archive_path=source,
        manifest_path=source_manifest,
    )
    assert built_source["candidate_library"]["sha256"] == (
        quality.ACCEPTED_CANDIDATE_LIBRARY_SHA256
    )
    source_copy = staging / "runtime-source-copy.zip"
    source_manifest_copy = staging / "runtime-source-copy.json"
    copied_source = subject.create_runtime_source_archive(
        repository_root=ROOT,
        candidate_library_path=CANDIDATE,
        feature_encoder_path=FEATURE,
        archive_path=source_copy,
        manifest_path=source_manifest_copy,
    )
    assert copied_source["archive"]["sha256"] == built_source["archive"]["sha256"]
    assert source_copy.read_bytes() == source.read_bytes()
    wheelhouse, wheel_manifest = _wheelhouse(staging)
    startup = staging / STARTUP.name
    shutil.copyfile(STARTUP, startup)
    launch_path = staging / "launch.json"
    launch = subject.build_local_launch_manifest(
        run_name="regular-hu-m31-fresh-quality-local-001",
        staging_directory=staging,
        quality_package_archive=package,
        runtime_source_archive=source,
        runtime_source_manifest_path=source_manifest,
        wheelhouse_archive=wheelhouse,
        wheelhouse_manifest_path=wheel_manifest,
        startup_script=startup,
        output_path=launch_path,
    )
    assert launch["wave_job_counts"] == [8, 7]
    assert [wave["job_count"] for wave in launch["waves"]] == [8, 7]
    assert launch["job_count"] == 15
    assert launch["cloud_launch_authorized"] is False
    assert subject.validate_local_launch_manifest(
        launch, staging_directory=staging
    ) == launch
    with pytest.raises(FileExistsError, match="create-only"):
        subject.build_local_launch_manifest(
            run_name=launch["run_name"],
            staging_directory=staging,
            quality_package_archive=package,
            runtime_source_archive=source,
            runtime_source_manifest_path=source_manifest,
            wheelhouse_archive=wheelhouse,
            wheelhouse_manifest_path=wheel_manifest,
            startup_script=startup,
            output_path=launch_path,
        )
    tampered = deepcopy(launch)
    tampered["jobs"][0]["job_manifest_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="selected job digest"):
        subject.validate_local_launch_manifest(
            tampered, staging_directory=staging
        )


def test_startup_bash_contract_and_all_15_job_bindings(
    quality_package: dict[str, object], tmp_path: Path
) -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is unavailable")
    syntax = subprocess.run(
        [bash, "-n", "scripts/startup_hu_m31_t3_step6d_fresh_quality_v1.sh"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        cwd=ROOT,
    )
    assert syntax.returncode == 0, syntax.stderr
    source_text = STARTUP.read_text("utf-8")
    assert "--no-index" in source_text
    assert "python3 -m venv --without-pip" in source_text
    assert 'PYTHONPATH="$PIP_BOOTSTRAP_WHEEL"' in source_text
    assert "PIP_BOOTSTRAP_WHEELS" in source_text
    assert "apt-get" not in source_text
    assert "RAYON_NUM_THREADS=16" in source_text
    assert "OFC_HU_M3_BATCH_THREADS=1" in source_text
    assert source_text.index("run-job") < source_text.rindex("DONE.json")
    for block in re.findall(r"<<'PY'\n(.*?)\nPY", source_text, flags=re.DOTALL):
        compile(block, str(STARTUP), "exec")

    staging = tmp_path / "startup-staging"
    staging.mkdir()
    package = staging / "fresh-quality.zip"
    shutil.copyfile(quality_package["archive"], package)
    source, source_manifest = _minimal_runtime_source(staging)
    wheelhouse, wheel_manifest = _wheelhouse(staging)
    startup = staging / STARTUP.name
    shutil.copyfile(STARTUP, startup)
    launch_path = staging / "launch.json"
    launch = subject.build_local_launch_manifest(
        run_name="regular-hu-m31-fresh-quality-startup-smoke-001",
        staging_directory=staging,
        quality_package_archive=package,
        runtime_source_archive=source,
        runtime_source_manifest_path=source_manifest,
        wheelhouse_archive=wheelhouse,
        wheelhouse_manifest_path=wheel_manifest,
        startup_script=startup,
        output_path=launch_path,
    )

    # The immutable Python replay is the platform-neutral startup preflight for
    # all fifteen jobs; bash syntax above separately verifies the shell layer.
    extracted = tmp_path / "all15-package"
    for row in launch["jobs"]:
        receipt = subject.validate_job_only(
            package_archive=package,
            package_archive_sha256=launch["quality_package"]["sha256"],
            job_id=row["job_id"],
            expected_job_manifest_sha256=row["job_manifest_sha256"],
            extraction_directory=extracted,
        )
        assert receipt["status"] == "selected_job_validated_not_executed"
        assert receipt["root_count"] == (
            10 if row["phase"] == quality.PRIMARY_PHASE else 2
        )

    def bash_path(path: Path) -> str:
        resolved = path.resolve()
        if "system32" in bash.casefold():
            drive = resolved.drive.rstrip(":").casefold()
            remainder = resolved.as_posix().split(":", 1)[1]
            return f"/mnt/{drive}{remainder}"
        return str(resolved)

    work = tmp_path / "startup-work"
    for row in launch["jobs"]:
        completed = subprocess.run(
            [
                bash,
                bash_path(startup),
                bash_path(staging),
                bash_path(launch_path),
                row["job_id"],
                bash_path(work),
                "validate-only",
            ],
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        assert completed.returncode == 0, (
            row["job_id"],
            completed.stdout,
            completed.stderr,
        )
        smoke = json.loads(completed.stdout.splitlines()[-1])
        assert smoke["job_id"] == row["job_id"]
        assert smoke["status"] == (
            "selected_job_transport_validated_not_executed"
        )


def test_offline_pip_wheel_bootstraps_venv_without_ensurepip(
    tmp_path: Path,
) -> None:
    """Exercise the worker bootstrap without invoking ensurepip or a network."""

    try:
        import ensurepip
    except ImportError:
        pytest.skip("test interpreter has no bundled pip wheel")
    bundled = Path(ensurepip.__file__).resolve().parent / "_bundled"
    candidates = sorted(bundled.glob("pip-*.whl"))
    if len(candidates) != 1:
        pytest.skip("test interpreter does not expose one bundled pip wheel")

    wheels = tmp_path / "wheels"
    wheels.mkdir()
    pip_wheel = wheels / candidates[0].name
    shutil.copyfile(candidates[0], pip_wheel)
    venv = tmp_path / "venv-without-pip"
    created = subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(venv)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert created.returncode == 0, created.stderr
    venv_python = (
        venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    )
    absent = subprocess.run(
        [str(venv_python), "-m", "pip", "--version"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert absent.returncode != 0

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(pip_wheel)
    bootstrapped = subprocess.run(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "--no-index",
            str(pip_wheel),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert bootstrapped.returncode == 0, bootstrapped.stderr
    installed = subprocess.run(
        [str(venv_python), "-m", "pip", "--version"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert installed.returncode == 0, installed.stderr
    assert str(venv) in installed.stdout
