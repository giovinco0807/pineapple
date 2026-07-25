from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import pytest

from ofc_regular import hu_m31_t3_step6d_full100_wave_package_v2 as subject
from ofc_regular import hu_m31_t3_step6d_full100_wave_plan_v2 as wave_v2
from ofc_regular import (
    hu_m31_t3_step6d_candidate02_performance_lock_v4_plan as lock_v4_plan,
)
from ofc_regular import (
    hu_m31_t3_step6d_performance_lock_v4_spot_package as lock_v4_package,
)


RUN_NAME = "regular-hu-m31-c02-f100wv2-20260722-010"
SALT = "123456789abcdef0123456789abcdef0"
IMAGE = "sha256:" + "3" * 64


def _sha(raw: bytes | str) -> str:
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture()
def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    science_root = tmp_path / "science"
    science_root.mkdir()
    source_name = subject.scientific.SOURCE_NAME
    source_raw = b"frozen-scientific-payload"
    (science_root / source_name).write_bytes(source_raw)
    (science_root / "manifest.json").write_bytes(b"{}")
    (science_root / "PACKAGE_READY.json").write_bytes(b"{}")
    plan = wave_v2.build_wave_plan(
        run_name=RUN_NAME,
        identity_salt=SALT,
        package_sha256=_sha(source_raw),
        image_digest=IMAGE,
    )
    job_records = []
    for frozen in plan["full100_plan"]["jobs"]:
        relative = f"jobs/{frozen['job_id']}.json"
        raw = subject.scientific.canonical_bytes(
            subject.scientific._job_manifest(plan["full100_plan"], frozen)
        )
        assert _sha(raw) == frozen["shard_manifest_sha256"]
        path = science_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        job_records.append(
            {
                "job_id": frozen["job_id"],
                "source_role": frozen["source_role"],
                "shard_index": frozen["shard_index"],
                "work_hand_indices": frozen["work_hand_indices"],
                "path": relative,
                "output_prefix": f"jobs/{frozen['job_id']}",
                "sha256": _sha(raw),
                "bytes": len(raw),
            }
        )
    science_manifest = {
        "schema": subject.scientific.PACKAGE_SCHEMA,
        "run_name": "regular-hu-m31-c02-full100-dev-20260717-002",
        "source_name": source_name,
        "source_sha256": _sha(source_raw),
        "source_bytes": len(source_raw),
        "plan_sha256": plan["full100_plan_sha256"],
        "run_contract_digest": plan["run_contract_digest"],
        "job_manifests": job_records,
    }
    (science_root / "manifest.json").write_bytes(
        subject.scientific.canonical_bytes(science_manifest)
    )
    monkeypatch.setattr(
        subject.scientific,
        "validate_package",
        lambda path: deepcopy(science_manifest),
    )
    wheelhouse = tmp_path / "wheelhouse.zip"
    wheelhouse.write_bytes(b"offline-wheelhouse")
    wheelhouse_manifest = tmp_path / "wheelhouse_manifest.json"
    wheelhouse_manifest.write_bytes(
        subject.perf_cloud.canonical_bytes(
            {
                "schema": "test-wheelhouse",
                "status": "fixture",
                "entries": [],
                "requirements_sha256": "4" * 64,
            }
        )
    )
    monkeypatch.setattr(
        subject.perf_cloud,
        "_validate_wheelhouse_archive",
        lambda archive, manifest: None,
    )
    startup = tmp_path / "startup.sh"
    startup.write_bytes(b"#!/usr/bin/env bash\nset -euo pipefail\n")
    expected_startup_sha256 = _sha(startup.read_bytes())
    baseline = wave_v2.build_observed_transition(
        plan,
        project_id="ofc-project-123",
        zone="asia-northeast1-b",
        observed_at_utc="2026-07-22T00:00:00Z",
        previous_transition_digest=None,
        attempt_history=wave_v2.empty_attempt_history(plan),
    )
    ledger = wave_v2.build_attempt_ledger(plan, transitions=[baseline])
    resume = wave_v2.build_resume_plan(plan, attempt_ledger=ledger)
    return {
        "science_root": science_root,
        "science_manifest": science_manifest,
        "startup": startup,
        "expected_startup_sha256": expected_startup_sha256,
        "wheelhouse": wheelhouse,
        "wheelhouse_manifest": wheelhouse_manifest,
        "plan": plan,
        "ledger": ledger,
        "resume": resume,
        "tmp": tmp_path,
    }


def _build(fixture: dict) -> dict:
    return subject.build_outer_manifest(
        scientific_package_dir=fixture["science_root"],
        startup_script=fixture["startup"],
        wheelhouse_archive=fixture["wheelhouse"],
        wheelhouse_manifest=fixture["wheelhouse_manifest"],
        expected_startup_sha256=fixture["expected_startup_sha256"],
        wave_plan=fixture["plan"],
    )


def _materialize(fixture: dict, destination: Path) -> dict:
    return subject.materialize_outer_package(
        output_dir=destination,
        scientific_package_dir=fixture["science_root"],
        startup_script=fixture["startup"],
        wheelhouse_archive=fixture["wheelhouse"],
        wheelhouse_manifest=fixture["wheelhouse_manifest"],
        expected_startup_sha256=fixture["expected_startup_sha256"],
        wave_plan=fixture["plan"],
    )


def _validate_package(fixture: dict, destination: Path) -> dict:
    return subject.validate_outer_package(
        destination,
        fixture["plan"],
        expected_startup_sha256=fixture["expected_startup_sha256"],
    )


def _validate_manifest(fixture: dict, manifest: dict) -> dict:
    return subject.validate_outer_manifest(
        fixture["plan"],
        manifest,
        expected_startup_sha256=fixture["expected_startup_sha256"],
    )


def _reseal_manifest(manifest: dict) -> dict:
    """Recompute all attacker-controlled self-digests after a forged edit."""

    forged = deepcopy(manifest)
    payload_sha = subject.canonical_sha256(
        subject._payload_records(forged["entries"])
    )
    prefix = f"{subject.CONTENT_PREFIX_ROOT}/{payload_sha}"
    forged["content_payload_sha256"] = payload_sha
    forged["content_prefix"] = prefix
    forged["entries"] = subject._bind_object_names(forged["entries"], prefix)
    forged["manifest_sha256"] = subject.canonical_sha256(
        {
            key: value
            for key, value in forged.items()
            if key != "manifest_sha256"
        }
    )
    return forged


def test_outer_manifest_binds_science_wave_startup_and_all_jobs(fixture: dict) -> None:
    manifest = _build(fixture)
    assert manifest["entry_count"] == 26
    assert manifest["legacy_launcher_authorized"] is False
    assert manifest["scientific_lineage"]["launcher_reuse_forbidden"] is True
    assert manifest["runtime_binding"]["package_sha256"] == fixture["science_manifest"]["source_sha256"]
    assert [row["job_id"] for row in manifest["entries"][6:]] == fixture["plan"]["coverage"]["job_ids"]
    assert len({row["object_name"] for row in manifest["entries"]}) == 26
    assert manifest["manifest_sha256"] == (
        "806e29ce1e7f2bd3d1c3404124423814ce3fe3c6675c52b99d6bb8b25b669bb3"
    )
    assert len(subject.canonical_bytes(manifest)) == 13840
    assert _sha(subject.canonical_bytes(manifest)) == (
        "80728bdced45ce659f93bd8c5e173dbaf82e86dbf7479a0e81ca91c0df9041ec"
    )
    assert _validate_manifest(fixture, manifest) == manifest


def test_outer_manifest_rejects_mixed_science_package_schema(
    fixture: dict,
) -> None:
    fixture["science_manifest"]["schema"] = (
        subject.science_registry.PERFORMANCE_LOCK_V4_PACKAGE_SCHEMA
    )
    with pytest.raises(ValueError, match="package manifest facade changed"):
        _build(fixture)


def test_outer_manifest_rejects_package_module_identity_drift(
    fixture: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        subject.scientific,
        "PACKAGE_SCHEMA",
        subject.science_registry.PERFORMANCE_LOCK_V4_PACKAGE_SCHEMA,
    )
    with pytest.raises(ValueError, match="package facade identity changed"):
        _build(fixture)


def test_v4_wave_uses_only_the_v4_package_facade(
    fixture: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_plan = deepcopy(fixture["plan"]["full100_plan"])
    lock_plan.update(
        {
            "schema": lock_v4_plan.PLAN_SCHEMA,
            "scope": lock_v4_plan.PLAN_SCOPE,
            "status": lock_v4_plan.PLAN_STATUS,
            "decision": lock_v4_plan.PLAN_DECISION,
        }
    )
    lock_plan_sha = wave_v2.canonical_sha256(lock_plan)
    lock_contract_digest = wave_v2.canonical_sha256(lock_plan["run_contract"])
    monkeypatch.setattr(lock_v4_plan, "PLAN_SHA256", lock_plan_sha)
    monkeypatch.setattr(
        lock_v4_plan, "RUN_CONTRACT_DIGEST", lock_contract_digest
    )
    monkeypatch.setattr(
        lock_v4_plan,
        "validate_performance_lock_v4_plan",
        lambda value: deepcopy(dict(value)),
    )

    source_raw = (
        fixture["science_root"] / subject.scientific.SOURCE_NAME
    ).read_bytes()
    lock_wave = wave_v2.build_wave_plan(
        run_name=RUN_NAME,
        identity_salt=SALT,
        package_sha256=_sha(source_raw),
        image_digest=IMAGE,
        full100_plan=lock_plan,
        execution_scope=wave_v2.PERFORMANCE_LOCK_V4_EXECUTION_SCOPE,
    )
    lock_root = fixture["tmp"] / "lock-v4-science"
    lock_root.mkdir()
    (lock_root / lock_v4_package.SOURCE_NAME).write_bytes(source_raw)
    (lock_root / lock_v4_package.READY_NAME).write_bytes(b"{}\n")
    for record in fixture["science_manifest"]["job_manifests"]:
        source = fixture["science_root"] / record["path"]
        target = lock_root / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    lock_manifest = {
        "schema": lock_v4_package.PACKAGE_SCHEMA,
        "run_name": "regular-hu-m31-c02-performance-lock-v4",
        "source_name": lock_v4_package.SOURCE_NAME,
        "source_sha256": _sha(source_raw),
        "source_bytes": len(source_raw),
        "plan_sha256": lock_plan_sha,
        "run_contract_digest": lock_contract_digest,
        "job_manifests": [
            {
                key: value
                for key, value in record.items()
                if key != "output_prefix"
            }
            for record in fixture["science_manifest"]["job_manifests"]
        ],
        "v4_extra_evidence": {"ignored_by_outer_transport": True},
    }
    (lock_root / lock_v4_package.MANIFEST_NAME).write_bytes(
        lock_v4_package.canonical_bytes(lock_manifest)
    )
    monkeypatch.setattr(
        lock_v4_package,
        "validate_package",
        lambda path: deepcopy(lock_manifest),
    )

    manifest = subject.build_outer_manifest(
        scientific_package_dir=lock_root,
        startup_script=fixture["startup"],
        wheelhouse_archive=fixture["wheelhouse"],
        wheelhouse_manifest=fixture["wheelhouse_manifest"],
        expected_startup_sha256=fixture["expected_startup_sha256"],
        wave_plan=lock_wave,
    )
    assert manifest["entry_count"] == 26
    assert manifest["full100_plan_sha256"] == lock_plan_sha
    assert manifest["run_contract_digest"] == lock_contract_digest
    assert manifest["scientific_lineage"]["source_sha256"] == _sha(source_raw)
    destination = fixture["tmp"] / "lock-v4-outer"
    materialized = subject.materialize_outer_package(
        output_dir=destination,
        scientific_package_dir=lock_root,
        startup_script=fixture["startup"],
        wheelhouse_archive=fixture["wheelhouse"],
        wheelhouse_manifest=fixture["wheelhouse_manifest"],
        expected_startup_sha256=fixture["expected_startup_sha256"],
        wave_plan=lock_wave,
    )
    assert materialized == manifest


def test_real_v4_source_package_materializes_and_validates_outer_transport(
    fixture: dict,
) -> None:
    science_root = lock_v4_package.DEFAULT_PACKAGE_DIR
    science = lock_v4_package.validate_package(science_root)
    lock_plan = lock_v4_plan.validate_performance_lock_v4_plan(
        wave_v2._read_frozen_plan(lock_v4_plan.DEFAULT_PLAN_PATH)
    )
    lock_wave = wave_v2.build_wave_plan(
        run_name=RUN_NAME,
        identity_salt=SALT,
        package_sha256=science["source_sha256"],
        image_digest=lock_plan["source_identity"]["image_digest"],
        full100_plan=lock_plan,
    )
    descriptor = subject.science_registry.descriptor_for_wave_plan(lock_wave)
    startup = descriptor.resolved_startup_path()
    destination = fixture["tmp"] / "real-v4-outer"
    materialized = subject.materialize_outer_package(
        output_dir=destination,
        scientific_package_dir=science_root,
        startup_script=startup,
        wheelhouse_archive=fixture["wheelhouse"],
        wheelhouse_manifest=fixture["wheelhouse_manifest"],
        expected_startup_sha256=descriptor.startup_sha256,
        wave_plan=lock_wave,
    )
    assert subject.validate_outer_package(
        destination,
        lock_wave,
        expected_startup_sha256=descriptor.startup_sha256,
    ) == materialized
    assert materialized["entry_count"] == 26
    assert materialized["expected_startup_sha256"] == descriptor.startup_sha256
    assert all(
        "output_prefix" not in record for record in science["job_manifests"]
    )


def test_outer_manifest_rejects_scientific_payload_mismatch(fixture: dict) -> None:
    fixture["science_manifest"]["source_sha256"] = "4" * 64
    with pytest.raises(ValueError, match="scientific payload"):
        subject.build_outer_manifest(
            scientific_package_dir=fixture["science_root"],
            startup_script=fixture["startup"],
            wheelhouse_archive=fixture["wheelhouse"],
            wheelhouse_manifest=fixture["wheelhouse_manifest"],
            expected_startup_sha256=fixture["expected_startup_sha256"],
            wave_plan=fixture["plan"],
        )


def test_materialized_package_is_create_only_and_tamper_evident(fixture: dict) -> None:
    destination = fixture["tmp"] / "outer"
    manifest = _materialize(fixture, destination)
    assert _validate_package(fixture, destination) == manifest
    with pytest.raises(FileExistsError):
        subject.materialize_outer_package(
            output_dir=destination,
            scientific_package_dir=fixture["science_root"],
            startup_script=fixture["startup"],
            wheelhouse_archive=fixture["wheelhouse"],
            wheelhouse_manifest=fixture["wheelhouse_manifest"],
            expected_startup_sha256=fixture["expected_startup_sha256"],
            wave_plan=fixture["plan"],
        )
    first_job = destination / manifest["entries"][6]["relative_path"]
    first_job.write_bytes(first_job.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="content changed"):
        _validate_package(fixture, destination)


def test_job_bootstrap_is_one_vm_one_job_and_attempt_bound(fixture: dict) -> None:
    manifest = _build(fixture)
    selected = fixture["resume"]["selected_attempts"]
    assert len(selected) == 8
    metadata = subject.build_job_bootstrap_metadata(
        wave_plan=fixture["plan"],
        attempt_ledger=fixture["ledger"],
        resume_plan=fixture["resume"],
        outer_manifest=manifest,
        job_id=selected[0]["job_id"],
        bucket="pokerhu-ofc-solver-485418-training",
        worker_principal="ofc-m31-t3-worker@ofc-solver-485418.iam.gserviceaccount.com",
        prelaunch_authorization_sha256="5" * 64,
        expected_startup_sha256=fixture["expected_startup_sha256"],
    )
    assert metadata["instance_name"] == selected[0]["instance_id"]
    assert metadata["attempt_id"] == "a00"
    assert metadata["one_vm_one_job_one_role"] is True
    assert metadata["additional_create_authorized"] is False
    assert metadata["hidden_truth_exposed"] is False
    assert subject.validate_job_bootstrap_metadata(
        fixture["plan"], fixture["ledger"], fixture["resume"], manifest, metadata,
        expected_startup_sha256=fixture["expected_startup_sha256"],
    ) == metadata


def test_job_bootstrap_rejects_unselected_job_and_mapping_tamper(fixture: dict) -> None:
    manifest = _build(fixture)
    later_job = fixture["plan"]["waves"][1]["job_ids"][0]
    with pytest.raises(ValueError, match="not selected"):
        subject.build_job_bootstrap_metadata(
            wave_plan=fixture["plan"],
            attempt_ledger=fixture["ledger"],
            resume_plan=fixture["resume"],
            outer_manifest=manifest,
            job_id=later_job,
            bucket="pokerhu-ofc-solver-485418-training",
            worker_principal="ofc-m31-t3-worker@ofc-solver-485418.iam.gserviceaccount.com",
            prelaunch_authorization_sha256="5" * 64,
            expected_startup_sha256=fixture["expected_startup_sha256"],
        )
    job = fixture["resume"]["selected_attempts"][0]["job_id"]
    metadata = subject.build_job_bootstrap_metadata(
        wave_plan=fixture["plan"],
        attempt_ledger=fixture["ledger"],
        resume_plan=fixture["resume"],
        outer_manifest=manifest,
        job_id=job,
        bucket="pokerhu-ofc-solver-485418-training",
        worker_principal="ofc-m31-t3-worker@ofc-solver-485418.iam.gserviceaccount.com",
        prelaunch_authorization_sha256="5" * 64,
        expected_startup_sha256=fixture["expected_startup_sha256"],
    )
    tampered = deepcopy(metadata)
    tampered["instance_name"] = fixture["resume"]["selected_attempts"][1]["instance_id"]
    tampered["bootstrap_sha256"] = subject.canonical_sha256(
        {key: value for key, value in tampered.items() if key != "bootstrap_sha256"}
    )
    with pytest.raises(ValueError, match="boundary changed"):
        subject.validate_job_bootstrap_metadata(
            fixture["plan"], fixture["ledger"], fixture["resume"], manifest, tampered,
            expected_startup_sha256=fixture["expected_startup_sha256"],
        )


def test_manifest_rejects_reauthorizing_legacy_launcher(fixture: dict) -> None:
    manifest = _build(fixture)
    tampered = deepcopy(manifest)
    tampered["legacy_launcher_authorized"] = True
    tampered["manifest_sha256"] = subject.canonical_sha256(
        {key: value for key, value in tampered.items() if key != "manifest_sha256"}
    )
    with pytest.raises(ValueError, match="boundary changed"):
        _validate_manifest(fixture, tampered)


def test_manifest_rederives_source_lineage_and_runtime(fixture: dict) -> None:
    manifest = _build(fixture)
    forged = deepcopy(manifest)
    forged["entries"][0]["sha256"] = "a" * 64
    forged["scientific_lineage"]["source_sha256"] = "a" * 64
    forged = _reseal_manifest(forged)
    with pytest.raises(ValueError, match="boundary changed"):
        _validate_manifest(fixture, forged)

    forged = deepcopy(manifest)
    forged["entries"][1]["sha256"] = "b" * 64
    forged = _reseal_manifest(forged)
    with pytest.raises(ValueError, match="job mapping changed"):
        _validate_manifest(fixture, forged)


@pytest.mark.parametrize(
    "field",
    ("wave_plan", "job_sha256", "source_role", "shard_index", "work_hands"),
)
def test_manifest_rederives_canonical_plan_and_job_contract(
    fixture: dict, field: str
) -> None:
    forged = deepcopy(_build(fixture))
    if field == "wave_plan":
        forged["entries"][5]["sha256"] = "c" * 64
    elif field == "job_sha256":
        forged["entries"][6]["sha256"] = "d" * 64
    elif field == "source_role":
        forged["entries"][6]["source_role"] = "reference"
    elif field == "shard_index":
        forged["entries"][6]["shard_index"] += 1
    else:
        forged["entries"][6]["work_hand_indices"] = list(
            reversed(forged["entries"][6]["work_hand_indices"])
        )
    forged = _reseal_manifest(forged)
    with pytest.raises(ValueError, match="job mapping changed"):
        _validate_manifest(fixture, forged)


@pytest.mark.parametrize("entry_index", (2, 3))
def test_manifest_rederives_wheelhouse_archive_and_manifest_binding(
    fixture: dict, entry_index: int
) -> None:
    forged = deepcopy(_build(fixture))
    forged["entries"][entry_index]["sha256"] = "e" * 64
    forged = _reseal_manifest(forged)
    with pytest.raises(ValueError, match="job mapping changed"):
        _validate_manifest(fixture, forged)


def test_expected_startup_hash_is_required_and_rederived(fixture: dict) -> None:
    with pytest.raises(ValueError, match="does not match expected hash"):
        subject.build_outer_manifest(
            scientific_package_dir=fixture["science_root"],
            startup_script=fixture["startup"],
            wheelhouse_archive=fixture["wheelhouse"],
            wheelhouse_manifest=fixture["wheelhouse_manifest"],
            expected_startup_sha256="f" * 64,
            wave_plan=fixture["plan"],
        )

    manifest = _build(fixture)
    with pytest.raises(ValueError, match="boundary changed"):
        subject.validate_outer_manifest(
            fixture["plan"],
            manifest,
            expected_startup_sha256="f" * 64,
        )


def test_build_rejects_startup_below_symlinked_parent(fixture: dict) -> None:
    real_parent = fixture["tmp"] / "real-startup"
    real_parent.mkdir()
    real_startup = real_parent / "startup.sh"
    real_startup.write_bytes(fixture["startup"].read_bytes())
    linked_parent = fixture["tmp"] / "linked-startup"
    try:
        linked_parent.symlink_to(real_parent, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")
    with pytest.raises(ValueError, match="symlink or junction"):
        subject.build_outer_manifest(
            scientific_package_dir=fixture["science_root"],
            startup_script=linked_parent / "startup.sh",
            wheelhouse_archive=fixture["wheelhouse"],
            wheelhouse_manifest=fixture["wheelhouse_manifest"],
            expected_startup_sha256=fixture["expected_startup_sha256"],
            wave_plan=fixture["plan"],
        )


def test_exact_tree_rejects_extra_file_and_directory(fixture: dict) -> None:
    file_destination = fixture["tmp"] / "outer-extra-file"
    _materialize(fixture, file_destination)
    (file_destination / "launch_authorization.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="extra or missing paths"):
        _validate_package(fixture, file_destination)

    directory_destination = fixture["tmp"] / "outer-extra-directory"
    _materialize(fixture, directory_destination)
    (directory_destination / "unexpected").mkdir()
    with pytest.raises(ValueError, match="extra or missing paths"):
        _validate_package(fixture, directory_destination)


def test_validate_rejects_symlinked_manifest(fixture: dict) -> None:
    destination = fixture["tmp"] / "outer-manifest-link"
    _materialize(fixture, destination)
    manifest_path = destination / subject.MANIFEST_NAME
    external = fixture["tmp"] / "external-manifest.json"
    external.write_bytes(manifest_path.read_bytes())
    manifest_path.unlink()
    try:
        manifest_path.symlink_to(external)
    except OSError as exc:
        pytest.skip(f"file symlinks are unavailable: {exc}")
    with pytest.raises(ValueError, match="symlink or junction"):
        _validate_package(fixture, destination)


def test_stage_is_validated_before_publish_and_cleaned_on_failure(
    fixture: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = fixture["tmp"] / "outer-stage-rejected"
    published = False

    def reject_stage(*args, **kwargs):
        package_dir = Path(args[0])
        assert package_dir.name.endswith(".staging")
        raise ValueError("injected stage validation failure")

    def unexpected_publish(stage: Path, target: Path) -> None:
        nonlocal published
        published = True

    monkeypatch.setattr(subject, "validate_outer_package", reject_stage)
    monkeypatch.setattr(subject, "_publish_no_replace", unexpected_publish)
    with pytest.raises(ValueError, match="injected stage validation failure"):
        _materialize(fixture, destination)
    assert published is False
    assert not destination.exists()
    assert not list(fixture["tmp"].glob(".*.staging"))


def test_publish_race_preserves_competing_destination_and_cleans_stage(
    fixture: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = fixture["tmp"] / "outer-publish-race"
    original_publish = subject._publish_no_replace

    def inject_competing_destination(stage: Path, target: Path) -> None:
        target.mkdir()
        (target / "owner.txt").write_text("competing creator", encoding="utf-8")
        original_publish(stage, target)

    monkeypatch.setattr(
        subject, "_publish_no_replace", inject_competing_destination
    )
    with pytest.raises(FileExistsError, match="immutable"):
        _materialize(fixture, destination)
    assert (destination / "owner.txt").read_text(encoding="utf-8") == (
        "competing creator"
    )
    assert not list(fixture["tmp"].glob(".*.staging"))


def test_stage_and_published_package_are_both_validated(
    fixture: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = fixture["tmp"] / "outer-double-validated"
    original_validate = subject.validate_outer_package
    validated_paths: list[Path] = []

    def record_validation(*args, **kwargs):
        validated_paths.append(Path(args[0]))
        return original_validate(*args, **kwargs)

    monkeypatch.setattr(subject, "validate_outer_package", record_validation)
    _materialize(fixture, destination)
    assert len(validated_paths) == 2
    assert validated_paths[0].name.endswith(".staging")
    assert validated_paths[1] == destination
