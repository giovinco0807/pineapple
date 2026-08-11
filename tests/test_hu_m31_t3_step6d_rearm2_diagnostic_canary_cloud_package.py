from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_package as subject,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_plan as plan,
)
from ofc_regular import run_hu_m31_t3_step6d_performance_v2 as runner


@pytest.fixture(scope="module")
def package(tmp_path_factory: pytest.TempPathFactory) -> Path:
    target = tmp_path_factory.mktemp("r2diag-cloud-package") / "package"
    subject.build_package(output_dir=target)
    return target


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): subject.sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_package_has_exact_three_runner_manifests_and_twenty_development_roots(
    package: Path,
) -> None:
    manifest = subject.validate_package(package)
    assert manifest["logical_job_count"] == 3
    assert manifest["development_root_count"] == 20
    assert [row["job_id"] for row in manifest["job_manifests"]] == [
        "candidate-shard-00",
        "candidate-shard-01",
        "reference-shard-01",
    ]
    assert [row["stage_id"] for row in manifest["job_manifests"]] == [
        plan.STAGE1_ID,
        plan.STAGE2_ID,
        plan.STAGE2_ID,
    ]
    assert manifest["job_manifests"][1]["work_hand_indices"] == manifest[
        "job_manifests"
    ][2]["work_hand_indices"]
    assert not (
        set(manifest["job_manifests"][0]["work_hand_indices"])
        & set(manifest["job_manifests"][1]["work_hand_indices"])
    )
    for record in manifest["job_manifests"]:
        value = json.loads((package / record["path"]).read_text(encoding="utf-8"))
        assert set(value) == {
            "schema",
            "run_contract",
            "run_contract_digest",
            "source_role",
            "work_hand_indices",
        }
        assert runner.validate_shard_manifest(value) == value
        assert value["run_contract_digest"] == subject.EXPECTED_RUN_CONTRACT_DIGEST


def test_runtime_closure_native_elf_and_minimal_requirement_are_pinned(
    package: Path,
) -> None:
    manifest = subject.validate_package(package)
    assert manifest["runtime_closure_sha256"] == (
        "9894c508e028792d36238eced2a3370ea78acab03a00a1b467c92932f4f51ad6"
    )
    assert manifest["runtime_closure_records"] == (
        subject._accepted_runtime_closure_records()
    )
    assert manifest["python_allowlist"] == list(subject.PYTHON_ALLOWLIST)
    assert len(manifest["python_allowlist"]) == 33
    assert manifest["accepted_candidate"]["sha256"] == plan.EXPECTED_CANDIDATE_SHA256
    assert manifest["accepted_reference"]["sha256"] == plan.EXPECTED_REFERENCE_SHA256
    assert manifest["feature_encoder"]["sha256"] == (
        plan.EXPECTED_FEATURE_ENCODER_SHA256
    )
    with zipfile.ZipFile(package / subject.SOURCE_NAME) as archive:
        requirement = archive.read(subject.RUNTIME_REQUIREMENTS_RELATIVE)
        assert requirement == b"numpy==2.2.6\n"
        for relative in (
            subject.CANDIDATE_PACKAGE_PATH,
            subject.REFERENCE_PACKAGE_PATH,
            subject.FEATURE_PACKAGE_PATH,
        ):
            raw = archive.read(relative)
            assert raw[:4] == b"\x7fELF"
            assert raw[4:6] == b"\x02\x01"
            assert raw[16:20] == b"\x03\x00\x3e\x00"


def test_startup_smoke_is_poisoned_root_safe_seeds_exact_roots_and_is_immutable(
    package: Path,
) -> None:
    before = _tree_hashes(package)
    report = subject.startup_smoke(package)
    after = _tree_hashes(package)
    assert before == after
    assert report["status"] == "pass_local_precontent_only_cloud_not_authorized"
    assert report["root_members_opened_by_precontent_verifier"] == 0
    assert report["poison_root_guard_passed"] is True
    assert report["startup_entrypoint_job_count"] == 3
    assert report["root_seed_job_count"] == 3
    assert report["seeded_root_file_count"] == 30
    assert report["python_allowlist_compiled"] == 33
    assert report["isolated_runtime_closure_imported"] is True
    assert report["accepted_elf_count"] == 3
    assert report["gcloud_invoked"] is False
    assert report["remote_write_performed"] is False
    assert subject.validate_package(package)["source_sha256"] == report[
        "source_sha256"
    ]


def test_precontent_verifier_never_calls_zip_read_for_a_root(
    package: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = subject.validate_package(package)
    record = manifest["job_manifests"][0]
    verifier = subject._load_verifier(package / subject.VERIFIER_NAME)
    original = zipfile.ZipFile.read
    opened: list[str] = []

    def guarded(
        archive: zipfile.ZipFile, name: str | zipfile.ZipInfo, *args: object, **kwargs: object
    ) -> bytes:
        value = name.filename if isinstance(name, zipfile.ZipInfo) else str(name)
        if value.startswith(subject.ROOT_PREFIX + "/"):
            raise AssertionError("pre-content verifier opened a root")
        opened.append(value)
        return original(archive, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "read", guarded)
    report = verifier.verify_package_precontent(
        source_path=package / subject.SOURCE_NAME,
        manifest_path=package / subject.MANIFEST_NAME,
        job_path=package / record["path"],
        expected_source_sha256=manifest["source_sha256"],
        expected_manifest_sha256=subject.sha256_file(
            package / subject.MANIFEST_NAME
        ),
        expected_job_sha256=record["sha256"],
        expected_job_id=record["job_id"],
        expected_stage_id=record["stage_id"],
        poison_root_reads=True,
    )
    assert report["root_members_opened"] == 0
    assert opened
    assert not any(value.startswith(subject.ROOT_PREFIX + "/") for value in opened)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("performance_lock_evidence", True),
        ("cloud_launch_authorized", True),
        ("training_eligible", True),
    ],
)
def test_evidence_or_authorization_tamper_fails_closed(
    package: Path, tmp_path: Path, field: str, value: object
) -> None:
    target = tmp_path / field
    shutil.copytree(package, target)
    manifest_path = target / subject.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_bytes(subject.canonical_bytes(manifest))
    with pytest.raises(ValueError):
        subject.validate_package(target)


def test_hidden_field_extra_file_and_job_tamper_fail_closed(
    package: Path, tmp_path: Path
) -> None:
    hidden = tmp_path / "hidden"
    shutil.copytree(package, hidden)
    manifest_path = hidden / subject.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["opponent_hidden_payload"] = ["As"]
    manifest_path.write_bytes(subject.canonical_bytes(manifest))
    with pytest.raises(ValueError):
        subject.validate_package(hidden)

    extra = tmp_path / "extra"
    shutil.copytree(package, extra)
    (extra / "unexpected.txt").write_text("not allowed", encoding="utf-8")
    with pytest.raises(ValueError, match="file set"):
        subject.validate_package(extra)

    changed_job = tmp_path / "job"
    shutil.copytree(package, changed_job)
    path = changed_job / "jobs" / "candidate-shard-00.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["serialized_hidden_payload"] = {"opponent_private_discard": "As"}
    path.write_bytes(subject.canonical_bytes(value))
    with pytest.raises(ValueError):
        subject.validate_package(changed_job)


def test_archive_symlink_member_fails_closed(package: Path, tmp_path: Path) -> None:
    manifest = subject.validate_package(package)
    contract = plan.validate_frozen_contract()
    source = package / subject.SOURCE_NAME
    tampered = tmp_path / "symlink.zip"
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(
        tampered, "x", compression=zipfile.ZIP_DEFLATED
    ) as changed:
        for old in original.infolist():
            info = zipfile.ZipInfo(old.filename, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (
                0o120777 << 16
                if old.filename == "src/ofc_regular/__init__.py"
                else 0o100644 << 16
            )
            changed.writestr(info, original.read(old.filename))
    with pytest.raises(ValueError, match="symlink|unsafe"):
        subject._validate_archive(
            source=tampered,
            manifest=manifest,
            contract=contract,
        )


def test_dirty_runtime_closure_cannot_self_manifest(tmp_path: Path) -> None:
    for relative, _size, _digest in subject.RUNTIME_CLOSURE_ACCEPTED:
        source = subject._REPO_ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    changed = tmp_path / "src/ofc_regular/action_key.py"
    changed.write_bytes(changed.read_bytes() + b"\n# dirty\n")
    with pytest.raises(ValueError, match="runtime closure changed"):
        subject._validate_runtime_closure(tmp_path)


def test_package_api_stops_before_cloud_authorization_or_launch(package: Path) -> None:
    manifest = subject.validate_package(package)
    assert manifest["cloud_worker_payload_complete"] is True
    assert manifest["cloud_executable"] is False
    assert manifest["launch_ready"] is False
    assert manifest["host_prerequisites"]["architecture"] == "x86_64"
    assert manifest["host_prerequisites"]["glibc_minimum"] == "2.34"
    assert manifest["host_prerequisites"]["python_minimum"] == "3.10"
    assert "libgcc-s1" in manifest["host_prerequisites"]["debian_packages"]
    assert manifest["host_prerequisites"]["gce_metadata_bootstrap_included"] is False
    assert manifest["host_prerequisites"]["object_download_transport_included"] is False
    assert manifest["host_prerequisites"]["remote_upload_transport_included"] is False
    assert manifest["rearm2_production_package_reused"] is False
    assert manifest["rearm2_production_source_opened"] is False
    assert manifest["rearm2_locked_roots_used"] is False
    assert manifest["all20_launcher_reused"] is False
    assert not hasattr(subject, "launch")
    assert not hasattr(subject, "authorize_launch")
    assert not hasattr(subject, "write_claim")
    assert not hasattr(subject, "upload")
    assert subject.sha256_file(plan.DEFAULT_CURRENT_PROFILE) == (
        plan.EXPECTED_CURRENT_PROFILE_FILE_SHA256
    )
