from __future__ import annotations

import copy
import json
import shutil
import zipfile
from pathlib import Path

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_candidate02_performance_lock_v4_plan as v4,
)
from ofc_regular import (
    hu_m31_t3_step6d_performance_lock_v4_spot_package as subject,
)
from ofc_regular import run_hu_m31_t3_step6d_performance_v2 as runner


REPO_ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_ROOTS = (
    REPO_ROOT
    / "outputs/hu_joint_policy/m31_t3_step6d/candidate02_development/"
    "tail_reselection_v2/roots"
)
V4_CONTRACT = runner.build_run_contract(
    candidate_library_sha256=v4.CANDIDATE_LIBRARY_SHA256,
    reference_library_sha256=v4.REFERENCE_LIBRARY_SHA256,
    variant=runner.CANDIDATE02_PERFORMANCE_LOCK_V4_VARIANT,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(subject.canonical_bytes(value))


def _v4_root(index: int) -> dict:
    base = json.loads(
        (DEVELOPMENT_ROOTS / f"hand_{index:03d}.json").read_text(encoding="utf-8")
    )
    row = runner.candidate02_performance_lock_v4_schedule_row(index)
    value = copy.deepcopy(base)
    value.update(
        {
            "schema": runner.CANDIDATE02_PERFORMANCE_LOCK_V4_ROOT_SCHEMA,
            "contract_canonical_sha256": (
                runner._candidate02_performance_lock_v4_contract_anchor_sha256()
            ),
            "schedule": runner.CANDIDATE02_PERFORMANCE_LOCK_V4_SCHEDULE,
            "schedule_row_sha256": runner.canonical_sha256(row),
            "hand_index": index,
            "root_indices": row["root_indices"],
            "profile": row["profile"],
            "seeds": row["seeds"],
            "budget": row["budget"],
            "current_profile_resolved": False,
            "opponent_private_discards_used": False,
            "training_eligible": False,
        }
    )
    runner._validate_root_artifact(
        V4_CONTRACT,
        value,
        index=index,
    )
    return value


@pytest.fixture(scope="module")
def valid_package(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("lock-v4-package")
    evidence = root / "evidence"
    output = evidence / "execution"
    roots_dir = output / "roots"
    roots_dir.mkdir(parents=True)
    plan_value = v4.build_performance_lock_v4_plan()
    plan_path = evidence / "plan.json"
    _write(plan_path, plan_value)
    claim = v4._claim_value(output)
    _write(output / v4.CLAIM_NAME, claim)
    roots = []
    for index in range(100):
        value = _v4_root(index)
        roots.append(value)
        _write(roots_dir / f"hand_{index:03d}.json", value)
    audit = v4._audit_roots(
        roots=roots,
        contract=plan_value["run_contract"],
        current_profile_unchanged=True,
    )
    materialization = v4.validate_materialization_receipt(
        v4._materialization_value(
            output_dir=output,
            claim=claim,
            audit=audit,
        )
    )
    materialization_path = output / "MATERIALIZATION_RECEIPT.json"
    _write(materialization_path, materialization)
    seal = v4.validate_root_seal(
        v4._seal_value(
            output_dir=output,
            claim=claim,
            materialization=materialization,
            audit=audit,
        )
    )
    seal_path = output / "ROOT_SEAL.json"
    _write(seal_path, seal)
    package_dir = root / "package"
    manifest = subject.create_package(
        destination=package_dir,
        repository_root=REPO_ROOT,
        plan_path=plan_path,
        materialization_receipt_path=materialization_path,
        root_seal_path=seal_path,
        root_output_dir=output,
    )
    return {
        "root": root,
        "package": package_dir,
        "manifest": manifest,
        "plan": plan_value,
        "audit": audit,
    }


def _copy_package(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    return destination


def _resign_package_with_archive_payloads(
    package: Path, payloads: dict[str, bytes]
) -> None:
    source = package / subject.SOURCE_NAME
    source.unlink()
    subject._zip_bytes(payloads, source)
    manifest_path = package / subject.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, raw in payloads.items():
        manifest["source_entries"][name] = {
            "sha256": subject.sha256_bytes(raw),
            "bytes": len(raw),
        }
    manifest["source_sha256"] = subject.sha256_file(source)
    manifest["source_bytes"] = source.stat().st_size
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = subject.canonical_sha256(manifest)
    manifest_path.write_bytes(subject.canonical_bytes(manifest))
    ready_path = package / subject.READY_NAME
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    ready["package_manifest_sha256"] = subject.sha256_file(manifest_path)
    ready["source_sha256"] = manifest["source_sha256"]
    ready["source_bytes"] = manifest["source_bytes"]
    ready_path.write_bytes(subject.canonical_bytes(ready))


def test_package_facade_replays_sealed_100_roots_and_is_outer_compatible(
    valid_package: dict,
) -> None:
    manifest = subject.validate_package(valid_package["package"])

    assert manifest == valid_package["manifest"]
    assert manifest["schema"] == subject.PACKAGE_SCHEMA
    assert manifest["source_name"] == subject.SOURCE_NAME
    assert manifest["plan_sha256"] == v4.PLAN_SHA256
    assert manifest["run_contract_digest"] == v4.RUN_CONTRACT_DIGEST
    assert manifest["root_file_count"] == 100
    assert manifest["root_hash_aggregate_sha256"] == (
        valid_package["audit"]["root_hash_aggregate_sha256"]
    )
    assert len(manifest["job_manifests"]) == 20
    assert {
        "source_name",
        "source_sha256",
        "source_bytes",
        "plan_sha256",
        "run_contract_digest",
        "job_manifests",
        "run_name",
    }.issubset(manifest)
    assert all(
        manifest[field] is False
        for field in (
            "gcloud_invoked",
            "spot_execution_authorized",
            "cloud_started",
            "performance_lock_authorized",
            "quality_pilot_authorized",
            "training_eligible",
            "current_profile_changed",
            "named_profile_added",
            "runtime_policy_activated",
            "m31_complete",
        )
    )


def test_package_destination_is_write_once(valid_package: dict) -> None:
    with pytest.raises(FileExistsError, match="already exists"):
        subject.create_package(destination=valid_package["package"])


def test_package_rejects_source_job_and_extra_tree_tamper(
    valid_package: dict, tmp_path: Path
) -> None:
    source_tamper = _copy_package(
        valid_package["package"], tmp_path / "source-tamper"
    )
    source = source_tamper / subject.SOURCE_NAME
    raw = bytearray(source.read_bytes())
    raw[len(raw) // 2] ^= 1
    source.write_bytes(raw)
    with pytest.raises(ValueError):
        subject.validate_package(source_tamper)

    job_tamper = _copy_package(valid_package["package"], tmp_path / "job-tamper")
    job = next((job_tamper / subject.JOB_DIRECTORY).glob("*.json"))
    job.write_bytes(job.read_bytes() + b" ")
    with pytest.raises(ValueError):
        subject.validate_package(job_tamper)

    extra = _copy_package(valid_package["package"], tmp_path / "extra")
    (extra / "unexpected.txt").write_text("no", encoding="utf-8")
    with pytest.raises(ValueError, match="tree changed"):
        subject.validate_package(extra)


def test_resigned_archive_cannot_hide_root_or_seal_tamper(
    valid_package: dict, tmp_path: Path
) -> None:
    root_tamper = _copy_package(valid_package["package"], tmp_path / "root-tamper")
    with zipfile.ZipFile(root_tamper / subject.SOURCE_NAME, "r") as archive:
        payloads = {name: archive.read(name) for name in archive.namelist()}
    relative = subject.ROOT_ARCHIVE_TEMPLATE.format(index=0)
    root = json.loads(payloads[relative].decode("utf-8"))
    root["seeds"]["hand"] += 1
    payloads[relative] = subject.canonical_bytes(root)
    _resign_package_with_archive_payloads(root_tamper, payloads)
    with pytest.raises(ValueError):
        subject.validate_package(root_tamper)

    seal_tamper = _copy_package(valid_package["package"], tmp_path / "seal-tamper")
    with zipfile.ZipFile(seal_tamper / subject.SOURCE_NAME, "r") as archive:
        payloads = {name: archive.read(name) for name in archive.namelist()}
    seal = json.loads(payloads[subject.SEAL_ARCHIVE_PATH].decode("utf-8"))
    seal["seat_counts"]["first"] = 99
    payloads[subject.SEAL_ARCHIVE_PATH] = subject.canonical_bytes(seal)
    _resign_package_with_archive_payloads(seal_tamper, payloads)
    with pytest.raises(ValueError):
        subject.validate_package(seal_tamper)


def test_package_does_not_change_current_profile(valid_package: dict) -> None:
    assert subject.sha256_file(REPO_ROOT / "src/ofc_regular/ai_profiles.py") == (
        v4.CURRENT_PROFILE_REGISTRY_SHA256
    )


def test_pinned_legacy_repair_adds_only_contract_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    repaired = tmp_path / "contract-repair"
    production_run = tmp_path / "production-run-must-stay-absent"
    manifest = subject.repair_legacy_package_contract(
        destination=repaired,
        legacy_package_dir=subject.LEGACY_PACKAGE_DIR,
        repository_root=REPO_ROOT,
        production_run_dir=production_run,
    )

    with zipfile.ZipFile(
        subject.LEGACY_PACKAGE_DIR / subject.SOURCE_NAME, "r"
    ) as archive:
        legacy_payloads = {
            name: archive.read(name) for name in archive.namelist()
        }
    with zipfile.ZipFile(repaired / subject.SOURCE_NAME, "r") as archive:
        repaired_payloads = {
            name: archive.read(name) for name in archive.namelist()
        }
    assert set(repaired_payloads) - set(legacy_payloads) == {
        subject.CONTRACT_RELATIVE_PATH
    }
    assert not (set(legacy_payloads) - set(repaired_payloads))
    assert all(
        repaired_payloads[name] == raw
        for name, raw in legacy_payloads.items()
    )
    assert subject.sha256_bytes(
        repaired_payloads[subject.CONTRACT_RELATIVE_PATH]
    ) == subject.PERFORMANCE_CONTRACT_SHA256
    assert manifest["source_entry_count"] == len(legacy_payloads) + 1
    assert production_run.exists() is False

    with pytest.raises(FileExistsError, match="package already exists"):
        subject.repair_legacy_package_contract(
            destination=repaired,
            legacy_package_dir=subject.LEGACY_PACKAGE_DIR,
            repository_root=REPO_ROOT,
            production_run_dir=production_run,
        )

    existing_production = tmp_path / "existing-production"
    existing_production.mkdir()
    with pytest.raises(FileExistsError, match="production run already exists"):
        subject.repair_legacy_package_contract(
            destination=tmp_path / "blocked-repair",
            legacy_package_dir=subject.LEGACY_PACKAGE_DIR,
            repository_root=REPO_ROOT,
            production_run_dir=existing_production,
        )

    tampered = _copy_package(repaired, tmp_path / "contract-tamper")
    changed_payloads = dict(repaired_payloads)
    contract = json.loads(
        changed_payloads[subject.CONTRACT_RELATIVE_PATH].decode("utf-8")
    )
    contract["tamper"] = True
    changed_payloads[subject.CONTRACT_RELATIVE_PATH] = subject.canonical_bytes(
        contract
    )
    _resign_package_with_archive_payloads(tampered, changed_payloads)
    with pytest.raises(ValueError, match="performance contract changed"):
        subject.validate_package(tampered)
