from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

import ofc_regular.hu_m43_attempt07_preflight_spot as spot
from ofc_regular.hu_m43_attempt06_teacher import ATTEMPT06_FROZEN_MODEL_SHA256
from ofc_regular.hu_m43_attempt07_contract import (
    AI_PROFILES_SHA256,
    M43_ATTEMPT07_PLAN_SHA256,
)
from ofc_regular.run_hu_m43_attempt07_preflight import (
    ATTEMPT07_PREFLIGHT_ROW_SCHEMA,
    ATTEMPT07_PREFLIGHT_SOURCE_SHA256,
    canonical_json_bytes,
)


ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "scripts" / "Start-GcpHuM43Attempt07PreflightRun.ps1"
STATUS = ROOT / "scripts" / "Get-GcpHuM43Attempt07PreflightRunStatus.ps1"
RECEIVE = ROOT / "scripts" / "Receive-GcpHuM43Attempt07PreflightRun.ps1"
STARTUP = ROOT / "scripts" / "startup_hu_m43_attempt07_preflight.sh"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(root: Path, *, exclude: set[str] | None = None) -> list[dict]:
    ignored = exclude or set()
    return [
        {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": _sha(path)}
        for path in sorted(
            (item for item in root.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(root).as_posix(),
        )
        if path.relative_to(root).as_posix() not in ignored
    ]


def _attempt06_base_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    run_name = "attempt06-base-test"
    base = tmp_path / run_name
    package = base / "package_src"
    files = {
        "artifacts/lambda_rank_candidate.pkl": b"lambda-model\n",
        "configs/hu_joint_policy_m43_attempt06.json": b'{"plan":"test"}\n',
        "configs/hu_joint_policy_m43_attempt06_status.json": b'{"status":"test"}\n',
        "models/model.bin": b"model\n",
        "target/release/native.so": b"native\n",
        "src/ofc_regular/ai_profiles.py": b"AI_PROFILES = {}\n",
        "shards_manifest.jsonl": b'{"shard":0}\n',
        "requirements-attempt06.txt": b"numpy==2.2.6\n",
    }
    for relative, payload in files.items():
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    model_manifest = package / "source_model_manifest.json"
    _write(
        model_manifest,
        {
            "schema": spot.ATTEMPT06_EXPECTED_MODEL_SCHEMA,
            "model_count": 1,
            "models": [
                {
                    "path": "models/model.bin",
                    "bytes": (package / "models/model.bin").stat().st_size,
                    "sha256": _sha(package / "models/model.bin"),
                }
            ],
        },
    )
    native_manifest = package / "source_native_manifest.json"
    _write(
        native_manifest,
        {
            "schema": spot.ATTEMPT06_EXPECTED_NATIVE_SCHEMA,
            "binary_count": 1,
            "binaries": [
                {
                    "path": "target/release/native.so",
                    "bytes": (package / "target/release/native.so").stat().st_size,
                    "sha256": _sha(package / "target/release/native.so"),
                    "platform": "linux-x86_64",
                }
            ],
        },
    )

    plan_sha = _sha(package / "configs/hu_joint_policy_m43_attempt06.json")
    status_sha = _sha(package / "configs/hu_joint_policy_m43_attempt06_status.json")
    model_sha = _sha(package / "artifacts/lambda_rank_candidate.pkl")
    ai_sha = _sha(package / "src/ofc_regular/ai_profiles.py")
    closure = {
        "schema": spot.ATTEMPT06_SOURCE_CLOSURE_SCHEMA,
        "status": "closed_no_fresh_seed_materialized",
        "run_name": run_name,
        "plan_sha256": plan_sha,
        "status_sha256": status_sha,
        "model_sha256": model_sha,
        "ai_profiles_sha256": ai_sha,
        "files": _rows(package),
        "fresh_seed_content_opened": False,
        "teacher_executed": False,
    }
    _write(package / "source_closure_manifest.json", closure)
    shutil.copy2(package / "source_closure_manifest.json", base / "source_closure_manifest.json")
    shutil.copy2(package / "shards_manifest.jsonl", base / "shards_manifest.jsonl")
    shutil.copy2(
        package / "configs/hu_joint_policy_m43_attempt06.json",
        base / "hu_joint_policy_m43_attempt06.json",
    )
    shutil.copy2(
        package / "configs/hu_joint_policy_m43_attempt06_status.json",
        base / "hu_joint_policy_m43_attempt06_status.json",
    )
    (base / "startup_hu_m43_attempt06_teacher.sh").write_bytes(b"#!/bin/sh\nexit 0\n")
    source_zip = base / "ofc_regular_hu_m43_attempt06_teacher_source.zip"
    spot._deterministic_zip(package, source_zip)

    monkeypatch.setattr(spot, "ATTEMPT06_BASE_RUN_NAME", run_name)
    monkeypatch.setattr(spot, "M43_ATTEMPT06_PLAN_SHA256", plan_sha)
    monkeypatch.setattr(spot, "ATTEMPT06_FROZEN_MODEL_SHA256", model_sha)
    monkeypatch.setattr(spot, "AI_PROFILES_SHA256", ai_sha)
    monkeypatch.setattr(spot, "ATTEMPT06_MODEL_MANIFEST_SHA256", _sha(model_manifest))
    monkeypatch.setattr(spot, "ATTEMPT06_NATIVE_MANIFEST_SHA256", _sha(native_manifest))
    monkeypatch.setattr(spot, "ATTEMPT06_EXPECTED_MODEL_COUNT", 1)
    monkeypatch.setattr(spot, "ATTEMPT06_EXPECTED_NATIVE_COUNT", 1)
    manifest = {
        "schema": spot.ATTEMPT06_PACKAGE_MANIFEST_SCHEMA,
        "status": "frozen_package_only_no_fresh_content",
        "run_name": run_name,
        "plan_sha256": plan_sha,
        "status_sha256": status_sha,
        "model_sha256": model_sha,
        "ai_profiles_sha256": ai_sha,
        "schedule_sha256": _sha(base / "shards_manifest.jsonl"),
        "source_closure_sha256": _sha(base / "source_closure_manifest.json"),
        "source_zip_sha256": _sha(source_zip),
        "startup_sha256": _sha(base / "startup_hu_m43_attempt06_teacher.sh"),
        "source_model_manifest_sha256": _sha(model_manifest),
        "source_native_manifest_sha256": _sha(native_manifest),
        "total_roots": 50,
        "total_shards": 50,
        "roots_per_shard": 1,
        "root_profile_assignment": "root_index_mod_5_in_frozen_profile_order",
        "candidate_samples": 8,
        "evaluation_samples": 128,
        "native_batch_threads": 4,
        "learned_nonbaseline_top_k": 8,
        "fresh_seed_content_opened": False,
        "teacher_executed": False,
        "gcloud_invoked": False,
        "instances_created": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    _write(base / "manifest.json", manifest)
    monkeypatch.setattr(spot, "ATTEMPT06_BASE_MANIFEST_SHA256", _sha(base / "manifest.json"))
    return base


def _closure(tmp_path: Path) -> tuple[Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    schedule = tmp_path / "shards_manifest.jsonl"
    schedule.write_bytes(
        b"".join(canonical_json_bytes(row) for row in spot.build_preflight_schedule())
    )
    manifest = tmp_path / "manifest.json"
    _write(
        manifest,
        {
            "schema": spot.PACKAGE_MANIFEST_SCHEMA,
            "status": "packaged_attempt06_copy_plus_overlay_without_execution",
            "run_name": "attempt07-preflight-test",
            "jobs": 5,
            "source_roots": [0, 1, 2],
            "machine_type": spot.ATTEMPT07_MACHINE_TYPE,
            "native_batch_threads": 4,
            "base_attempt06_manifest_sha256": spot.ATTEMPT06_BASE_MANIFEST_SHA256,
            "base_attempt06_package_tree_sha256": "1" * 64,
            "base_attempt06_source_zip_sha256": "2" * 64,
            "package_tree_sha256": "3" * 64,
            "overlay_closure_sha256": "4" * 64,
            "source_zip_sha256": "5" * 64,
            "source_zip_bytes": 1,
            "startup_sha256": "6" * 64,
            "schedule_sha256": _sha(schedule),
            "preflight_plan_sha256": _sha(
                ROOT / "configs" / "hu_joint_policy_m43_attempt07_preflight.json"
            ),
            "attempt07_plan_sha256": M43_ATTEMPT07_PLAN_SHA256,
            "source_merged_sha256": ATTEMPT07_PREFLIGHT_SOURCE_SHA256,
            "model_sha256": ATTEMPT06_FROZEN_MODEL_SHA256,
            "ai_profiles_sha256": AI_PROFILES_SHA256,
            "new_root_generated": False,
            "teacher_executed": False,
            "gcloud_invoked": False,
            "instances_created": False,
            "arm_selection_performed": False,
            "current_profile_resolved": False,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
        },
    )
    receipts = tmp_path / "evidence"
    spot.create_local_evidence_receipt(
        suite="attempt07_pytest",
        passed=99,
        failed=0,
        output=receipts / "attempt07.json",
    )
    spot.create_local_evidence_receipt(
        suite="rust_parity",
        passed=9,
        failed=0,
        output=receipts / "rust.json",
    )
    spot.create_local_evidence_receipt(
        suite="package_tests",
        passed=spot.ATTEMPT07_PACKAGE_TESTS_PASSED,
        failed=0,
        manifest_path=manifest,
        output=receipts / "package.json",
    )
    authorization = tmp_path / "spot_authorization.json"
    spot.create_launch_authorization(
        manifest_path=manifest,
        attempt07_tests_receipt=receipts / "attempt07.json",
        rust_parity_receipt=receipts / "rust.json",
        package_tests_receipt=receipts / "package.json",
        output=authorization,
    )
    return manifest, schedule, authorization


def _job_fixture(
    root: Path,
    *,
    manifest: Path,
    schedule: Path,
    authorization: Path,
    job_index: int,
) -> Path:
    spec = spot.build_preflight_schedule()[job_index]
    job_dir = root / spec["output_prefix"]
    proof = job_dir / "preflight.json"
    _write(
        proof,
        {
            "schema": ATTEMPT07_PREFLIGHT_ROW_SCHEMA,
            "status": "pass_preflight_only_no_arm_selection",
            "source": {"source_root_index": spec["source_root_index"]},
            "execution": {
                "batch_child_selectors": spec["batch_child_selectors"],
                "native_batch_threads": 4,
            },
            "contract": {"plan_sha256": M43_ATTEMPT07_PLAN_SHA256},
            "result_proof": {
                "opaque_teacher_sha256": hashlib.sha256(
                    f"job-{job_index}".encode("ascii")
                ).hexdigest(),
                "semantic_parity_sha256": "a" * 64,
            },
            "science_boundary": {
                "arm_selection_allowed": False,
                "current_profile_resolved": False,
            },
        },
    )
    artifacts = {
        "checkpoint.json": b'{"status":"complete"}\n',
        "heartbeat.json": b'{"status":"complete"}\n',
        "summary.json": b'{"status":"complete"}\n',
        "run.log": b"synthetic bounded preflight\n",
    }
    for name, payload in artifacts.items():
        (job_dir / name).write_bytes(payload)
    _write(
        job_dir / "DONE.json",
        {
            "schema": spot.DONE_SCHEMA,
            "status": "complete",
            "run_name": "attempt07-preflight-test",
            "job_index": job_index,
            "job_id": spec["job_id"],
            "source_root_index": spec["source_root_index"],
            "batch_child_selectors": spec["batch_child_selectors"],
            "native_batch_threads": 4,
            "output_prefix": spec["output_prefix"],
            "output_sha256": _sha(proof),
            "checkpoint_sha256": _sha(job_dir / "checkpoint.json"),
            "heartbeat_sha256": _sha(job_dir / "heartbeat.json"),
            "summary_sha256": _sha(job_dir / "summary.json"),
            "run_log_sha256": _sha(job_dir / "run.log"),
            "manifest_sha256": _sha(manifest),
            "authorization_sha256": _sha(authorization),
            "schedule_sha256": _sha(schedule),
            "attempt07_plan_sha256": M43_ATTEMPT07_PLAN_SHA256,
            "source_merged_sha256": ATTEMPT07_PREFLIGHT_SOURCE_SHA256,
            "model_sha256": ATTEMPT06_FROZEN_MODEL_SHA256,
            "ai_profiles_sha256": AI_PROFILES_SHA256,
            "elapsed_seconds": 1.25 + job_index,
            "peak_rss_bytes": 1000 + job_index,
            "teacher_values_exported": False,
            "arm_selection_performed": False,
            "new_root_generated": False,
            "current_profile_resolved": False,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
        },
    )
    return job_dir


def test_schedule_is_exact_five_bounded_one_job_per_vm_assignments() -> None:
    rows = spot.build_preflight_schedule()
    spot.validate_preflight_schedule(rows)
    assert [row["job_id"] for row in rows] == [
        "root0_batch_a",
        "root0_batch_b",
        "root0_scalar",
        "root1_batch",
        "root2_batch",
    ]
    assert [row["source_root_index"] for row in rows] == [0, 0, 0, 1, 2]
    assert [row["batch_child_selectors"] for row in rows] == [
        True,
        True,
        False,
        True,
        True,
    ]
    assert {row["machine_type"] for row in rows} == {"c4-standard-4"}
    assert {row["native_batch_threads"] for row in rows} == {4}
    assert all(row["new_root_generation_allowed"] is False for row in rows)


def test_schedule_mutation_fails_closed() -> None:
    rows = [dict(row) for row in spot.build_preflight_schedule()]
    rows[2]["batch_child_selectors"] = True
    with pytest.raises(ValueError, match="schedule changed"):
        spot.validate_preflight_schedule(rows)


def test_launch_authorization_binds_three_receipts_without_prejudging_result(
    tmp_path: Path,
) -> None:
    manifest, _, authorization = _closure(tmp_path)
    payload = json.loads(authorization.read_text(encoding="utf-8"))
    assert payload["status"] == "authorized_for_bounded_spot_preflight"
    assert payload["manifest_sha256"] == _sha(manifest)
    assert payload["actual_scalar_batch_result"] == (
        "pending_spot_preflight_receive_and_aggregate"
    )
    assert payload["actual_operational_go_no_go"] == (
        "pending_spot_preflight_receive_and_aggregate"
    )
    assert payload["local_evidence"]["attempt07_pytest"]["passed"] == 99
    assert payload["local_evidence"]["rust_parity"]["passed"] == 9
    assert payload["local_evidence"]["package_tests"]["passed"] == 16
    assert set(payload) == spot._LAUNCH_AUTHORIZATION_KEYS
    assert payload["current_profile_mutated"] is False


def test_local_evidence_receipt_uses_fixed_commands_and_manifest_binding(
    tmp_path: Path,
) -> None:
    manifest, _, _ = _closure(tmp_path)
    output = tmp_path / "generated-package-receipt.json"
    receipt = spot.create_local_evidence_receipt(
        suite="package_tests",
        passed=16,
        failed=0,
        manifest_path=manifest,
        output=output,
    )
    assert receipt["manifest_sha256"] == _sha(manifest)
    assert receipt["command"] == (
        "python -m pytest tests/test_hu_m43_attempt07_preflight_spot.py -q"
    )
    assert output.read_bytes() == canonical_json_bytes(receipt)
    with pytest.raises(ValueError, match="counts changed"):
        spot.create_local_evidence_receipt(
            suite="attempt07_pytest",
            passed=100,
            failed=0,
            output=tmp_path / "bad.json",
        )
    with pytest.raises(ValueError, match="counts changed"):
        spot.create_local_evidence_receipt(
            suite="package_tests",
            passed=15,
            failed=0,
            manifest_path=manifest,
            output=tmp_path / "bad-package.json",
        )
    mutated_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    mutated_manifest["unexpected"] = False
    _write(manifest, mutated_manifest)
    with pytest.raises(ValueError, match="manifest schema changed"):
        spot.create_local_evidence_receipt(
            suite="package_tests",
            passed=16,
            failed=0,
            manifest_path=manifest,
            output=tmp_path / "bad-manifest.json",
        )


def test_launch_authorization_is_no_clobber_or_same_hash(tmp_path: Path) -> None:
    manifest, _, authorization = _closure(tmp_path)
    receipts = tmp_path / "evidence"
    before = authorization.read_bytes()
    spot.create_launch_authorization(
        manifest_path=manifest,
        attempt07_tests_receipt=receipts / "attempt07.json",
        rust_parity_receipt=receipts / "rust.json",
        package_tests_receipt=receipts / "package.json",
        output=authorization,
    )
    assert authorization.read_bytes() == before
    receipt_path = receipts / "rust.json"
    receipt_before = receipt_path.read_bytes()
    receipt = json.loads(receipt_before)
    receipt["unexpected"] = False
    _write(receipt_path, receipt)
    with pytest.raises(ValueError, match="launch evidence changed"):
        spot.create_launch_authorization(
            manifest_path=manifest,
            attempt07_tests_receipt=receipts / "attempt07.json",
            rust_parity_receipt=receipt_path,
            package_tests_receipt=receipts / "package.json",
            output=tmp_path / "unexpected-auth.json",
        )
    receipt_path.write_bytes(receipt_before)
    authorization.write_bytes(b"different\n")
    with pytest.raises(FileExistsError, match="immutable artifact already differs"):
        spot.create_launch_authorization(
            manifest_path=manifest,
            attempt07_tests_receipt=receipts / "attempt07.json",
            rust_parity_receipt=receipts / "rust.json",
            package_tests_receipt=receipts / "package.json",
            output=authorization,
        )


def test_validate_received_job_binds_done_authorization_and_redacted_proof(
    tmp_path: Path,
) -> None:
    manifest, schedule, authorization = _closure(tmp_path)
    job = _job_fixture(
        tmp_path / "jobs",
        manifest=manifest,
        schedule=schedule,
        authorization=authorization,
        job_index=0,
    )
    audit_path = job / "received_audit.json"
    audit = spot.validate_received_job(
        job_dir=job,
        schedule_path=schedule,
        manifest_path=manifest,
        authorization_path=authorization,
        job_index=0,
        output=audit_path,
    )
    assert audit["authorization_sha256"] == _sha(authorization)
    assert audit["teacher_values_opened"] is False
    assert audit_path.read_bytes() == canonical_json_bytes(audit)
    auth = json.loads(authorization.read_text(encoding="utf-8"))
    auth["unexpected"] = False
    _write(authorization, auth)
    with pytest.raises(ValueError, match="launch authorization fields changed"):
        spot.validate_received_job(
            job_dir=job,
            schedule_path=schedule,
            manifest_path=manifest,
            authorization_path=authorization,
            job_index=0,
            output=audit_path,
        )


def test_validate_received_job_rejects_bool_int_confusion(
    tmp_path: Path,
) -> None:
    for case, (field, value) in enumerate(
        (("job_index", True), ("peak_rss_bytes", True), ("batch_child_selectors", 1))
    ):
        case_root = tmp_path / f"case-{case}"
        manifest, schedule, authorization = _closure(case_root)
        job = _job_fixture(
            case_root / "jobs",
            manifest=manifest,
            schedule=schedule,
            authorization=authorization,
            job_index=0,
        )
        done_path = job / "DONE.json"
        done = json.loads(done_path.read_text(encoding="utf-8"))
        done[field] = value
        _write(done_path, done)
        with pytest.raises(ValueError, match="DONE"):
            spot.validate_received_job(
                job_dir=job,
                schedule_path=schedule,
                manifest_path=manifest,
                authorization_path=authorization,
                job_index=0,
                output=job / "received_audit.json",
            )


def test_merge_is_exact_five_and_no_clobber_or_same_hash(tmp_path: Path) -> None:
    manifest, schedule, authorization = _closure(tmp_path)
    jobs = tmp_path / "jobs"
    for job_index in range(5):
        job = _job_fixture(
            jobs,
            manifest=manifest,
            schedule=schedule,
            authorization=authorization,
            job_index=job_index,
        )
        spot.validate_received_job(
            job_dir=job,
            schedule_path=schedule,
            manifest_path=manifest,
            authorization_path=authorization,
            job_index=job_index,
            output=job / "received_audit.json",
        )
    output = tmp_path / "merged" / "preflight.jsonl"
    receipt = tmp_path / "merged" / "receipt.json"
    merged = spot.merge_received_jobs(
        jobs_root=jobs,
        schedule_path=schedule,
        manifest_path=manifest,
        output=output,
        receipt=receipt,
    )
    assert merged["jobs"] == 5
    assert merged["authorization_sha256"] == _sha(authorization)
    before = output.read_bytes()
    spot.merge_received_jobs(
        jobs_root=jobs,
        schedule_path=schedule,
        manifest_path=manifest,
        output=output,
        receipt=receipt,
    )
    assert output.read_bytes() == before
    output.write_bytes(b"different\n")
    with pytest.raises(FileExistsError, match="immutable artifact already differs"):
        spot.merge_received_jobs(
            jobs_root=jobs,
            schedule_path=schedule,
            manifest_path=manifest,
            output=output,
            receipt=receipt,
        )


def test_attempt06_base_validation_closes_actual_tree_and_runtime_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _attempt06_base_fixture(tmp_path / "valid", monkeypatch)
    manifest, manifest_sha, tree_sha = spot._validate_base_package(base)
    assert manifest["run_name"] == "attempt06-base-test"
    assert manifest_sha == _sha(base / "manifest.json")
    assert tree_sha == spot._tree_digest(spot._tree_rows(base / "package_src"))

    mutations = (
        ("extra", "extra.bin", b"unlisted\n", "tree differs"),
        ("listed", "models/model.bin", b"changed\n", "closure file changed"),
        (
            "model-manifest",
            "source_model_manifest.json",
            b'{"unexpected":true}\n',
            "closure file changed",
        ),
        (
            "native",
            "target/release/native.so",
            b"changed-native\n",
            "closure file changed",
        ),
    )
    mutation_root = tmp_path / "tree-mutations"
    mutation_root.mkdir()
    for name, relative, payload, message in mutations:
        mutated = mutation_root / name
        shutil.copytree(base, mutated)
        (mutated / "package_src" / relative).write_bytes(payload)
        with pytest.raises(ValueError, match=message):
            spot._validate_base_package(mutated)


def test_attempt06_base_validation_rejects_rebound_closure_and_zip_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closure_base = _attempt06_base_fixture(tmp_path / "closure", monkeypatch)
    closure = json.loads(
        (closure_base / "source_closure_manifest.json").read_text(encoding="utf-8")
    )
    closure["files"][0]["sha256"] = "0" * 64
    _write(closure_base / "source_closure_manifest.json", closure)
    _write(closure_base / "package_src/source_closure_manifest.json", closure)
    manifest = json.loads((closure_base / "manifest.json").read_text(encoding="utf-8"))
    manifest["source_closure_sha256"] = _sha(
        closure_base / "source_closure_manifest.json"
    )
    _write(closure_base / "manifest.json", manifest)
    monkeypatch.setattr(
        spot, "ATTEMPT06_BASE_MANIFEST_SHA256", _sha(closure_base / "manifest.json")
    )
    with pytest.raises(ValueError, match="source closure file changed"):
        spot._validate_base_package(closure_base)

    zip_base = _attempt06_base_fixture(tmp_path / "zip", monkeypatch)
    source_zip = zip_base / "ofc_regular_hu_m43_attempt06_teacher_source.zip"
    package = zip_base / "package_src"
    with zipfile.ZipFile(source_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(
            (item for item in package.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(package).as_posix(),
        ):
            relative = path.relative_to(package).as_posix()
            payload = path.read_bytes()
            if relative == "models/model.bin":
                payload = b"evil!\n"
            archive.writestr(relative, payload)
    manifest = json.loads((zip_base / "manifest.json").read_text(encoding="utf-8"))
    manifest["source_zip_sha256"] = _sha(source_zip)
    _write(zip_base / "manifest.json", manifest)
    monkeypatch.setattr(
        spot, "ATTEMPT06_BASE_MANIFEST_SHA256", _sha(zip_base / "manifest.json")
    )
    with pytest.raises(ValueError, match="ZIP member SHA-256 changed"):
        spot._validate_base_package(zip_base)


def test_scripts_keep_spot_boundary_and_authorization_binding() -> None:
    start = _read(START)
    startup = _read(STARTUP)
    status = _read(STATUS)
    receive = _read(RECEIVE)
    for text in (start, startup, status, receive):
        assert "AUTHORIZATION_SHA256" in text or "authorization_sha256" in text
        assert "c4-standard-16" not in text
    for token in (
        "c4-standard-4",
        "NativeBatchThreads -ne 4",
        "PackageOnly and CreateInstances are mutually exclusive",
        "authorized_for_bounded_spot_preflight",
        "scalar_batch_parity_test_harness",
        "pending_spot_preflight_receive_and_aggregate",
        "--if-generation-match=0",
        "manifest is the publication commit marker and is always last",
        "Complete a previously interrupted publication with the commit marker last",
        "$remoteManifestPresent",
        "--instance-termination-action', 'STOP'",
        "ResumeStoppedInstances",
        "$ExpectedPackageTestsPassed = 16",
        "Assert-M43A7ExactPropertySet",
    ):
        assert token in start
    for token in (
        'gcloud storage cp "$PREFIX/source/spot_authorization.json"',
        '[[ "$(sha /tmp/spot_authorization.json)" == "$AUTHORIZATION_SHA256" ]]',
        "upload_once_or_verify \"$RESULT/DONE.json\" \"$DONE_URI\"",
        "write_checkpoint running",
        "write_heartbeat running",
        "deterministic_recompute_allowed",
        "SELF_DELETE",
        "attempt07_preflight_executing_startup.sh",
        "instance/attributes/startup-script",
    ):
        assert token in startup
    assert "run_hu_m43_attempt07_preflight" in startup
    assert "generate_hu_m4_t1_data" not in startup
    assert "result_payload_downloaded = $false" in status
    assert "log_payload_downloaded = $false" in status
    assert "Status reads exact DONE metadata only" in status
    assert "validate-received-job" in receive
    assert "merge-received-jobs" in receive
    assert "ofc_regular.aggregate_hu_m43_attempt07_preflight" in receive
    assert "preflight_aggregate.json" in receive
    assert "[string]$aggregate.decision -notin @('go', 'no_go')" in receive
    assert "finalize_hu_m43_attempt07_preflight" not in receive
    assert "immutable output is never overwritten" in receive
    assert "os.replace" not in receive


def _powershell() -> str | None:
    return shutil.which("powershell") or shutil.which("pwsh")


@pytest.mark.skipif(_powershell() is None, reason="PowerShell unavailable")
@pytest.mark.parametrize("path", [START, STATUS, RECEIVE])
def test_powershell_scripts_parse(path: Path) -> None:
    shell = _powershell()
    assert shell is not None
    escaped = str(path).replace("'", "''")
    command = (
        "$ErrorActionPreference='Stop';"
        f"[void][scriptblock]::Create((Get-Content -LiteralPath '{escaped}' -Raw))"
    )
    completed = subprocess.run(
        [shell, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
def test_startup_bash_syntax() -> None:
    completed = subprocess.run(
        [shutil.which("bash") or "bash", "-n", "scripts/startup_hu_m43_attempt07_preflight.sh"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_package_source_is_copy_then_overlay_never_attempt06_in_place() -> None:
    source = _read(ROOT / "src" / "ofc_regular" / "hu_m43_attempt07_preflight_spot.py")
    assert "shutil.copytree(base / \"package_src\", package_src" in source
    assert "Attempt06 base package changed while copying" in source
    assert 'sha256_file(startup) != existing.get("startup_sha256")' in source
    assert '_tree_digest(_tree_rows(destination / "package_src"))' in source
    assert "os.replace(staging, destination)" in source
    assert "os.replace(temporary, path)" not in source[source.index("def _write_atomic_no_clobber") : source.index("def _load_mapping")]
