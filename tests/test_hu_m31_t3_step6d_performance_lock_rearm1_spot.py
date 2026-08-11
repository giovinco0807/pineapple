from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_performance_lock_rearm1_spot as subject,
)
from ofc_regular import hu_m31_t3_step6d_performance_lock_spot_v1 as spot_v1
from ofc_regular import run_hu_m31_t3_step6d_performance_v2 as runner


ACTUAL_REARM1_PACKAGE = Path(
    "D:/ofc-gcp-runs/"
    "regular-hu-m31-c02-performance-lock-rearm1-20260717-001/package"
)


def _fake_rearm_modules(root_claim_path: Path) -> tuple[Any, Any]:
    plan = SimpleNamespace(
        CANDIDATE_VARIANT=runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_VARIANT,
        RUN_ID=runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_RUN_ID,
        SCHEDULE=runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SCHEDULE,
        ROOT_SCHEMA=runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_ROOT_SCHEMA,
        DONE_SCHEMA=runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_DONE_SCHEMA,
        SHARD_MANIFEST_SCHEMA=(
            runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SHARD_MANIFEST_SCHEMA
        ),
        LOCK_RUN_CONTRACT_DIGEST=(
            "f02e8845401eef92ba617f2c832204bdc74c20aa9e91810c52bf99687c4886b5"
        ),
        PRECONTENT_PLAN_SHA256="a" * 64,
        PLAN_SCOPE="performance_lock_rearm1_fresh_roots_only",
        CANDIDATE_LIBRARY_SHA256="b" * 64,
        REFERENCE_LIBRARY_SHA256="c" * 64,
        FEATURE_ENCODER_SHA256="d" * 64,
        CURRENT_PROFILE_REGISTRY_SHA256="e" * 64,
        validate_precontent_plan=lambda value: value,
    )
    lock_open = SimpleNamespace(
        CLAIM_SCHEMA="test_rearm1_claim_v2",
        MATERIALIZATION_SCHEMA="test_rearm1_materialization_v2",
        MATERIALIZATION_STATUS="test_rearm1_materialized",
        SEAL_SCHEMA="test_rearm1_seal_v2",
        PerformanceLockInputs=object,
        OLD_V1_GLOBAL_CLAIM_SHA256="1" * 64,
        OLD_V1_SEAL_SHA256="2" * 64,
        STARTUP_FAILURE_RECEIPT_SHA256="3" * 64,
        DEFAULT_GLOBAL_CLAIM_PATH=root_claim_path,
        DEFAULT_PRECONTENT_PLAN_PATH=root_claim_path.with_name("plan.json"),
        validate_open_claim=lambda inputs: inputs,
        validate_root_seal=lambda inputs: inputs,
    )
    return plan, lock_open


def _write_canonical(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(spot_v1.canonical_bytes(value))


def _preauthorize_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, Path]:
    run_dir = tmp_path / "package"
    run_dir.mkdir()
    root_claim_path = tmp_path / "SIGNED-Rearm1-Root-Claim.json"
    plan, lock_open = _fake_rearm_modules(root_claim_path)
    monkeypatch.setattr(subject, "_load_rearm_modules", lambda: (plan, lock_open))

    root_claim = {
        "schema": lock_open.CLAIM_SCHEMA,
        "global_claim_path": str(root_claim_path.resolve()),
        "lock_output_directory": str((tmp_path / "SIGNED-Lock-Output").resolve()),
    }
    root_seal = {"schema": lock_open.SEAL_SCHEMA, "identity": "sealed"}
    _write_canonical(root_claim_path, root_claim)

    source_path = run_dir / spot_v1.SOURCE_NAME
    with zipfile.ZipFile(
        source_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr(
            spot_v1.OPEN_CLAIM_PACKAGE_PATH,
            spot_v1.canonical_bytes(root_claim),
        )
        archive.writestr(
            spot_v1.ROOT_SEAL_PACKAGE_PATH,
            spot_v1.canonical_bytes(root_seal),
        )
        for index in range(100):
            archive.writestr(
                f"{spot_v1.ROOT_PACKAGE_DIR}/hand_{index:03d}.json",
                (
                    '{"hand_index":'
                    f"{index},"
                    '"would_fail_if_the_verifier_reads_this_root":true}\n'
                ).encode("ascii"),
            )

    startup_path = run_dir / spot_v1.STARTUP_NAME
    startup_path.write_text(
        """#!/usr/bin/env bash
python3 - "$@" <<'PY'
import json
import pathlib
import sys
import zipfile

ROOT_PREFIX = "frozen/full100_roots/"
CLAIM_MEMBER = "frozen/performance_lock_open_claim.json"

source_path, manifest_path, authorization_path = map(pathlib.Path, sys.argv[1:4])
job_path = pathlib.Path(sys.argv[4])
job_id, job_sha256 = sys.argv[5:7]
manifest = json.loads(manifest_path.read_bytes())
authorization = json.loads(authorization_path.read_bytes())
job = json.loads(job_path.read_bytes())
with zipfile.ZipFile(source_path) as archive:
    names = archive.namelist()
    # PRE-CONTENT ORDERING: only signed control content is opened here.
    claim = json.loads(archive.read(CLAIM_MEMBER))
    roots = [name for name in names if name.startswith(ROOT_PREFIX)]
    assert roots == [
        f"frozen/full100_roots/hand_{index:03d}.json"
        for index in range(100)
    ]
    # Do not archive.read() a root here.
    preview = authorization["global_spot_claim"]
    assert preview["global_root_claim_path"] == claim["global_claim_path"]
    assert preview["lock_output_directory"] == claim["lock_output_directory"]
    assert preview["precontent_plan_sha256"] == manifest["plan_sha256"]
    assert preview["run_contract_digest"] == manifest["run_contract_digest"]
    assert job["schema"] == (
        "hu_m31_t3_step6d_candidate02_"
        "performance_lock_recovery_shard_manifest_v2"
    )
    assert job_id == f"{job['source_role']}-shard-{int(job_id[-2:]):02d}"
    import hashlib
    assert hashlib.sha256(job_path.read_bytes()).hexdigest() == job_sha256
print(
    "lock_rearm1|"
    "hu_m31_t3_step6d_candidate02_"
    "performance_lock_recovery_source_shard_done_v2"
)
PY
""",
        encoding="utf-8",
        newline="\n",
    )
    run_contract = runner.build_run_contract(
        candidate_library_sha256=(
            runner.CANDIDATE02_PERFORMANCE_LOCK_ACCEPTED_CANDIDATE_LIBRARY_SHA256
        ),
        reference_library_sha256=(
            runner.CANDIDATE02_PERFORMANCE_LOCK_ACCEPTED_REFERENCE_LIBRARY_SHA256
        ),
        variant=runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_VARIANT,
    )
    job_records = []
    for role in runner.SOURCE_ROLES:
        for shard_index in range(10):
            job_id = f"{role}-shard-{shard_index:02d}"
            work = list(range(shard_index * 10, (shard_index + 1) * 10))
            job = runner.build_shard_manifest(
                run_contract=run_contract,
                source_role=role,
                work_hand_indices=work,
            )
            job_path = run_dir / "jobs" / f"{job_id}.json"
            _write_canonical(job_path, job)
            job_records.append(
                {
                    "job_id": job_id,
                    "source_role": role,
                    "shard_index": shard_index,
                    "work_hand_indices": work,
                    "path": f"jobs/{job_id}.json",
                    "output_prefix": f"jobs/{job_id}",
                    "sha256": spot_v1.sha256_file(job_path),
                    "bytes": job_path.stat().st_size,
                }
            )
    manifest = {
        "schema": "test_rearm1_package",
        "run_name": "rearm1-local-smoke",
        "source_sha256": spot_v1.sha256_file(source_path),
        "startup_sha256": spot_v1.sha256_file(startup_path),
        "plan_sha256": plan.PRECONTENT_PLAN_SHA256,
        "run_contract": run_contract,
        "run_contract_digest": plan.LOCK_RUN_CONTRACT_DIGEST,
        "job_manifests": job_records,
        "launch_target": {"machine_type": "c4-standard-16"},
        "cost_guard": {"cloud_cost_cap_usd": 0},
        "tail_qualification": {
            "summary_sha256": "4" * 64,
            "validation_sha256": "5" * 64,
            "open_claim_sha256": spot_v1.canonical_sha256(root_claim),
            "root_seal_sha256": spot_v1.canonical_sha256(root_seal),
        },
    }
    _write_canonical(run_dir / spot_v1.MANIFEST_NAME, manifest)
    global_spot_claim_path = root_claim_path.with_name(subject.GLOBAL_SPOT_CLAIM_NAME)
    return run_dir, root_claim_path, global_spot_claim_path, startup_path


def test_context_uses_rearm_modules_and_restores_every_v1_global(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_plan = spot_v1.lock_plan
    original_open = spot_v1.lock_open
    original_claim_name = spot_v1.GLOBAL_SPOT_CLAIM_NAME
    original_archive_validator = spot_v1._validate_archive
    fake_plan, fake_open = _fake_rearm_modules(tmp_path / "claim.json")
    monkeypatch.setattr(
        subject,
        "_load_rearm_modules",
        lambda: (fake_plan, fake_open),
    )

    def delegated(run_dir: str | Path) -> dict[str, Any]:
        assert Path(run_dir) == tmp_path
        assert spot_v1.lock_plan is fake_plan
        assert spot_v1.lock_open is fake_open
        assert spot_v1.GLOBAL_SPOT_CLAIM_NAME == (
            "GLOBAL_PERFORMANCE_LOCK_REARM1_SPOT_CLAIM.json"
        )
        assert spot_v1._validate_archive is subject._validate_rearm_archive
        return {"delegated": True}

    monkeypatch.setattr(spot_v1, "validate_package", delegated)
    assert subject.validate_package(tmp_path) == {"delegated": True}
    assert spot_v1.lock_plan is original_plan
    assert spot_v1.lock_open is original_open
    assert spot_v1.GLOBAL_SPOT_CLAIM_NAME == original_claim_name
    assert spot_v1._validate_archive is original_archive_validator

    def rejected(_run_dir: str | Path) -> dict[str, Any]:
        raise RuntimeError("delegated failure")

    monkeypatch.setattr(spot_v1, "validate_package", rejected)
    with pytest.raises(RuntimeError, match="delegated failure"):
        subject.validate_package(tmp_path)
    assert spot_v1.lock_plan is original_plan
    assert spot_v1.lock_open is original_open
    assert spot_v1.GLOBAL_SPOT_CLAIM_NAME == original_claim_name
    assert spot_v1._validate_archive is original_archive_validator


def test_context_process_lock_serializes_parallel_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_plan, fake_open = _fake_rearm_modules(tmp_path / "claim.json")
    monkeypatch.setattr(
        subject,
        "_load_rearm_modules",
        lambda: (fake_plan, fake_open),
    )
    gate = threading.Lock()
    active = 0
    maximum = 0

    def delegated(_run_dir: str | Path) -> dict[str, Any]:
        nonlocal active, maximum
        with gate:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.03)
        with gate:
            active -= 1
        return {"ok": True}

    monkeypatch.setattr(spot_v1, "validate_package", delegated)
    threads = [
        threading.Thread(target=subject.validate_package, args=(tmp_path,))
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()
    assert maximum == 1


def test_preauthorize_smoke_runs_exact_verifier_without_persistent_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, root_claim_path, global_spot_claim_path, startup_path = (
        _preauthorize_fixture(tmp_path, monkeypatch)
    )
    before = {
        "root": hashlib.sha256(root_claim_path.read_bytes()).hexdigest(),
        "source": spot_v1.sha256_file(run_dir / spot_v1.SOURCE_NAME),
        "manifest": spot_v1.sha256_file(run_dir / spot_v1.MANIFEST_NAME),
        "startup": spot_v1.sha256_file(startup_path),
    }

    report = subject.preauthorize_smoke(run_dir)

    assert report["schema"] == subject.PREAUTHORIZE_SMOKE_SCHEMA
    assert report["signed_root_claim_strings_preserved"] is True
    assert report["startup_package_phase"] == "lock_rearm1"
    assert report["shard_manifest_schema"] == (
        runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SHARD_MANIFEST_SCHEMA
    )
    assert report["done_schema"] == (
        runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_DONE_SCHEMA
    )
    assert report["verified_job_count"] == 20
    assert report["startup_invocation_count"] == 20
    assert report["verified_job_ids"] == list(spot_v1.authorized_job_ids())
    assert report["actual_package_manifest_validated"] is True
    assert report["preview_authorization_validated"] is True
    assert report["full_no_root_read_startup_contract_executed"] is True
    assert report["receipt_sha_bound_before_cloud_mutation"] is True
    assert report["temporary_poisoned_root_count"] == 100
    assert report["startup_root_member_read"] is False
    assert report["persistent_claim_or_authorization_written"] is False
    assert report["cloud_mutation_executed"] is False
    assert not global_spot_claim_path.exists()
    assert not (run_dir / spot_v1.AUTHORIZATION_NAME).exists()
    assert before == {
        "root": hashlib.sha256(root_claim_path.read_bytes()).hexdigest(),
        "source": spot_v1.sha256_file(run_dir / spot_v1.SOURCE_NAME),
        "manifest": spot_v1.sha256_file(run_dir / spot_v1.MANIFEST_NAME),
        "startup": spot_v1.sha256_file(startup_path),
    }


@pytest.mark.skipif(
    not ACTUAL_REARM1_PACKAGE.is_dir(),
    reason="local immutable rearm1 package is unavailable",
)
def test_current_startup_executes_against_all_actual_rearm1_job_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for the 20/20 recovery-v2 shard-schema startup failure."""

    run_dir = tmp_path / "actual-rearm1-package"
    (run_dir / "jobs").mkdir(parents=True)
    for name in (spot_v1.SOURCE_NAME, spot_v1.MANIFEST_NAME):
        shutil.copy2(ACTUAL_REARM1_PACKAGE / name, run_dir / name)
    shutil.copy2(
        Path("scripts/startup_hu_m31_t3_step6d_full100_v1.sh"),
        run_dir / spot_v1.STARTUP_NAME,
    )
    for job_id in spot_v1.authorized_job_ids():
        shutil.copy2(
            ACTUAL_REARM1_PACKAGE / "jobs" / f"{job_id}.json",
            run_dir / "jobs" / f"{job_id}.json",
        )
    manifest_path = run_dir / spot_v1.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["startup_sha256"] = spot_v1.sha256_file(
        run_dir / spot_v1.STARTUP_NAME
    )
    _write_canonical(manifest_path, manifest)
    monkeypatch.setattr(
        spot_v1,
        "_global_spot_claim_path",
        lambda _claim: tmp_path / "nonexistent-global-spot-claim.json",
    )

    receipt = subject.preauthorize_smoke(run_dir)

    assert receipt["verified_job_count"] == 20
    assert receipt["startup_invocation_count"] == 20
    assert receipt["shard_manifest_schema"] == (
        runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SHARD_MANIFEST_SCHEMA
    )
    assert receipt["startup_root_member_read"] is False


def test_preauthorize_smoke_rejects_actual_recovery_job_schema_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _root_claim_path, _global_spot_claim_path, _startup_path = (
        _preauthorize_fixture(tmp_path, monkeypatch)
    )
    job_path = run_dir / "jobs" / "candidate-shard-00.json"
    job = json.loads(job_path.read_text(encoding="utf-8"))
    assert job["schema"] == (
        runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SHARD_MANIFEST_SCHEMA
    )
    job["schema"] = runner.SHARD_MANIFEST_SCHEMA
    _write_canonical(job_path, job)
    manifest_path = run_dir / spot_v1.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["job_manifests"][0]["sha256"] = spot_v1.sha256_file(job_path)
    manifest["job_manifests"][0]["bytes"] = job_path.stat().st_size
    _write_canonical(manifest_path, manifest)

    with pytest.raises(ValueError, match="shard manifest changed"):
        subject.preauthorize_smoke(run_dir)


def test_actual_package_smoke_receipt_is_write_once_and_replayable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _root_claim_path, _global_spot_claim_path, _startup_path = (
        _preauthorize_fixture(tmp_path, monkeypatch)
    )
    receipt_path = tmp_path / "receipt.json"
    receipt = subject.write_preauthorize_smoke_receipt(
        run_dir,
        output_path=receipt_path,
    )
    assert receipt["schema"] == subject.PREAUTHORIZE_SMOKE_SCHEMA
    assert receipt["status"] == subject.PREAUTHORIZE_SMOKE_STATUS
    assert receipt["write_once_receipt_required"] is True
    assert subject.validate_preauthorize_smoke_receipt(
        receipt_path,
        run_dir=run_dir,
    ) == receipt
    with pytest.raises(FileExistsError):
        subject.write_preauthorize_smoke_receipt(
            run_dir,
            output_path=receipt_path,
        )


def test_preauthorize_smoke_rejects_prior_claim_before_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _root_claim_path, global_spot_claim_path, _startup_path = (
        _preauthorize_fixture(tmp_path, monkeypatch)
    )
    _write_canonical(global_spot_claim_path, {"already": "claimed"})

    def reject_subprocess(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            f"subprocess ran after an existing claim: {args}, {kwargs}"
        )

    monkeypatch.setattr(subject.subprocess, "run", reject_subprocess)
    with pytest.raises(ValueError, match="must precede every Spot/authorization write"):
        subject.preauthorize_smoke(run_dir)
    assert json.loads(global_spot_claim_path.read_text(encoding="utf-8")) == {
        "already": "claimed"
    }


def test_authorize_runs_smoke_before_the_write_capable_delegate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_plan, fake_open = _fake_rearm_modules(tmp_path / "claim.json")
    monkeypatch.setattr(
        subject,
        "_load_rearm_modules",
        lambda: (fake_plan, fake_open),
    )
    order: list[str] = []

    def smoke(run_dir: str | Path) -> dict[str, Any]:
        assert Path(run_dir) == tmp_path
        order.append("smoke")
        return {"smoke": True}

    def authorize(run_dir: str | Path) -> dict[str, Any]:
        assert Path(run_dir) == tmp_path
        assert spot_v1.lock_plan is fake_plan
        assert spot_v1.lock_open is fake_open
        order.append("authorize")
        return {"authorized": True}

    monkeypatch.setattr(subject, "_preauthorize_smoke_unlocked", smoke)
    monkeypatch.setattr(spot_v1, "authorize_launch", authorize)
    assert subject.authorize_launch(tmp_path) == {"authorized": True}
    assert order == ["smoke", "authorize"]


def test_rearm_archive_guard_requires_zero_old_lock_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan, lock_open = _fake_rearm_modules(tmp_path / "claim.json")
    materialization = {
        "schema": lock_open.MATERIALIZATION_SCHEMA,
        "status": lock_open.MATERIALIZATION_STATUS,
        "fresh_recovery_seed_schedule": True,
        "old_v1_attempt1_reused": False,
        "old_v1_root_reused": False,
        "reseeded": False,
    }
    seal = {
        "old_performance_lock_comparison": {
            "old_v1_global_claim_sha256": lock_open.OLD_V1_GLOBAL_CLAIM_SHA256,
            "old_v1_seal_sha256": lock_open.OLD_V1_SEAL_SHA256,
            "old_v1_root_count": 100,
            "rearm1_fingerprint_overlap_count": 0,
            "rearm1_root_hash_overlap_count": 0,
            "rearm1_seed_overlap_count": 0,
            "old_v1_attempt1_reused": False,
            "old_v1_root_reused": False,
        },
        "rearm_guards": {
            "incident_receipt_sha256": lock_open.STARTUP_FAILURE_RECEIPT_SHA256,
            "fresh_700_series_seed_schedule": True,
            "same_identity_resume_only": True,
            "old_v1_attempt1_reused": False,
            "old_v1_root_reused": False,
            "post_claim_reseeded": False,
        },
    }
    monkeypatch.setattr(spot_v1, "lock_open", lock_open)
    monkeypatch.setattr(
        subject,
        "_BASE_VALIDATE_ARCHIVE",
        lambda _source, _entries: ({}, {}, seal, materialization),
    )
    parity_calls: list[Path] = []
    monkeypatch.setattr(
        subject,
        "_validate_rearm_parity_binding",
        lambda source, **_kwargs: parity_calls.append(source),
    )
    assert subject._validate_rearm_archive(tmp_path / "source.zip", {}) == (
        {},
        {},
        seal,
        materialization,
    )
    assert parity_calls == [tmp_path / "source.zip"]

    seal["old_performance_lock_comparison"]["old_v1_root_reused"] = True
    with pytest.raises(ValueError, match="no-reuse proof changed"):
        subject._validate_rearm_archive(tmp_path / "source.zip", {})


def test_rearm_archive_parity_binding_is_pinned_to_packaged_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan, lock_open = _fake_rearm_modules(tmp_path / "claim.json")
    monkeypatch.setattr(spot_v1, "lock_open", lock_open)
    claim = {"schema": lock_open.CLAIM_SCHEMA}
    materialization = {"schema": lock_open.MATERIALIZATION_SCHEMA}
    seal = {"schema": lock_open.SEAL_SCHEMA}
    validator_path = Path(subject.__file__).with_name(
        "verify_hu_m31_t3_feature_encoder_platform_parity_rearm1.py"
    )
    validator = validator_path.read_bytes()
    assert len(validator) == subject.EXPECTED_REARM1_PARITY_VALIDATOR_BYTES
    assert (
        hashlib.sha256(validator).hexdigest()
        == subject.EXPECTED_REARM1_PARITY_VALIDATOR_SHA256
    )

    def write_archive(path: Path, *, reused: bool) -> None:
        receipt = {
            "lock_chain": {
                "contract": "performance_lock_rearm1_v2",
                "validator_schema": subject.REARM1_PARITY_VALIDATOR_SCHEMA,
                "validator_status": subject.REARM1_PARITY_VALIDATOR_STATUS,
                "validator_bytes": len(validator),
                "validator_sha256": hashlib.sha256(validator).hexdigest(),
                "global_claim_sha256": spot_v1.canonical_sha256(claim),
                "materialization_sha256": spot_v1.canonical_sha256(materialization),
                "seal_sha256": spot_v1.canonical_sha256(seal),
                "old_v1_global_claim_sha256": (
                    lock_open.OLD_V1_GLOBAL_CLAIM_SHA256
                ),
                "old_v1_seal_sha256": lock_open.OLD_V1_SEAL_SHA256,
                "startup_failure_receipt_sha256": (
                    lock_open.STARTUP_FAILURE_RECEIPT_SHA256
                ),
                "old_v1_attempt1_reused": False,
                "old_v1_root_reused": reused,
                "fresh_recovery_seed_schedule": True,
                "reseeded": False,
                "current_profile_changed": False,
            }
        }
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(
                spot_v1.MATERIALIZER_PARITY_RECEIPT_PACKAGE_PATH,
                spot_v1.canonical_bytes(receipt),
            )
            archive.writestr(
                subject.REARM1_PARITY_VALIDATOR_PACKAGE_PATH,
                validator,
            )

    source = tmp_path / "source.zip"
    write_archive(source, reused=False)
    subject._validate_rearm_parity_binding(
        source,
        claim=claim,
        materialization=materialization,
        seal=seal,
    )

    tampered = tmp_path / "tampered.zip"
    write_archive(tampered, reused=True)
    with pytest.raises(ValueError, match="validator binding changed"):
        subject._validate_rearm_parity_binding(
            tampered,
            claim=claim,
            materialization=materialization,
            seal=seal,
        )


def test_cli_and_public_api_cover_the_full_rearm_lifecycle() -> None:
    required = {
        "package_performance_lock",
        "validate_package",
        "validate_global_spot_claim",
        "preauthorize_smoke",
        "write_preauthorize_smoke_receipt",
        "validate_preauthorize_smoke_receipt",
        "authorize_launch",
        "validate_launch_authorization",
        "validate_launch_chain",
        "launch_jobs",
        "preflight_resume",
        "resume_jobs",
        "validate_resume_chain",
        "cloud_status",
        "claim_results",
        "validate_result_open_claim",
        "receive_jobs",
        "validate_received_directory",
    }
    assert required.issubset(subject.__all__)
    parsed = subject._parser().parse_args(
        ["preauthorize-smoke", "--run-dir", "local-package"]
    )
    assert parsed.command == "preauthorize-smoke"
    assert parsed.run_dir == Path("local-package")
