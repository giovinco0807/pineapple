from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_performance_lock_rearm1_spot as rearm1_spot,
)
from ofc_regular import (
    hu_m31_t3_step6d_performance_lock_rearm2_spot as subject,
)
from ofc_regular import hu_m31_t3_step6d_performance_lock_spot_v1 as spot_v1
from ofc_regular import run_hu_m31_t3_step6d_performance_v2 as runner


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(spot_v1.canonical_bytes(value))


def _fake_modules(root_claim_path: Path) -> tuple[Any, Any]:
    contract = runner.build_run_contract(
        candidate_library_sha256=(
            runner.CANDIDATE02_PERFORMANCE_LOCK_ACCEPTED_CANDIDATE_LIBRARY_SHA256
        ),
        reference_library_sha256=(
            runner.CANDIDATE02_PERFORMANCE_LOCK_ACCEPTED_REFERENCE_LIBRARY_SHA256
        ),
        variant=runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_VARIANT,
    )
    plan = SimpleNamespace(
        CANDIDATE_VARIANT=runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_VARIANT,
        RUN_ID=runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_RUN_ID,
        SCHEDULE=runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_SCHEDULE,
        ROOT_SCHEMA=runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_ROOT_SCHEMA,
        DONE_SCHEMA=runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_DONE_SCHEMA,
        SHARD_MANIFEST_SCHEMA=(
            runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_SHARD_MANIFEST_SCHEMA
        ),
        LOCK_RUN_CONTRACT_DIGEST=spot_v1.canonical_sha256(contract),
        PRECONTENT_PLAN_SHA256="a" * 64,
        PLAN_SCOPE=subject.REARM2_PLAN_SCOPE,
        CANDIDATE_LIBRARY_SHA256="b" * 64,
        REFERENCE_LIBRARY_SHA256="c" * 64,
        FEATURE_ENCODER_SHA256="d" * 64,
        CURRENT_PROFILE_REGISTRY_SHA256="e" * 64,
        validate_precontent_plan=lambda value: value,
    )
    lock_open = SimpleNamespace(
        CLAIM_SCHEMA="test_rearm2_claim_v1",
        CLAIM_STATUS="test_rearm2_claimed",
        MATERIALIZATION_SCHEMA="test_rearm2_materialization_v1",
        MATERIALIZATION_STATUS="test_rearm2_materialized",
        SEAL_SCHEMA="test_rearm2_seal_v1",
        SEAL_STATUS="test_rearm2_sealed",
        PerformanceLockInputs=object,
        DEFAULT_GLOBAL_CLAIM_PATH=root_claim_path,
        DEFAULT_PRECONTENT_PLAN_PATH=root_claim_path.with_name("plan.json"),
        OLD_V1_GLOBAL_CLAIM_SHA256="1" * 64,
        OLD_V1_SEAL_SHA256="2" * 64,
        REARM1_GLOBAL_CLAIM_SHA256="3" * 64,
        REARM1_SEAL_SHA256="4" * 64,
        validate_open_claim=lambda inputs: inputs,
        validate_root_seal=lambda inputs: inputs,
        _write_once_durable=lambda path, value: spot_v1._write_once(path, value),
    )
    return plan, lock_open


def _smoke_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, dict[str, Any], Any, Any]:
    run_dir = tmp_path / "package"
    run_dir.mkdir()
    root_claim_path = tmp_path / "SIGNED-Rearm2-Root-Claim.json"
    plan, lock_open = _fake_modules(root_claim_path)
    monkeypatch.setattr(subject, "_load_rearm_modules", lambda: (plan, lock_open))

    root_claim = {
        "schema": lock_open.CLAIM_SCHEMA,
        "global_claim_path": str(root_claim_path.resolve()),
        "lock_output_directory": str((tmp_path / "SIGNED-Rearm2-Roots").resolve()),
    }
    _write(root_claim_path, root_claim)
    source_path = run_dir / spot_v1.SOURCE_NAME
    with zipfile.ZipFile(source_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            spot_v1.OPEN_CLAIM_PACKAGE_PATH,
            spot_v1.canonical_bytes(root_claim),
        )
        for index in range(100):
            archive.writestr(
                f"{spot_v1.ROOT_PACKAGE_DIR}/hand_{index:03d}.json",
                b'{"poisoned_for_no_root_read_proof":true}\n',
            )

    startup_path = run_dir / spot_v1.STARTUP_NAME
    startup_path.write_text(
        """#!/usr/bin/env bash
python3 - "$@" <<'PY'
import hashlib
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
    assert len(roots) == 100
    assert roots[0] == "frozen/full100_roots/hand_000.json"
    assert roots[-1] == "frozen/full100_roots/hand_099.json"
    # Do not archive.read() a root here.
    global_claim = authorization["global_spot_claim"]
    receipt_sha = authorization["preauthorize_smoke_receipt_sha256"]
    assert global_claim["schema"] == (
        "hu_m31_t3_step6d_performance_lock_rearm2_global_spot_claim_v1"
    )
    assert global_claim["preauthorize_smoke_receipt_sha256"] == receipt_sha
    assert len(receipt_sha) == 64
    assert global_claim["global_root_claim_path"] == claim["global_claim_path"]
    assert global_claim["lock_output_directory"] == claim["lock_output_directory"]
    assert global_claim["precontent_plan_sha256"] == manifest["plan_sha256"]
    assert job["schema"] == (
        "hu_m31_t3_step6d_candidate02_"
        "performance_lock_recovery_shard_manifest_v3"
    )
    assert hashlib.sha256(job_path.read_bytes()).hexdigest() == job_sha256
print(
    "lock_rearm2|"
    "hu_m31_t3_step6d_candidate02_"
    "performance_lock_recovery_source_shard_done_v3"
)
PY
""",
        encoding="utf-8",
        newline="\n",
    )

    contract = runner.build_run_contract(
        candidate_library_sha256=(
            runner.CANDIDATE02_PERFORMANCE_LOCK_ACCEPTED_CANDIDATE_LIBRARY_SHA256
        ),
        reference_library_sha256=(
            runner.CANDIDATE02_PERFORMANCE_LOCK_ACCEPTED_REFERENCE_LIBRARY_SHA256
        ),
        variant=runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_VARIANT,
    )
    records: list[dict[str, Any]] = []
    for role in runner.SOURCE_ROLES:
        for shard_index in range(10):
            job_id = f"{role}-shard-{shard_index:02d}"
            work = list(range(shard_index * 10, (shard_index + 1) * 10))
            job = runner.build_shard_manifest(
                run_contract=contract,
                source_role=role,
                work_hand_indices=work,
            )
            job_path = run_dir / "jobs" / f"{job_id}.json"
            _write(job_path, job)
            records.append(
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
        "schema": "test-rearm2-package",
        "run_name": "rearm2-local-smoke",
        "source_sha256": spot_v1.sha256_file(source_path),
        "startup_sha256": spot_v1.sha256_file(startup_path),
        "plan_sha256": plan.PRECONTENT_PLAN_SHA256,
        "run_contract": contract,
        "run_contract_digest": plan.LOCK_RUN_CONTRACT_DIGEST,
        "job_manifests": records,
        "launch_target": {"machine_type": "c4-standard-16"},
        "cost_guard": {"cloud_cost_cap_usd": 0},
        "tail_qualification": {
            "summary_sha256": "1" * 64,
            "validation_sha256": "2" * 64,
            "open_claim_sha256": spot_v1.canonical_sha256(root_claim),
            "root_seal_sha256": "3" * 64,
        },
    }
    _write(run_dir / spot_v1.MANIFEST_NAME, manifest)
    global_spot_path = root_claim_path.with_name(subject.GLOBAL_SPOT_CLAIM_NAME)
    return run_dir, root_claim_path, global_spot_path, manifest, plan, lock_open


def test_context_restores_v1_and_rearm1_globals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, lock_open = _fake_modules(tmp_path / "claim.json")
    monkeypatch.setattr(subject, "_load_rearm_modules", lambda: (plan, lock_open))
    prior = (
        spot_v1.lock_plan,
        spot_v1.lock_open,
        spot_v1.GLOBAL_SPOT_CLAIM_NAME,
        spot_v1._validate_archive,
        spot_v1._GLOBAL_SPOT_CLAIM_KEYS,
        spot_v1._AUTHORIZATION_KEYS,
        spot_v1._global_spot_claim_payload,
        spot_v1.validate_launch_authorization,
        rearm1_spot.GLOBAL_SPOT_CLAIM_NAME,
        rearm1_spot.REARM1_PLAN_SCOPE,
        rearm1_spot.REARM1_STARTUP_PACKAGE_PHASE,
    )
    with subject._rearm2_context(bind_smoke_receipt=True):
        assert spot_v1.lock_plan is plan
        assert spot_v1.lock_open is lock_open
        assert spot_v1.GLOBAL_SPOT_CLAIM_NAME == subject.GLOBAL_SPOT_CLAIM_NAME
        assert spot_v1._validate_archive is subject._validate_rearm2_archive
        assert subject.PREAUTHORIZE_SMOKE_SHA_FIELD in (
            spot_v1._GLOBAL_SPOT_CLAIM_KEYS
        )
        assert subject.PREAUTHORIZE_SMOKE_SHA_FIELD in spot_v1._AUTHORIZATION_KEYS
        assert rearm1_spot.REARM1_PLAN_SCOPE == subject.REARM2_PLAN_SCOPE
        assert (
            rearm1_spot.REARM1_STARTUP_PACKAGE_PHASE
            == subject.REARM2_STARTUP_PACKAGE_PHASE
        )
    assert prior == (
        spot_v1.lock_plan,
        spot_v1.lock_open,
        spot_v1.GLOBAL_SPOT_CLAIM_NAME,
        spot_v1._validate_archive,
        spot_v1._GLOBAL_SPOT_CLAIM_KEYS,
        spot_v1._AUTHORIZATION_KEYS,
        spot_v1._global_spot_claim_payload,
        spot_v1.validate_launch_authorization,
        rearm1_spot.GLOBAL_SPOT_CLAIM_NAME,
        rearm1_spot.REARM1_PLAN_SCOPE,
        rearm1_spot.REARM1_STARTUP_PACKAGE_PHASE,
    )


def test_exhaustive_smoke_executes_all_20_jobs_and_writes_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, _root_claim, global_spot, _manifest, _plan, _open = (
        _smoke_fixture(tmp_path, monkeypatch)
    )
    receipt_path = run_dir / subject.PREAUTHORIZE_SMOKE_NAME
    receipt = subject.write_preauthorize_smoke_receipt(run_dir)

    assert receipt["schema"] == subject.PREAUTHORIZE_SMOKE_SCHEMA
    assert receipt["verified_job_count"] == 20
    assert receipt["startup_invocation_count"] == 20
    assert receipt["verified_job_ids"] == list(spot_v1.authorized_job_ids())
    assert receipt["startup_package_phase"] == "lock_rearm2"
    assert receipt["startup_root_member_read"] is False
    assert receipt["temporary_poisoned_root_count"] == 100
    assert receipt["receipt_sha_bound_before_cloud_mutation"] is True
    assert receipt_path.read_bytes() == spot_v1.canonical_bytes(receipt)
    assert not global_spot.exists()
    assert not (run_dir / spot_v1.AUTHORIZATION_NAME).exists()
    assert subject.validate_preauthorize_smoke_receipt(
        receipt_path,
        run_dir=run_dir,
    ) == receipt
    with pytest.raises(FileExistsError):
        subject.write_preauthorize_smoke_receipt(run_dir)


def test_authorization_binds_exact_receipt_sha_in_both_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, root_claim_path, global_spot, manifest, _plan, _open = (
        _smoke_fixture(tmp_path, monkeypatch)
    )
    receipt = subject.write_preauthorize_smoke_receipt(run_dir)
    receipt_path = run_dir / subject.PREAUTHORIZE_SMOKE_NAME
    receipt_sha = spot_v1.sha256_file(receipt_path)
    assert receipt_sha == spot_v1.canonical_sha256(receipt)

    monkeypatch.setattr(spot_v1, "validate_package", lambda _target: manifest)

    def validate_authorization(
        target: str | Path,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        authorization = spot_v1._read_canonical(
            Path(target) / spot_v1.AUTHORIZATION_NAME,
            "test rearm2 authorization",
        )
        return manifest, authorization

    monkeypatch.setattr(
        subject,
        "_BASE_VALIDATE_LAUNCH_AUTHORIZATION",
        validate_authorization,
    )
    authorization = subject.authorize_launch(run_dir)

    assert global_spot.is_file()
    global_claim = json.loads(global_spot.read_text(encoding="utf-8"))
    assert global_claim["schema"] == subject.GLOBAL_SPOT_CLAIM_SCHEMA
    assert global_claim["status"] == subject.GLOBAL_SPOT_CLAIM_STATUS
    assert global_claim[subject.PREAUTHORIZE_SMOKE_SHA_FIELD] == receipt_sha
    assert (
        authorization[subject.PREAUTHORIZE_SMOKE_SHA_FIELD] == receipt_sha
    )
    assert (
        authorization["global_spot_claim"][subject.PREAUTHORIZE_SMOKE_SHA_FIELD]
        == receipt_sha
    )
    assert (
        authorization["global_spot_claim"]["global_root_claim_path"]
        == str(root_claim_path.resolve())
    )


def test_authorization_refuses_missing_receipt_before_global_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, _root_claim, global_spot, manifest, _plan, _open = (
        _smoke_fixture(tmp_path, monkeypatch)
    )
    monkeypatch.setattr(spot_v1, "validate_package", lambda _target: manifest)

    with pytest.raises(ValueError, match="receipt is missing or unsafe"):
        subject.authorize_launch(run_dir)
    assert not global_spot.exists()
    assert not (run_dir / spot_v1.AUTHORIZATION_NAME).exists()


def test_archive_guard_requires_zero_v1_and_rearm1_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    false_guards = {
        "old_v1_root_reused": False,
        "rearm1_attempt1_reused": False,
        "rearm1_package_reused": False,
        "rearm1_root_reused": False,
        "rearm1_seed_reused": False,
        "rearm1_claim_reused": False,
    }
    claim = {"rearm_guards": dict(false_guards)}
    materialization = {
        **false_guards,
        "fresh_recovery_v3_seed_schedule": True,
        "reseeded": False,
    }
    _plan, fake_open = _fake_modules(tmp_path / "claim.json")
    seal = {
        "prior_lock_comparison": {
            **{
                f"{prior}_{kind}_overlap_count": 0
                for prior in ("old_v1", "rearm1")
                for kind in ("root_hash", "fingerprint", "seed")
            },
            "old_v1_global_claim_sha256": fake_open.OLD_V1_GLOBAL_CLAIM_SHA256,
            "old_v1_seal_sha256": fake_open.OLD_V1_SEAL_SHA256,
            "rearm1_global_claim_sha256": fake_open.REARM1_GLOBAL_CLAIM_SHA256,
            "rearm1_seal_sha256": fake_open.REARM1_SEAL_SHA256,
            "rearm1_attempt1_reused": False,
            "rearm1_package_reused": False,
            "rearm1_root_reused": False,
            "rearm1_seed_reused": False,
            "rearm1_claim_reused": False,
        },
        "rearm_guards": {
            **false_guards,
            "fresh_710_series_seed_schedule": True,
            "same_identity_resume_only": True,
            "post_claim_reseeded": False,
        },
    }
    monkeypatch.setattr(spot_v1, "lock_open", fake_open)
    monkeypatch.setattr(
        subject,
        "_BASE_VALIDATE_ARCHIVE",
        lambda _source, _entries: ({}, claim, seal, materialization),
    )
    parity_calls: list[Path] = []
    monkeypatch.setattr(
        subject,
        "_validate_rearm2_parity_binding",
        lambda source, **_kwargs: parity_calls.append(source),
    )
    source = tmp_path / "source.zip"
    assert subject._validate_rearm2_archive(source, {}) == (
        {},
        claim,
        seal,
        materialization,
    )
    assert parity_calls == [source]

    seal["prior_lock_comparison"]["rearm1_seed_overlap_count"] = 1
    with pytest.raises(ValueError, match="overlap proof changed"):
        subject._validate_rearm2_archive(source, {})


def test_archive_parity_binding_is_pinned_to_rearm2_validator(
    tmp_path: Path,
) -> None:
    claim = {"schema": "rearm2-claim"}
    materialization = {"schema": "rearm2-materialization"}
    seal = {"schema": "rearm2-seal"}
    validator_path = Path(subject.__file__).with_name(
        "verify_hu_m31_t3_feature_encoder_platform_parity_rearm2.py"
    )
    validator = validator_path.read_bytes()
    validator_sha = hashlib.sha256(validator).hexdigest()
    assert len(validator) == subject.EXPECTED_REARM2_PARITY_VALIDATOR_BYTES
    assert validator_sha == subject.EXPECTED_REARM2_PARITY_VALIDATOR_SHA256

    def write_archive(path: Path, *, rearm1_reused: bool) -> None:
        receipt = {
            "lock_chain": {
                "contract": "performance_lock_rearm2_v1",
                "validator_schema": subject.REARM2_PARITY_VALIDATOR_SCHEMA,
                "validator_status": subject.REARM2_PARITY_VALIDATOR_STATUS,
                "validator_bytes": len(validator),
                "validator_sha256": validator_sha,
                "global_claim_sha256": spot_v1.canonical_sha256(claim),
                "materialization_sha256": (
                    spot_v1.canonical_sha256(materialization)
                ),
                "seal_sha256": spot_v1.canonical_sha256(seal),
                "v1_reused": False,
                "rearm1_reused": rearm1_reused,
                "fresh_recovery_v3_seed_schedule": True,
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
                subject.REARM2_PARITY_VALIDATOR_PACKAGE_PATH,
                validator,
            )

    source = tmp_path / "source.zip"
    write_archive(source, rearm1_reused=False)
    subject._validate_rearm2_parity_binding(
        source,
        claim=claim,
        materialization=materialization,
        seal=seal,
    )
    tampered = tmp_path / "tampered.zip"
    write_archive(tampered, rearm1_reused=True)
    with pytest.raises(ValueError, match="validator binding changed"):
        subject._validate_rearm2_parity_binding(
            tampered,
            claim=claim,
            materialization=materialization,
            seal=seal,
        )


def test_original_probe_and_legacy_adapter_identities_are_unchanged() -> None:
    original_probe = Path(
        "scripts/verify_hu_m31_t3_feature_encoder_platform_parity.py"
    )
    assert hashlib.sha256(original_probe.read_bytes()).hexdigest() == (
        "a1905bc951ac54680e4993a5f63cc85a2d2579f0f962529fd3a0c8968bfcd921"
    )
    assert rearm1_spot.GLOBAL_SPOT_CLAIM_NAME == (
        "GLOBAL_PERFORMANCE_LOCK_REARM1_SPOT_CLAIM.json"
    )
    assert rearm1_spot.REARM1_STARTUP_PACKAGE_PHASE == "lock_rearm1"


def test_cli_and_public_api_cover_full_rearm2_lifecycle() -> None:
    required = {
        "package_performance_lock",
        "validate_package",
        "validate_global_spot_claim",
        "preauthorize_smoke",
        "write_preauthorize_smoke_receipt",
        "validate_preauthorize_smoke_receipt",
        "authorize_launch",
        "validate_launch_authorization",
        "launch_jobs",
        "validate_launch_chain",
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
