from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ofc_regular import (
    close_hu_m31_t3_step6d_performance_lock_rearm1_startup_failure as subject,
)
from ofc_regular import run_hu_m31_t3_step6d_performance_v2 as runner


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(subject.canonical_bytes(value))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    package = tmp_path / "package"
    package.mkdir()
    source = package / subject.SOURCE_NAME
    source.write_bytes(b"opaque-source")
    startup = package / subject.STARTUP_NAME
    startup.write_text(
        '\n'.join(
            (
                'require(j["schema"] == '
                '"hu_m31_t3_step6d_performance_shard_manifest_v2", '
                '"runner job schema changed")',
                '[[ "${#JOB_FIELDS[@]}" -eq 5 ]]',
            )
        )
        + "\n",
        encoding="utf-8",
    )
    contract = runner.build_run_contract(
        candidate_library_sha256=(
            runner.CANDIDATE02_PERFORMANCE_LOCK_ACCEPTED_CANDIDATE_LIBRARY_SHA256
        ),
        reference_library_sha256=(
            runner.CANDIDATE02_PERFORMANCE_LOCK_ACCEPTED_REFERENCE_LIBRARY_SHA256
        ),
        variant=runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_VARIANT,
    )
    digest = runner.canonical_sha256(contract)
    monkeypatch.setattr(subject, "RUN_CONTRACT_DIGEST", digest)
    records = []
    for ordinal, job_id in enumerate(subject.JOB_IDS):
        role = job_id.split("-")[0]
        hands = list(range((ordinal % 10) * 10, (ordinal % 10 + 1) * 10))
        job = runner.build_shard_manifest(
            run_contract=contract,
            source_role=role,
            work_hand_indices=hands,
        )
        path = package / "jobs" / f"{job_id}.json"
        _write_json(path, job)
        records.append(
            {
                "job_id": job_id,
                "path": f"jobs/{job_id}.json",
                "sha256": _sha(path),
                "bytes": path.stat().st_size,
                "source_role": role,
                "work_hand_indices": hands,
            }
        )

    root_claim_path = tmp_path / subject.GLOBAL_ROOT_CLAIM_NAME
    root_claim = {
        "schema": subject.REARM1_ROOT_CLAIM_SCHEMA,
        "status": subject.REARM1_ROOT_CLAIM_STATUS,
        "global_claim_path": "C:/repo/REARM1.json",
        "lock_output_directory": "D:/run/rearm1/roots-open",
        "lock_run_contract_digest": digest,
        "ai_profiles_current": {"sha256": subject.CURRENT_PROFILE_SHA256},
        "seed_contract": {
            "seed_set_sha256": (
                runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SEED_SET_SHA256
            )
        },
        "restrictions": {
            "reseed_allowed": False,
            "cloud_authorized": False,
        },
    }
    _write_json(root_claim_path, root_claim)

    materialization = tmp_path / "roots-open" / "materialization.json"
    _write_json(
        materialization,
        {
            "schema": subject.REARM1_MATERIALIZATION_SCHEMA,
            "root_count": 100,
            "reseeded": False,
            "current_profile_changed": False,
        },
    )
    seal = materialization.parent / "seal.json"
    _write_json(
        seal,
        {
            "schema": subject.REARM1_SEAL_SCHEMA,
            "root_count": 100,
            "observation_count": 200,
            "visibility": {"opponent_private_discards_used": False},
            "current_profile_changed": False,
        },
    )

    manifest = {
        "schema": "hu_m31_t3_step6d_performance_lock_spot_package_v1",
        "status": "immutable_performance_lock_package_ready_not_authorized",
        "run_name": subject.RUN_NAME,
        "source_name": subject.SOURCE_NAME,
        "source_sha256": _sha(source),
        "source_bytes": source.stat().st_size,
        "startup_name": subject.STARTUP_NAME,
        "startup_sha256": _sha(startup),
        "plan_sha256": subject.PLAN_SHA256,
        "run_contract": contract,
        "run_contract_digest": digest,
        "job_manifests": records,
        "spot_execution_authorized": False,
        "performance_lock_authorized": False,
        "quality_pilot_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
        "m31_complete": False,
        "gcloud_invoked": False,
    }
    manifest_path = package / subject.MANIFEST_NAME
    _write_json(manifest_path, manifest)

    spot_claim_path = tmp_path / subject.GLOBAL_SPOT_CLAIM_NAME
    spot_claim = {
        "schema": "hu_m31_t3_step6d_performance_lock_global_spot_claim_v1",
        "status": "global_one_shot_spot_identity_claimed_before_authorization",
        "run_name": subject.RUN_NAME,
        "global_root_claim_sha256": _sha(root_claim_path),
        "package_manifest_sha256": _sha(manifest_path),
        "source_sha256": _sha(source),
        "startup_sha256": _sha(startup),
        "precontent_plan_sha256": subject.PLAN_SHA256,
        "run_contract_digest": digest,
        "authorized_job_ids": list(subject.JOB_IDS),
        "max_initial_jobs": 20,
        "max_resume_attempts": 1,
        "alternate_package_authorization_allowed": False,
        "quality_pilot_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
        "runtime_policy_activated": False,
    }
    _write_json(spot_claim_path, spot_claim)

    ready = {
        "schema": "hu_m31_t3_step6d_performance_lock_package_ready_v1",
        "status": "immutable_local_performance_lock_package_complete",
        "run_name": subject.RUN_NAME,
        "package_manifest_sha256": _sha(manifest_path),
        "source_sha256": _sha(source),
        "startup_sha256": _sha(startup),
        "plan_sha256": subject.PLAN_SHA256,
        "run_contract_digest": digest,
        "logical_job_count": 20,
        "gcloud_invoked": False,
        "spot_vm_started": False,
        "current_profile_changed": False,
    }
    _write_json(package / subject.READY_NAME, ready)
    authorization = {
        "schema": "hu_m31_t3_step6d_performance_lock_launch_authorization_v1",
        "status": "explicit_one_shot_performance_lock_spot_authorization",
        "run_name": subject.RUN_NAME,
        "package_manifest_sha256": _sha(manifest_path),
        "source_sha256": _sha(source),
        "startup_sha256": _sha(startup),
        "plan_sha256": subject.PLAN_SHA256,
        "run_contract_digest": digest,
        "authorized_job_ids": list(subject.JOB_IDS),
        "logical_job_count": 20,
        "global_spot_claim": spot_claim,
        "spot_execution_authorized": True,
        "performance_lock_authorized": True,
        "quality_pilot_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
    }
    authorization_path = package / subject.AUTHORIZATION_NAME
    _write_json(authorization_path, authorization)
    launch_claim = {
        "schema": "hu_m31_t3_step6d_performance_lock_launch_claim_v1",
        "status": "exclusive_performance_lock_launch_claim_before_remote_mutation",
        "run_name": subject.RUN_NAME,
        "selected_job_ids": list(subject.JOB_IDS),
        "package_manifest_sha256": _sha(manifest_path),
        "launch_authorization_sha256": _sha(authorization_path),
        "plan_sha256": subject.PLAN_SHA256,
        "run_contract_digest": digest,
        "crash_reuse_authorized": False,
    }
    launch_claim_path = package / subject.LAUNCH_CLAIM_NAME
    _write_json(launch_claim_path, launch_claim)
    launch_result = {
        "schema": "hu_m31_t3_step6d_performance_lock_launch_result_v1",
        "status": "performance_lock_jobs_created",
        "run_name": subject.RUN_NAME,
        "selected_job_ids": list(subject.JOB_IDS),
        "created": [
            {"job_id": job_id, "attempt_index": 0, "status": "created"}
            for job_id in subject.JOB_IDS
        ],
        "failures": [],
        "logical_job_count": 20,
        "launch_claim_sha256": _sha(launch_claim_path),
        "performance_lock_authorized": True,
        "quality_pilot_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
        "runtime_policy_activated": False,
    }
    _write_json(package / subject.LAUNCH_RESULT_NAME, launch_result)

    anchors = {
        name: _sha(path)
        for name, path in {
            subject.MANIFEST_NAME: manifest_path,
            subject.READY_NAME: package / subject.READY_NAME,
            subject.AUTHORIZATION_NAME: authorization_path,
            subject.LAUNCH_CLAIM_NAME: launch_claim_path,
            subject.LAUNCH_RESULT_NAME: package / subject.LAUNCH_RESULT_NAME,
            subject.SOURCE_NAME: source,
            subject.STARTUP_NAME: startup,
            subject.GLOBAL_ROOT_CLAIM_NAME: root_claim_path,
            subject.GLOBAL_SPOT_CLAIM_NAME: spot_claim_path,
            subject.MATERIALIZATION_NAME: materialization,
            subject.SEAL_NAME: seal,
        }.items()
    }
    monkeypatch.setattr(subject, "EXPECTED_HASHES", anchors)

    logs = tmp_path / "logs"
    for job_id in subject.JOB_IDS:
        path = logs / "jobs" / job_id / "attempt-0" / "startup.log"
        path.parent.mkdir(parents=True)
        path.write_text(
            "\n".join(
                (
                    f"Copying {subject.REMOTE_PREFIX}/source/jobs/{job_id}.json",
                    subject.ERROR_LINE,
                    (
                        'require(j["schema"] == '
                        '"hu_m31_t3_step6d_performance_shard_manifest_v2", '
                        f'"{subject.ERROR_LINE}")'
                    ),
                    subject.PYTHON_FAILURE_LINE,
                    subject.SHELL_FAILURE_LINE,
                )
            )
            + "\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(
        subject,
        "EXPECTED_LOG_SHA256",
        {
            job_id: _sha(
                logs / "jobs" / job_id / "attempt-0" / "startup.log"
            )
            for job_id in subject.JOB_IDS
        },
    )
    return {
        "package": package,
        "root_claim": root_claim_path,
        "spot_claim": spot_claim_path,
        "roots": materialization.parent,
        "logs": logs,
    }


def _build(evidence: dict[str, Path], **kwargs: object) -> dict:
    counts = kwargs.pop(
        "direct_counts", {key: 0 for key in subject.COUNT_KEYS}
    )
    return subject.build_closeout(
        package_dir=evidence["package"],
        global_root_claim_path=evidence["root_claim"],
        global_spot_claim_path=evidence["spot_claim"],
        rearm1_root_directory=evidence["roots"],
        startup_log_tree=evidence["logs"],
        direct_counts=counts,
        **kwargs,
    )


def test_consumed_rearm1_attempt0_is_closed_without_authorizing_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _evidence(tmp_path, monkeypatch)
    receipt = _build(evidence)

    assert receipt["incident"]["failed_jobs"] == 20
    assert receipt["incident"]["actual_job_schema"] == (
        runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SHARD_MANIFEST_SCHEMA
    )
    assert receipt["incident"]["startup_expected_job_schema"] == (
        runner.SHARD_MANIFEST_SCHEMA
    )
    assert receipt["startup_logs"]["job_ids"] == list(subject.JOB_IDS)
    assert receipt["control_plane_counts"] == {
        key: 0 for key in subject.COUNT_KEYS
    }
    assert receipt["disposition"]["rearm1_attempt1_authorized"] is False
    assert receipt["disposition"]["rearm1_package_reuse_authorized"] is False
    assert receipt["disposition"]["rearm1_root_reuse_authorized"] is False
    assert receipt["disposition"]["rearm1_seed_reuse_authorized"] is False
    assert receipt["disposition"]["rearm1_claim_reuse_authorized"] is False
    assert receipt["disposition"]["fresh_rearm2_plan_authorized"] is False
    assert receipt["content_access"]["root_content_opened"] is False
    assert receipt["content_access"]["result_content_opened"] is False
    assert receipt["content_access"]["cloud_mutated"] is False


def test_closeout_is_write_once_and_canonical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _evidence(tmp_path, monkeypatch)
    output = tmp_path / "closeout.json"
    receipt = subject.generate_closeout(
        package_dir=evidence["package"],
        global_root_claim_path=evidence["root_claim"],
        global_spot_claim_path=evidence["spot_claim"],
        rearm1_root_directory=evidence["roots"],
        startup_log_tree=evidence["logs"],
        output_path=output,
        direct_counts={key: 0 for key in subject.COUNT_KEYS},
    )
    assert output.read_bytes() == subject.canonical_bytes(receipt)
    with pytest.raises(FileExistsError, match="write-once"):
        subject.generate_closeout(
            package_dir=evidence["package"],
            global_root_claim_path=evidence["root_claim"],
            global_spot_claim_path=evidence["spot_claim"],
            rearm1_root_directory=evidence["roots"],
            startup_log_tree=evidence["logs"],
            output_path=output,
            direct_counts={key: 0 for key in subject.COUNT_KEYS},
        )


@pytest.mark.parametrize("key", ("done", "heartbeat", "result", "attempt1"))
def test_nonzero_remote_state_fails_closed(
    key: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _evidence(tmp_path, monkeypatch)
    counts = {name: 0 for name in subject.COUNT_KEYS}
    counts[key] = 1
    with pytest.raises(ValueError, match="exactly zero"):
        _build(evidence, direct_counts=counts)


@pytest.mark.parametrize(
    "replacement",
    ("other schema", "runner job schema changed\nunexpected result content"),
)
def test_log_or_content_boundary_tamper_fails_closed(
    replacement: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _evidence(tmp_path, monkeypatch)
    log = (
        evidence["logs"]
        / "jobs"
        / subject.JOB_IDS[0]
        / "attempt-0"
        / "startup.log"
    )
    raw = log.read_text(encoding="utf-8")
    log.write_text(raw.replace(subject.ERROR_LINE, replacement), encoding="utf-8")
    with pytest.raises(ValueError):
        _build(evidence)


def test_attempt1_control_or_job_schema_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _evidence(tmp_path, monkeypatch)
    (evidence["package"] / "resume_claim.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="attempt1"):
        _build(evidence)

    (evidence["package"] / "resume_claim.json").unlink()
    job = evidence["package"] / "jobs" / f"{subject.JOB_IDS[0]}.json"
    job.write_bytes(job.read_bytes() + b" ")
    with pytest.raises(ValueError, match="canonical|anchor"):
        _build(evidence)
