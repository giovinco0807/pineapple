from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ofc_regular import (
    close_hu_m31_t3_step6d_performance_lock_startup_failure as subject,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(subject.canonical_bytes(value))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    package = tmp_path / "package"
    package.mkdir()
    source = package / subject.SOURCE_NAME
    source.write_bytes(b"opaque archive bytes; never opened")
    startup = package / subject.STARTUP_NAME
    startup.write_text(
        "\n".join(
            (
                'value.get("global_root_claim_path") == '
                'root_claim.get("global_claim_path")',
                'value.get("lock_output_directory")',
                '== root_claim.get("lock_output_directory")',
                subject.ERROR_LINE,
            )
        ),
        encoding="utf-8",
    )

    job_records = []
    for index, job_id in enumerate(subject.JOB_IDS):
        role = job_id.split("-")[0]
        hands = list(range((index % 10) * 10, (index % 10 + 1) * 10))
        value = {
            "schema": "hu_m31_t3_step6d_performance_shard_manifest_v2",
            "run_contract_digest": subject.RUN_CONTRACT_DIGEST,
            "source_role": role,
            "work_hand_indices": hands,
        }
        relative = f"jobs/{job_id}.json"
        path = package / "jobs" / f"{job_id}.json"
        _write_json(path, value)
        job_records.append(
            {
                "job_id": job_id,
                "path": relative,
                "sha256": _sha(path),
                "bytes": path.stat().st_size,
                "source_role": role,
                "work_hand_indices": hands,
            }
        )

    root_claim_path = tmp_path / subject.GLOBAL_ROOT_CLAIM_NAME
    root_claim = {
        "schema": "hu_m31_t3_step6d_performance_lock_global_claim_v1",
        "status": (
            "global_one_shot_claim_persisted_before_lock_root_touch_"
            "crash_consumes_claim"
        ),
        "global_claim_path": r"c:\users\owner\repo\global_performance_lock_claim.json",
        "lock_output_directory": r"d:\ofc-gcp-runs\run\roots-open",
    }
    _write_json(root_claim_path, root_claim)

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
        "run_contract_digest": subject.RUN_CONTRACT_DIGEST,
        "job_manifests": job_records,
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
    manifest_sha = _sha(manifest_path)

    spot_claim_path = tmp_path / subject.GLOBAL_SPOT_CLAIM_NAME
    spot_claim = {
        "schema": "hu_m31_t3_step6d_performance_lock_global_spot_claim_v1",
        "status": "global_one_shot_spot_identity_claimed_before_authorization",
        "run_name": subject.RUN_NAME,
        "global_root_claim_path": r"C:\Users\Owner\repo\GLOBAL_PERFORMANCE_LOCK_CLAIM.json",
        "global_root_claim_sha256": _sha(root_claim_path),
        "lock_output_directory": r"D:\ofc-gcp-runs\run\roots-open",
        "package_manifest_sha256": manifest_sha,
        "source_sha256": _sha(source),
        "startup_sha256": _sha(startup),
        "precontent_plan_sha256": subject.PLAN_SHA256,
        "run_contract_digest": subject.RUN_CONTRACT_DIGEST,
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
        "package_manifest_sha256": manifest_sha,
        "source_sha256": _sha(source),
        "startup_sha256": _sha(startup),
        "plan_sha256": subject.PLAN_SHA256,
        "run_contract_digest": subject.RUN_CONTRACT_DIGEST,
        "logical_job_count": 20,
        "gcloud_invoked": False,
        "spot_vm_started": False,
        "current_profile_changed": False,
    }
    ready_path = package / subject.READY_NAME
    _write_json(ready_path, ready)

    authorization = {
        "schema": "hu_m31_t3_step6d_performance_lock_launch_authorization_v1",
        "status": "explicit_one_shot_performance_lock_spot_authorization",
        "run_name": subject.RUN_NAME,
        "package_manifest_sha256": manifest_sha,
        "source_sha256": _sha(source),
        "startup_sha256": _sha(startup),
        "plan_sha256": subject.PLAN_SHA256,
        "run_contract_digest": subject.RUN_CONTRACT_DIGEST,
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
        "package_manifest_sha256": manifest_sha,
        "launch_authorization_sha256": _sha(authorization_path),
        "plan_sha256": subject.PLAN_SHA256,
        "run_contract_digest": subject.RUN_CONTRACT_DIGEST,
        "crash_reuse_authorized": False,
    }
    launch_claim_path = package / subject.LAUNCH_CLAIM_NAME
    _write_json(launch_claim_path, launch_claim)

    created = [
        {
            "job_id": job_id,
            "attempt_index": 0,
            "instance": f"{subject.RUN_NAME}-j{index:02d}",
            "status": "created",
        }
        for index, job_id in enumerate(subject.JOB_IDS)
    ]
    launch_result = {
        "schema": "hu_m31_t3_step6d_performance_lock_launch_result_v1",
        "status": "performance_lock_jobs_created",
        "run_name": subject.RUN_NAME,
        "selected_job_ids": list(subject.JOB_IDS),
        "created": created,
        "failures": [],
        "cleanup": [],
        "logical_job_count": 20,
        "launch_claim_sha256": _sha(launch_claim_path),
        "performance_lock_authorized": True,
        "quality_pilot_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
        "runtime_policy_activated": False,
    }
    launch_result_path = package / subject.LAUNCH_RESULT_NAME
    _write_json(launch_result_path, launch_result)

    anchors = {
        subject.MANIFEST_NAME: _sha(manifest_path),
        subject.READY_NAME: _sha(ready_path),
        subject.AUTHORIZATION_NAME: _sha(authorization_path),
        subject.LAUNCH_CLAIM_NAME: _sha(launch_claim_path),
        subject.LAUNCH_RESULT_NAME: _sha(launch_result_path),
        subject.SOURCE_NAME: _sha(source),
        subject.STARTUP_NAME: _sha(startup),
        subject.GLOBAL_ROOT_CLAIM_NAME: _sha(root_claim_path),
        subject.GLOBAL_SPOT_CLAIM_NAME: _sha(spot_claim_path),
    }
    monkeypatch.setattr(subject, "EXPECTED_HASHES", anchors)

    logs = tmp_path / "startup-tree"
    for job_id in subject.JOB_IDS:
        path = logs / "jobs" / job_id / "attempt-0" / "startup.log"
        path.parent.mkdir(parents=True)
        path.write_text(
            "\n".join(
                (
                    f"Copying {subject.REMOTE_PREFIX}/source/{subject.SOURCE_NAME}",
                    f"Copying {subject.REMOTE_PREFIX}/{subject.MANIFEST_NAME}",
                    f"Copying {subject.REMOTE_PREFIX}/source/jobs/{job_id}.json",
                    (
                        f"Copying {subject.REMOTE_PREFIX}/source/"
                        f"{subject.AUTHORIZATION_NAME}"
                    ),
                    (
                        f"Copying {subject.REMOTE_PREFIX}/control/"
                        f"{subject.LAUNCH_CLAIM_NAME}"
                    ),
                    subject.ERROR_LINE,
                    subject.FIRST_FAILURE_PREFIX,
                    subject.SECOND_FAILURE_PREFIX,
                    ')"',
                )
            )
            + "\n",
            encoding="utf-8",
        )
    return {
        "package": package,
        "root_claim": root_claim_path,
        "spot_claim": spot_claim_path,
        "logs": logs,
    }


def _generate(evidence: dict[str, Path], output: Path, **kwargs: object) -> dict:
    return subject.generate_closeout(
        package_dir=evidence["package"],
        global_root_claim_path=evidence["root_claim"],
        global_spot_claim_path=evidence["spot_claim"],
        startup_log_tree=evidence["logs"],
        output_path=output,
        direct_counts={key: 0 for key in subject.COUNT_KEYS},
        **kwargs,
    )


def test_valid_evidence_writes_canonical_irrecoverable_receipt_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _build_evidence(tmp_path, monkeypatch)
    output = tmp_path / "closeout.json"
    receipt = _generate(evidence, output)

    assert output.read_bytes() == subject.canonical_bytes(receipt)
    assert receipt["startup_logs"]["count"] == 20
    assert receipt["control_plane_counts"] == {
        key: 0 for key in subject.COUNT_KEYS
    }
    assert receipt["disposition"]["current_run_irrecoverable"] is True
    assert receipt["disposition"]["attempt1_authorized"] is False
    assert receipt["disposition"]["alternate_seed_authorized"] is False
    assert receipt["disposition"]["alternate_package_authorized"] is False
    assert receipt["disposition"]["quality_pilot_authorized"] is False
    assert receipt["disposition"]["training_eligible"] is False
    assert receipt["disposition"]["current_profile_changed"] is False
    assert receipt["content_access"]["hand_content_opened"] is False
    assert receipt["content_access"]["root_content_opened"] is False
    assert receipt["content_access"]["result_content_opened"] is False
    assert receipt["content_access"]["cloud_query_performed"] is False

    with pytest.raises(FileExistsError, match="write-once"):
        _generate(evidence, output)


def test_canonical_metadata_snapshot_is_supported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _build_evidence(tmp_path, monkeypatch)
    snapshot = tmp_path / "snapshot.json"
    _write_json(
        snapshot,
        {
            "schema": subject.METADATA_SNAPSHOT_SCHEMA,
            "status": subject.METADATA_SNAPSHOT_STATUS,
            "run_name": subject.RUN_NAME,
            "counts": {key: 0 for key in subject.COUNT_KEYS},
            "metadata_only": True,
            "hand_content_opened": False,
            "root_content_opened": False,
            "result_content_opened": False,
            "cloud_mutated": False,
        },
    )
    receipt = subject.generate_closeout(
        package_dir=evidence["package"],
        global_root_claim_path=evidence["root_claim"],
        global_spot_claim_path=evidence["spot_claim"],
        startup_log_tree=evidence["logs"],
        output_path=tmp_path / "closeout.json",
        metadata_snapshot_path=snapshot,
    )
    assert (
        receipt["metadata_evidence"]["mode"]
        == "supplied_metadata_snapshot_no_cloud_query"
    )
    assert receipt["metadata_evidence"]["sha256"] == _sha(snapshot)


@pytest.mark.parametrize("key", subject.COUNT_KEYS)
def test_every_nonzero_control_plane_count_fails_closed(
    key: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _build_evidence(tmp_path, monkeypatch)
    counts = {name: 0 for name in subject.COUNT_KEYS}
    counts[key] = 1
    with pytest.raises(ValueError, match=f"{key} count must be exactly zero"):
        subject.generate_closeout(
            package_dir=evidence["package"],
            global_root_claim_path=evidence["root_claim"],
            global_spot_claim_path=evidence["spot_claim"],
            startup_log_tree=evidence["logs"],
            output_path=tmp_path / "closeout.json",
            direct_counts=counts,
        )


def test_missing_or_changed_startup_log_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _build_evidence(tmp_path, monkeypatch)
    missing = evidence["logs"] / "jobs" / subject.JOB_IDS[0] / "attempt-0" / "startup.log"
    missing.unlink()
    with pytest.raises(ValueError, match="exact 20/20"):
        _generate(evidence, tmp_path / "closeout.json")


def test_wrong_precontent_error_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _build_evidence(tmp_path, monkeypatch)
    path = evidence["logs"] / "jobs" / subject.JOB_IDS[0] / "attempt-0" / "startup.log"
    path.write_text(
        path.read_text(encoding="utf-8").replace(subject.ERROR_LINE, "other error"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exact pre-content failure"):
        _generate(evidence, tmp_path / "closeout.json")


def test_resume_control_or_package_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _build_evidence(tmp_path, monkeypatch)
    (evidence["package"] / "resume_claim.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden post-attempt0"):
        _generate(evidence, tmp_path / "closeout.json")

    (evidence["package"] / "resume_claim.json").unlink()
    manifest = evidence["package"] / subject.MANIFEST_NAME
    manifest.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(ValueError, match="claimed run anchor"):
        _generate(evidence, tmp_path / "closeout.json")

