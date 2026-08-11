from __future__ import annotations

import hashlib
import os
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import pytest

from ofc_regular import hu_m31_t3_step6d_candidate02_full100_plan as plan_subject
from ofc_regular import hu_m31_t3_step6d_full100_spot_v1 as subject


REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_LIBRARY = (
    REPO_ROOT / "outputs/gcp_runs/regular-hu-m31-c02-tail-v2-20260717-001/"
    "package_src/native/candidate/release/libofc_hu_m3_engine.so"
)


@pytest.fixture(scope="module")
def authorized_package(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("full100-package") / "run"
    subject.package_full100(
        output_dir=output,
        run_name="full100-lifecycle-test-001",
        candidate_library=CANDIDATE_LIBRARY,
    )
    subject.authorize_launch(output)
    return output


def _clone_package(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination, copy_function=os.link)
    return destination


def _quota() -> dict[str, dict[str, int]]:
    return {
        metric: {"limit": 400, "usage": 0, "available": 400}
        for metric in ("CPUS", "PREEMPTIBLE_CPUS")
    }


def _created(
    manifest: Mapping[str, Any], selected: tuple[str, ...], attempt: int
) -> list[dict[str, Any]]:
    values = []
    for identifier in selected:
        record = subject._job_record_by_id(manifest, identifier)
        ordinal = subject.authorized_job_ids().index(identifier)
        values.append(
            {
                "job_id": identifier,
                "source_role": record["source_role"],
                "shard_index": record["shard_index"],
                "work_hand_indices": record["work_hand_indices"],
                "instance": subject._instance_name(manifest, identifier, attempt),
                "zone": subject.DEFAULT_ZONES[ordinal % len(subject.DEFAULT_ZONES)],
                "attempt_index": attempt,
                "status": "created",
            }
        )
    return values


def _mock_initial_launch(
    monkeypatch: pytest.MonkeyPatch, run_dir: Path
) -> dict[str, Any]:
    manifest = subject.validate_package(run_dir)
    monkeypatch.setattr(subject, "_quota_snapshot", lambda **_kwargs: _quota())
    monkeypatch.setattr(subject, "_instance_rows", lambda **_kwargs: [])
    monkeypatch.setattr(subject, "_object_exists", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        subject,
        "_publish_package",
        lambda **_kwargs: (
            f"gs://{subject.DEFAULT_BUCKET}/runs/{manifest['run_name']}/full100"
        ),
    )
    monkeypatch.setattr(
        subject.tail_spot, "_publish_once", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        subject,
        "_launch_selected",
        lambda **kwargs: (_created(manifest, tuple(kwargs["selected"]), 0), []),
    )
    return subject.launch_jobs(run_dir=run_dir)


def test_package_authorization_and_phase_cost_are_bounded(
    authorized_package: Path,
) -> None:
    manifest, authorization = subject.validate_launch_authorization(authorized_package)
    guard = subject.validate_cost_guard(manifest["cost_guard"])

    assert len(manifest["job_manifests"]) == 20
    assert all(len(row["work_hand_indices"]) == 10 for row in manifest["job_manifests"])
    assert authorization["authorized_job_ids"] == list(subject.authorized_job_ids())
    assert guard["max_concurrent_vms"] == 20
    assert guard["max_attempts_per_job"] == 2
    assert guard["all_attempts_estimated_max_compute_usd"] == pytest.approx(
        23.333333333333332
    )
    assert (
        guard["all_attempts_estimated_max_compute_usd"]
        <= guard["phase_compute_cap_usd"]
        < guard["m31_total_compute_cap_usd"]
    )
    assert manifest["current_profile_changed"] is False


def test_launch_preflight_is_exact_all20_and_has_no_remote_mutation(
    monkeypatch: pytest.MonkeyPatch, authorized_package: Path
) -> None:
    manifest = subject.validate_package(authorized_package)
    monkeypatch.setattr(subject, "_quota_snapshot", lambda **_kwargs: _quota())
    monkeypatch.setattr(subject, "_instance_rows", lambda **_kwargs: [])
    monkeypatch.setattr(subject, "_object_exists", lambda *_args, **_kwargs: False)

    preflight = subject.preflight_launch(
        manifest=manifest,
        selected=subject.authorized_job_ids(),
    )
    assert preflight["required_vcpus"] == 320
    assert preflight["max_concurrent_vms"] == 20
    assert preflight["selected_estimated_max_compute_usd"] == pytest.approx(
        11.666666666666666
    )
    assert subject._validate_launch_preflight(preflight, manifest=manifest) == preflight
    with pytest.raises(ValueError, match="exactly all twenty"):
        subject.preflight_launch(
            manifest=manifest,
            selected=subject.authorized_job_ids()[:-1],
        )


def test_launch_and_one_retry_hash_chains_are_write_once_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
    authorized_package: Path,
    tmp_path: Path,
) -> None:
    run_dir = _clone_package(authorized_package, tmp_path / "run")
    result = _mock_initial_launch(monkeypatch, run_dir)
    manifest = subject.validate_package(run_dir)
    subject.validate_launch_chain(run_dir=run_dir, manifest=manifest)
    assert result["logical_job_count"] == 20

    selected = subject.authorized_job_ids()[:2]
    preflight = {
        "schema": subject.RESUME_PREFLIGHT_SCHEMA,
        "status": "full100_attempt1_checks_passed",
        "checked_unix_seconds": 1.0,
        "region": subject.DEFAULT_REGION,
        "attempt_index": 1,
        "selected_job_ids": list(selected),
        "selected_initial_instance_names": [
            subject._instance_name(manifest, job, 0) for job in selected
        ],
        "selected_resume_instance_names": [
            subject._instance_name(manifest, job, 1) for job in selected
        ],
        "required_vcpus": 2 * subject.VCPUS_PER_VM,
        "quota": _quota(),
        "no_active_initial_instances": True,
        "resume_instance_names_absent": True,
        "validated_completed_job_ids": [
            job for job in subject.authorized_job_ids() if job not in set(selected)
        ],
        "all_incomplete_jobs_selected": True,
        "cost_guard_sha256": subject.canonical_sha256(manifest["cost_guard"]),
        "max_cumulative_vm_jobs": subject.MAX_CUMULATIVE_VM_JOBS,
        "all_attempts_estimated_max_compute_usd": (
            subject.ALL_ATTEMPTS_ESTIMATED_MAX_COMPUTE_USD
        ),
        "phase_compute_cap_usd": subject.PHASE_COMPUTE_CAP_USD,
    }
    monkeypatch.setattr(subject, "preflight_resume", lambda **_kwargs: preflight)
    monkeypatch.setattr(
        subject,
        "_launch_selected",
        lambda **kwargs: (_created(manifest, tuple(kwargs["selected"]), 1), []),
    )
    resumed = subject.resume_jobs(run_dir=run_dir, selected=selected)
    assert resumed["attempt_index"] == 1
    assert resumed["third_attempt_authorized"] is False
    subject.validate_resume_chain(run_dir=run_dir, manifest=manifest)
    with pytest.raises(FileExistsError, match="already claimed"):
        subject.resume_jobs(run_dir=run_dir, selected=selected)


def _artifact_sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _receipt() -> dict[str, Any]:
    plan = plan_subject.build_full100_plan()
    jobs = []
    for row in plan["jobs"]:
        identifier = row["job_id"]
        role = row["source_role"]
        work = list(row["work_hand_indices"])
        jobs.append(
            {
                "job_id": identifier,
                "source_role": role,
                "shard_index": row["shard_index"],
                "work_hand_indices": work,
                "done_path": f"jobs/{identifier}/DONE.json",
                "done_sha256": _artifact_sha(f"done-{identifier}"),
                "root_artifacts": [
                    {
                        "hand_index": index,
                        "path": f"jobs/{identifier}/roots/hand_{index:03d}.json",
                        "sha256": subject.sha256_file(
                            plan_subject.DEFAULT_ROOT_DIR / f"hand_{index:03d}.json"
                        ),
                    }
                    for index in work
                ],
                "source_hand_artifacts": [
                    {
                        "hand_index": index,
                        "path": (
                            f"jobs/{identifier}/hands/{role}/" f"hand_{index:03d}.json"
                        ),
                        "sha256": _artifact_sha(f"hand-{role}-{index}"),
                    }
                    for index in work
                ],
                "run_contract_digest": plan_subject.FULL_RUN_CONTRACT_DIGEST,
            }
        )
    return {
        "schema": subject.RECEIVE_SCHEMA,
        "status": "exact_full100_source_shards_received_and_validated",
        "run_name": "full100-lifecycle-test-001",
        "package_manifest_sha256": "1" * 64,
        "launch_authorization_sha256": "2" * 64,
        "launch_claim_sha256": "3" * 64,
        "launch_result_sha256": "4" * 64,
        "resume_claim_sha256": None,
        "resume_result_sha256": None,
        "plan_sha256": plan_subject.FULL100_PLAN_SHA256,
        "tail_summary_sha256": plan_subject.TAIL_SUMMARY_SHA256,
        "tail_validation_sha256": plan_subject.TAIL_VALIDATION_SHA256,
        "run_contract_digest": plan_subject.FULL_RUN_CONTRACT_DIGEST,
        "launch_target": subject.build_launch_target(),
        "work_hand_indices": list(range(100)),
        "source_roles": ["candidate", "reference"],
        "logical_job_count": 20,
        "paired_hand_count": 100,
        "root_count": 200,
        "jobs": jobs,
        "candidate_done_paths": [
            f"jobs/{identifier}/DONE.json"
            for identifier in subject.authorized_job_ids()[:10]
        ],
        "reference_done_paths": [
            f"jobs/{identifier}/DONE.json"
            for identifier in subject.authorized_job_ids()[10:]
        ],
        "source_isolation_validated": True,
        "root_pairing_validated": True,
        "merge_executed": False,
        "performance_lock_authorized": False,
        "quality_pilot_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
    }


def test_receipt_deep_mapping_and_artifact_tamper_fail_closed() -> None:
    receipt = _receipt()
    assert subject.validate_receive_receipt(receipt) == receipt

    mutations = []
    extra = deepcopy(receipt)
    extra["unexpected"] = False
    mutations.append(extra)
    path = deepcopy(receipt)
    path["jobs"][0]["root_artifacts"][0]["path"] = "roots/escaped.json"
    mutations.append(path)
    role = deepcopy(receipt)
    role["jobs"][0]["source_role"] = "reference"
    mutations.append(role)
    root = deepcopy(receipt)
    root["jobs"][10]["root_artifacts"][0]["sha256"] = "f" * 64
    mutations.append(root)
    coverage = deepcopy(receipt)
    coverage["jobs"][0]["root_artifacts"][0]["hand_index"] = 99
    mutations.append(coverage)

    for value in mutations:
        with pytest.raises(ValueError):
            subject.validate_receive_receipt(value)


def test_validate_received_directory_returns_absolute_source_isolated_done_paths(
    monkeypatch: pytest.MonkeyPatch,
    authorized_package: Path,
    tmp_path: Path,
) -> None:
    receive_dir = _clone_package(authorized_package, tmp_path / "received")
    launch_claim = {"status": "test-launch-claim"}
    launch_result = {"status": "test-launch-result"}
    subject._write_once(receive_dir / subject.LAUNCH_CLAIM_NAME, launch_claim)
    subject._write_once(receive_dir / subject.LAUNCH_RESULT_NAME, launch_result)

    jobs = _receipt()["jobs"]
    jobs_by_id = {row["job_id"]: row for row in jobs}
    for row in jobs:
        done_path = receive_dir / row["done_path"]
        done_path.parent.mkdir(parents=True, exist_ok=True)
        subject._write_once(done_path, {"job_id": row["job_id"]})

    manifest = subject.validate_package(receive_dir)
    receipt = subject.validate_receive_receipt(
        subject._build_receive_receipt(
            target=receive_dir,
            manifest=manifest,
            jobs=jobs,
            resume_present=False,
        )
    )
    subject._write_once(receive_dir / "receive_receipt.json", receipt)

    monkeypatch.setattr(
        subject,
        "validate_launch_chain",
        lambda **_kwargs: {"status": "validated"},
    )
    monkeypatch.setattr(subject, "validate_resume_chain", lambda **_kwargs: None)
    monkeypatch.setattr(
        subject,
        "_validate_received_job",
        lambda **kwargs: jobs_by_id[kwargs["record"]["job_id"]],
    )

    validation = subject.validate_received_directory(
        receive_dir,
        expected_run_name="full100-lifecycle-test-001",
    )
    candidate_paths = validation["candidate_done_paths"]
    reference_paths = validation["reference_done_paths"]
    assert len(candidate_paths) == len(reference_paths) == 10
    assert all(Path(path).is_absolute() for path in candidate_paths + reference_paths)
    assert all(Path(path).is_file() for path in candidate_paths + reference_paths)
    assert not set(candidate_paths) & set(reference_paths)


def test_authorization_tamper_fails_closed(
    authorized_package: Path, tmp_path: Path
) -> None:
    run_dir = _clone_package(authorized_package, tmp_path / "tampered")
    path = run_dir / subject.AUTHORIZATION_NAME
    value = subject._read_canonical(path, "authorization")
    value["performance_lock_authorized"] = True
    path.unlink()
    path.write_bytes(subject.canonical_bytes(value))
    with pytest.raises(ValueError, match="authorization changed"):
        subject.validate_launch_authorization(run_dir)
