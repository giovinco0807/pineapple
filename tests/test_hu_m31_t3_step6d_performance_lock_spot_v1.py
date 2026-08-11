from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_candidate02_performance_lock_plan as lock_plan,
)
from ofc_regular import hu_m31_t3_step6d_full100_spot_v1 as transport
from ofc_regular import hu_m31_t3_step6d_performance_lock_open as lock_open
from ofc_regular import hu_m31_t3_step6d_performance_lock_spot_v1 as subject


def _qualification_inputs() -> tuple[dict, dict, dict, dict]:
    plan = lock_plan.build_precontent_plan()
    claim = {
        "development_go": {
            "all_gates_passed": True,
            "performance_lock_authorized": True,
        }
    }
    seal = {
        "aggregate_root_sha256": "1" * 64,
        "root_topology_sha256": "2" * 64,
        "observation_fingerprint_sha256": "3" * 64,
        "development_comparison": {
            "lock_fingerprint_overlap_count": 0,
            "lock_root_hash_overlap_count": 0,
            "lock_seed_overlap_count": 0,
        },
    }
    materialization = {"schema": lock_open.MATERIALIZATION_SCHEMA}
    return plan, claim, seal, materialization


def _manifest(run_name: str = "regular-hu-m31-lock-test-001") -> dict:
    plan, claim, seal, materialization = _qualification_inputs()
    return {
        "schema": subject.PACKAGE_SCHEMA,
        "status": "immutable_performance_lock_package_ready_not_authorized",
        "run_name": run_name,
        "source_name": subject.SOURCE_NAME,
        "source_sha256": "a" * 64,
        "source_bytes": 1,
        "startup_name": subject.STARTUP_NAME,
        "startup_sha256": "b" * 64,
        "plan_package_path": subject.PLAN_PACKAGE_PATH,
        "plan_sha256": lock_plan.PRECONTENT_PLAN_SHA256,
        "tail_qualification": subject._lock_qualification(
            plan=plan,
            claim=claim,
            seal=seal,
            materialization=materialization,
        ),
        "run_contract": plan["run_contract"],
        "run_contract_digest": lock_plan.LOCK_RUN_CONTRACT_DIGEST,
        "accepted_reference": {},
        "accepted_candidate": {},
        "feature_encoder": {},
        "image": dict(lock_open.IMAGE),
        "allocation": dict(lock_open.ALLOCATION),
        "launch_target": transport.build_launch_target(),
        "cost_guard": transport.build_cost_guard(),
        "schedule": {},
        "job_manifests": [],
        "source_entries": {},
        "source_entry_count": 0,
        "checkpoint": {},
        "heartbeat": {},
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


def test_authorized_job_ids_are_exact_source_isolated_arithmetic_shards() -> None:
    assert subject.authorized_job_ids() == tuple(
        f"{role}-shard-{index:02d}"
        for role in ("candidate", "reference")
        for index in range(10)
    )


def test_lock_qualification_binds_claim_seal_and_zero_overlap() -> None:
    plan, claim, seal, materialization = _qualification_inputs()
    value = subject._lock_qualification(
        plan=plan,
        claim=claim,
        seal=seal,
        materialization=materialization,
    )
    assert value["summary_sha256"] == (lock_plan.OFFICIAL_DEVELOPMENT_SUMMARY_SHA256)
    assert value["open_claim_sha256"] == subject.canonical_sha256(claim)
    assert value["root_seal_sha256"] == subject.canonical_sha256(seal)
    assert value["materialization_sha256"] == subject.canonical_sha256(materialization)
    assert value["lock_development_overlap_count"] == 0

    tampered = deepcopy(seal)
    tampered["development_comparison"]["lock_seed_overlap_count"] = 1
    changed = subject._lock_qualification(
        plan=plan,
        claim=claim,
        seal=tampered,
        materialization=materialization,
    )
    assert changed["lock_development_overlap_count"] == 1


def test_package_fails_before_staging_when_global_claim_is_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = lock_open.PerformanceLockInputs(
        repository_root=Path.cwd(),
        plan_path=subject.DEFAULT_PLAN_PATH,
        lock_output_directory=tmp_path / "lock-output",
        candidate_library=tmp_path / "candidate.so",
        reference_library=tmp_path / "reference.so",
        feature_encoder=tmp_path / "feature.so",
        startup_source=Path.cwd() / "scripts/startup_hu_m31_t3_step6d_full100_v1.sh",
    )

    def reject(_inputs):
        raise ValueError("claim replay failed")

    monkeypatch.setattr(lock_open, "validate_open_claim", reject)
    destination = tmp_path / "package"
    with pytest.raises(ValueError, match="claim replay failed"):
        subject.package_performance_lock(
            output_dir=destination,
            run_name="regular-hu-m31-lock-test-001",
            lock_inputs=inputs,
        )
    assert not destination.exists()
    assert not destination.with_name(
        f".{destination.name}.{__import__('os').getpid()}.staging"
    ).exists()


def test_authorization_is_lock_only_and_write_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    subject._write_once(tmp_path / subject.MANIFEST_NAME, manifest)
    monkeypatch.setattr(subject, "validate_package", lambda _path: manifest)
    monkeypatch.setattr(
        subject,
        "_acquire_global_spot_claim",
        lambda **_kwargs: {"claimed": True},
    )
    monkeypatch.setattr(
        subject,
        "validate_global_spot_claim",
        lambda *_args, **_kwargs: {"claimed": True},
    )

    authorization = subject.authorize_launch(tmp_path)
    assert authorization["performance_development_only"] is False
    assert authorization["global_spot_claim"] == {"claimed": True}
    assert authorization["spot_execution_authorized"] is True
    assert authorization["performance_lock_authorized"] is True
    assert authorization["quality_pilot_authorized"] is False
    assert authorization["training_eligible"] is False
    assert authorization["current_profile_changed"] is False

    with pytest.raises(FileExistsError):
        subject.authorize_launch(tmp_path)


def test_lifecycle_mutex_serializes_resume_and_result_open(
    tmp_path: Path,
) -> None:
    with subject._lifecycle_mutex(tmp_path, "resume"):
        with pytest.raises(RuntimeError, match="lifecycle is busy"):
            with subject._lifecycle_mutex(tmp_path, "result-open"):
                pytest.fail("concurrent lifecycle mutation must not run")


def test_global_spot_claim_rejects_second_package_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root_claim_path = tmp_path / "root-claim.json"
    root_claim = {
        "schema": lock_open.CLAIM_SCHEMA,
        "global_claim_path": str(root_claim_path.resolve()),
        "lock_output_directory": str((tmp_path / "roots").resolve()),
    }
    subject._write_once(root_claim_path, root_claim)
    global_spot_path = tmp_path / subject.GLOBAL_SPOT_CLAIM_NAME
    monkeypatch.setattr(
        subject,
        "_read_packaged_open_claim",
        lambda *_args, **_kwargs: root_claim,
    )
    monkeypatch.setattr(
        subject,
        "_global_spot_claim_path",
        lambda _claim: global_spot_path,
    )

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = _manifest("regular-hu-m31-lock-first-001")
    second = _manifest("regular-hu-m31-lock-second-001")
    for target, manifest in ((first_dir, first), (second_dir, second)):
        manifest["tail_qualification"]["open_claim_sha256"] = subject.canonical_sha256(
            root_claim
        )
        subject._write_once(target / subject.MANIFEST_NAME, manifest)

    acquired = subject._acquire_global_spot_claim(
        target=first_dir,
        manifest=first,
    )
    assert acquired["run_name"] == first["run_name"]
    assert acquired["alternate_package_authorization_allowed"] is False

    with pytest.raises(ValueError, match="global performance-lock Spot claim changed"):
        subject._acquire_global_spot_claim(
            target=second_dir,
            manifest=second,
        )


def test_global_spot_claim_preserves_signed_windows_path_strings(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    subject._write_once(tmp_path / subject.MANIFEST_NAME, manifest)
    root_claim = {
        "global_claim_path": (
            r"c:\users\owner\repo\outputs\global_performance_lock_claim.json"
        ),
        "lock_output_directory": r"d:\ofc-gcp-runs\lock\roots-open",
    }

    claim = subject._global_spot_claim_payload(
        target=tmp_path,
        manifest=manifest,
        root_claim=root_claim,
        claimed_unix_ns=1,
    )

    assert claim["global_root_claim_path"] == root_claim["global_claim_path"]
    assert claim["lock_output_directory"] == root_claim["lock_output_directory"]
    assert claim["global_root_claim_path"].startswith(r"c:\users")
    assert claim["lock_output_directory"].startswith(r"d:\ofc-gcp-runs")


def test_launch_claim_cannot_authorize_quality_or_reuse(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    subject._write_once(tmp_path / subject.MANIFEST_NAME, manifest)
    authorization = {
        "schema": subject.AUTHORIZATION_SCHEMA,
        "status": "test",
    }
    subject._write_once(tmp_path / subject.AUTHORIZATION_NAME, authorization)
    preflight = {"status": "test"}

    claim = subject._acquire_launch_claim(
        target=tmp_path,
        manifest=manifest,
        preflight=preflight,
    )
    assert claim["run_contract_digest"] == lock_plan.LOCK_RUN_CONTRACT_DIGEST
    assert claim["crash_reuse_authorized"] is False
    assert "quality_pilot_authorized" not in claim
    with pytest.raises(FileExistsError):
        subject._acquire_launch_claim(
            target=tmp_path,
            manifest=manifest,
            preflight=preflight,
        )


def test_resume_preflight_uses_done_existence_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    incomplete = "candidate-shard-00"
    monkeypatch.setattr(
        subject,
        "validate_launch_chain",
        lambda _path: (manifest, {}, {}),
    )
    monkeypatch.setattr(
        subject.transport,
        "_instance_rows",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        subject.transport,
        "_object_exists",
        lambda uri, **_kwargs: incomplete not in uri,
    )
    monkeypatch.setattr(
        subject.transport,
        "_quota_snapshot",
        lambda **_kwargs: {
            "CPUS": {"limit": 1000, "usage": 0, "available": 1000},
            "PREEMPTIBLE_CPUS": {
                "limit": 1000,
                "usage": 0,
                "available": 1000,
            },
        },
    )
    monkeypatch.setattr(
        subject.tail_spot,
        "_run",
        lambda *_args, **_kwargs: pytest.fail(
            "resume preflight must not download result content"
        ),
    )

    result = subject.preflight_resume(
        run_dir=tmp_path,
        selected=[incomplete],
    )
    assert result["incomplete_job_ids"] == [incomplete]
    assert result["result_content_read"] is False
    assert result["all_incomplete_jobs_selected"] is True


def test_resume_chain_replays_attempt1_and_rejects_third_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    plan = lock_plan.build_precontent_plan()
    manifest["job_manifests"] = [
        {
            "job_id": row["job_id"],
            "source_role": row["source_role"],
            "shard_index": row["shard_index"],
            "work_hand_indices": row["work_hand_indices"],
        }
        for row in plan["jobs"]
    ]
    for name in (
        subject.MANIFEST_NAME,
        subject.AUTHORIZATION_NAME,
        subject.LAUNCH_CLAIM_NAME,
        subject.LAUNCH_RESULT_NAME,
    ):
        subject._write_once(tmp_path / name, {"name": name})
    selected = ["candidate-shard-00"]
    preflight = {"metadata_only": True}
    monkeypatch.setattr(
        subject,
        "_validate_resume_preflight",
        lambda value, **_kwargs: dict(value),
    )
    claim = {
        "schema": subject.RESUME_CLAIM_SCHEMA,
        "status": ("exclusive_performance_lock_attempt1_claim_before_remote_mutation"),
        "run_name": manifest["run_name"],
        "attempt_index": 1,
        "selected_job_ids": selected,
        "initial_launch_claim_sha256": subject.sha256_file(
            tmp_path / subject.LAUNCH_CLAIM_NAME
        ),
        "initial_launch_result_sha256": subject.sha256_file(
            tmp_path / subject.LAUNCH_RESULT_NAME
        ),
        "package_manifest_sha256": subject.sha256_file(
            tmp_path / subject.MANIFEST_NAME
        ),
        "launch_authorization_sha256": subject.sha256_file(
            tmp_path / subject.AUTHORIZATION_NAME
        ),
        "plan_sha256": lock_plan.PRECONTENT_PLAN_SHA256,
        "run_contract_digest": lock_plan.LOCK_RUN_CONTRACT_DIGEST,
        "launch_target": subject.transport.build_launch_target(),
        "cost_guard_sha256": subject.canonical_sha256(manifest["cost_guard"]),
        "preflight_sha256": subject.canonical_sha256(preflight),
        "claimed_unix_seconds": time.time(),
        "third_attempt_authorized": False,
    }
    subject._write_once(tmp_path / subject.RESUME_CLAIM_NAME, claim)
    row = manifest["job_manifests"][0]
    result = {
        "schema": subject.RESUME_RESULT_SCHEMA,
        "status": "performance_lock_resume_jobs_created",
        "run_name": manifest["run_name"],
        "attempt_index": 1,
        "selected_job_ids": selected,
        "created": [
            {
                "job_id": selected[0],
                "source_role": row["source_role"],
                "shard_index": row["shard_index"],
                "work_hand_indices": row["work_hand_indices"],
                "instance": subject.transport._instance_name(manifest, selected[0], 1),
                "zone": subject.DEFAULT_ZONES[0],
                "attempt_index": 1,
                "status": "created",
            }
        ],
        "failures": [],
        "cleanup": [],
        "cleanup_compute_stopped_or_absent": True,
        "logical_job_count": 1,
        "max_resume_jobs": subject.MAX_LOGICAL_JOBS,
        "max_cumulative_vm_jobs": subject.transport.MAX_CUMULATIVE_VM_JOBS,
        "run_contract_digest": lock_plan.LOCK_RUN_CONTRACT_DIGEST,
        "launch_target": manifest["launch_target"],
        "cost_guard": manifest["cost_guard"],
        "preflight": preflight,
        "resume_claim_sha256": subject.sha256_file(
            tmp_path / subject.RESUME_CLAIM_NAME
        ),
        "initial_launch_claim_sha256": subject.sha256_file(
            tmp_path / subject.LAUNCH_CLAIM_NAME
        ),
        "initial_launch_result_sha256": subject.sha256_file(
            tmp_path / subject.LAUNCH_RESULT_NAME
        ),
        "third_attempt_authorized": False,
        "performance_lock_authorized": True,
        "quality_pilot_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
        "runtime_policy_activated": False,
    }
    subject._write_once(tmp_path / subject.RESUME_RESULT_NAME, result)

    replayed = subject.validate_resume_chain(
        run_dir=tmp_path,
        manifest=manifest,
    )
    assert replayed == (claim, result)

    tampered = deepcopy(result)
    tampered["third_attempt_authorized"] = True
    monkeypatch.setattr(
        subject,
        "_read_canonical",
        lambda path, _label: (
            claim if Path(path).name == subject.RESUME_CLAIM_NAME else tampered
        ),
    )
    with pytest.raises(ValueError, match="resume hash chain changed"):
        subject.validate_resume_chain(
            run_dir=tmp_path,
            manifest=manifest,
        )


def test_result_claim_blocks_download_while_compute_is_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    monkeypatch.setattr(
        subject,
        "validate_launch_chain",
        lambda _path: (manifest, {}, {}),
    )
    monkeypatch.setattr(
        subject,
        "validate_resume_chain",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        subject,
        "_active_compute_instances",
        lambda *_args, **_kwargs: ["still-running"],
    )
    monkeypatch.setattr(
        subject.tail_spot,
        "_run",
        lambda *_args, **_kwargs: pytest.fail(
            "DONE must not be downloaded while compute is active"
        ),
    )
    with pytest.raises(RuntimeError, match="compute is still active"):
        subject.claim_results(run_dir=tmp_path)


def test_received_done_must_equal_precontent_claim() -> None:
    jobs = [
        {
            "job_id": "candidate-shard-00",
            "source_role": "candidate",
            "work_hand_indices": list(range(10)),
            "done_sha256": "a" * 64,
        }
    ]
    markers = [
        {
            "job_id": identifier,
            "source_role": identifier.split("-", 1)[0],
            "work_hand_indices": (
                list(range(10)) if identifier == "candidate-shard-00" else []
            ),
            "file_sha256": (
                "b" * 64 if identifier == "candidate-shard-00" else "c" * 64
            ),
        }
        for identifier in subject.authorized_job_ids()
    ]
    with pytest.raises(ValueError, match="received DONE differs from claim"):
        subject._validate_received_done_against_claim(
            jobs=jobs,
            result_claim={"done_markers": markers},
        )


def test_receive_file_set_rejects_untracked_content(tmp_path: Path) -> None:
    job_dir = tmp_path / "jobs/candidate-shard-00"
    (job_dir / "roots").mkdir(parents=True)
    (job_dir / "hands/candidate").mkdir(parents=True)
    for relative in subject._expected_received_relatives("candidate", []):
        path = job_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    for name in (
        "receive_receipt.json",
        subject.RESULT_OPEN_CLAIM_NAME,
        subject.MANIFEST_NAME,
        subject.AUTHORIZATION_NAME,
    ):
        (tmp_path / name).touch()
    jobs = [
        {
            "job_id": "candidate-shard-00",
            "source_role": "candidate",
            "work_hand_indices": [],
        }
    ]
    subject._validate_receive_file_set(receive_dir=tmp_path, jobs=jobs)
    (tmp_path / "untracked.txt").touch()
    with pytest.raises(ValueError, match="receive file set changed"):
        subject._validate_receive_file_set(receive_dir=tmp_path, jobs=jobs)
