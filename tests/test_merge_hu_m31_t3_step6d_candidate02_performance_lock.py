from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_candidate02_performance_lock_plan as lock_plan,
)
from ofc_regular import hu_m31_t3_step6d_performance_lock_open as lock_open
from ofc_regular import hu_m31_t3_step6d_performance_lock_spot_v1 as lock_spot
from ofc_regular import (
    merge_hu_m31_t3_step6d_candidate02_performance_lock as subject,
)
from ofc_regular import merge_hu_m31_t3_step6d_performance_v2 as base
from ofc_regular import run_hu_m31_t3_step6d_performance_v2 as runner
from ofc_regular.hu_m31_t3_behavior_roots import M31_T3_BEHAVIOR_PROFILES


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(runner.canonical_bytes(value))


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha(index: int) -> str:
    return f"{index + 1:064x}"


@dataclass
class _Fixture:
    inputs: lock_open.PerformanceLockInputs
    receive_inputs: subject.PerformanceLockReceiveInputs
    plan: dict[str, Any]
    generic: dict[str, Any]
    claim_path: Path
    materialization_path: Path
    seal_path: Path
    done_paths: dict[str, list[Path]]
    producer_calls: dict[str, int]


def _performance(
    *,
    first_p95: float = 150.0,
    first_p99: float = 240.0,
    first_max: float = 240.0,
    second_p95: float = 5.0,
    peak_rss: int = 858_993_459,
) -> dict[str, Any]:
    return {
        "reference_by_seat": {},
        "candidate_by_seat": {
            "first": {
                "count": 100,
                "p95_seconds": first_p95,
                "p99_seconds": first_p99,
                "max_seconds": first_max,
            },
            "second": {
                "count": 100,
                "p95_seconds": second_p95,
                "p99_seconds": second_p95,
                "max_seconds": second_p95,
            },
        },
        "first_paired_speedups": [],
        "first_geometric_mean_speedup_diagnostic": 1.0,
        "tail_heavy_first": None,
        "peak_source_process_rss_bytes": peak_rss,
    }


@pytest.fixture
def lock_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Fixture:
    plan = lock_plan.build_precontent_plan()
    plan_path = tmp_path / "precontent-plan.json"
    _write(plan_path, plan)

    output = tmp_path / "lock-output"
    run_dir = tmp_path / "spot-run"
    receive_dir = tmp_path / "spot-receive"
    claim_path = tmp_path / "global" / "GLOBAL_PERFORMANCE_LOCK_CLAIM.json"
    materialization_path = output / "materialization.json"
    seal_path = output / "seal.json"
    candidate_library = tmp_path / "candidate.so"
    reference_library = tmp_path / "reference.so"
    feature_encoder = tmp_path / "feature.py"
    startup_source = tmp_path / "startup.sh"
    for path in (
        candidate_library,
        reference_library,
        feature_encoder,
        startup_source,
    ):
        path.write_bytes(path.name.encode("ascii"))

    inputs = lock_open.PerformanceLockInputs(
        repository_root=tmp_path / "repository",
        plan_path=plan_path,
        lock_output_directory=output,
        candidate_library=candidate_library,
        reference_library=reference_library,
        feature_encoder=feature_encoder,
        startup_source=startup_source,
        development_summary_path=tmp_path / "development-summary.json",
        development_validation_path=tmp_path / "development-validation.json",
        development_root_directory=tmp_path / "development-roots",
        global_claim_path=claim_path,
    )
    receive_inputs = subject.PerformanceLockReceiveInputs(
        run_dir=run_dir,
        receive_dir=receive_dir,
        project="test-project",
        bucket="test-bucket",
    )
    source_identity = plan["source_identity"]
    claim = {
        "precontent_plan": {
            "path": str(plan_path.resolve()),
            "sha256": lock_open.sha256_file(plan_path),
            "bytes": plan_path.stat().st_size,
            "schema": lock_plan.PLAN_SCHEMA,
            "canonical_sha256": lock_plan.canonical_sha256(plan),
        },
        "global_claim_path": os.path.normcase(os.path.abspath(os.fspath(claim_path))),
        "lock_output_directory": os.path.normcase(os.path.abspath(os.fspath(output))),
        "lock_run_contract": plan["run_contract"],
        "lock_run_contract_digest": plan["run_contract_digest"],
        "image": plan["image"],
        "allocation": plan["allocation"],
        "seed_contract": plan["root_contract"]["seed_contract"],
        "accepted_binaries": {
            "candidate": {"sha256": source_identity["candidate_library_sha256"]},
            "reference": {"sha256": source_identity["reference_library_sha256"]},
            "feature_encoder": {"sha256": source_identity["feature_encoder_sha256"]},
        },
        "ai_profiles_current": {
            "sha256": source_identity["current_profile_registry_sha256"]
        },
        "restrictions": {
            "timing_used_for_root_selection": False,
            "q_used_for_root_selection": False,
            "ev_used_for_root_selection": False,
            "alternate_seed_allowed": False,
            "reseed_allowed": False,
            "cloud_authorized": False,
            "training_authorized": False,
            "promotion_authorized": False,
            "current_profile_resolution_allowed": False,
            "runtime_activation_allowed": False,
            "opponent_private_discards_allowed": False,
        },
    }
    _write(claim_path, claim)

    root_hashes = [_sha(index) for index in runner.CONTRACT_HAND_INDICES]
    materialization = {
        "schema": lock_open.MATERIALIZATION_SCHEMA,
        "status": "all_100_lock_roots_materialized_same_identity",
        "global_claim_sha256": lock_open.canonical_sha256(claim),
        "plan_sha256": lock_open.sha256_file(plan_path),
        "run_contract_digest": plan["run_contract_digest"],
        "hand_indices": list(runner.CONTRACT_HAND_INDICES),
        "root_count": 100,
        "root_artifact_sha256": root_hashes,
        "aggregate_root_sha256": lock_open.canonical_sha256(root_hashes),
        "same_identity_resume_only": True,
        "reseeded": False,
        "training_eligible": False,
        "current_profile_changed": False,
    }
    _write(materialization_path, materialization)
    seal = {
        "schema": lock_open.SEAL_SCHEMA,
        "status": "sealed_100_disjoint_hidden_safe_performance_lock_roots",
        "global_claim_sha256": lock_open.canonical_sha256(claim),
        "materialization_sha256": lock_open.canonical_sha256(materialization),
        "plan_sha256": lock_open.sha256_file(plan_path),
        "run_contract_digest": plan["run_contract_digest"],
        "hand_indices": list(runner.CONTRACT_HAND_INDICES),
        "root_count": 100,
        "observation_count": 200,
        "profile_counts": {profile: 20 for profile in M31_T3_BEHAVIOR_PROFILES},
        "seat_counts": {"first": 100, "second": 100},
        "root_artifact_sha256": root_hashes,
        "aggregate_root_sha256": lock_open.canonical_sha256(root_hashes),
        "root_topology_sha256": _sha(1000),
        "observation_fingerprint_sha256": _sha(1001),
        "root_artifact_unique": True,
        "observation_fingerprint_unique": True,
        "development_comparison": {
            "development_all100_root_sha256": _sha(1002),
            "development_root_count": 100,
            "lock_fingerprint_overlap_count": 0,
            "lock_root_hash_overlap_count": 0,
            "lock_seed_overlap_count": 0,
        },
        "visibility": {
            "runner_validator_replayed_all_roots": True,
            "first_observation_count": 100,
            "second_observation_count": 100,
            "opponent_private_discards_used": False,
            "current_profile_resolved": False,
        },
        "selection_inputs": {
            "timing_used": False,
            "q_used": False,
            "ev_used": False,
            "all_100_preregistered_hands_used": True,
        },
        "training_eligible": False,
        "quality_evidence": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
    }
    _write(seal_path, seal)

    producer_calls = {"claim": 0, "seal": 0, "receive": 0}

    def validate_claim(
        supplied: lock_open.PerformanceLockInputs,
    ) -> dict[str, Any]:
        assert Path(supplied.global_claim_path).resolve() == claim_path.resolve()
        producer_calls["claim"] += 1
        return _read(claim_path)

    def validate_seal(
        supplied: lock_open.PerformanceLockInputs,
    ) -> dict[str, Any]:
        assert Path(supplied.lock_output_directory).resolve() == output.resolve()
        producer_calls["seal"] += 1
        return _read(seal_path)

    monkeypatch.setattr(lock_open, "validate_open_claim", validate_claim)
    monkeypatch.setattr(lock_open, "validate_root_seal", validate_seal)

    done_paths: dict[str, list[Path]] = {
        "candidate": [],
        "reference": [],
    }
    source_done_inputs: dict[str, list[dict[str, Any]]] = {
        "candidate": [],
        "reference": [],
    }
    for job in plan["jobs"]:
        role = job["source_role"]
        path = receive_dir / "jobs" / job["job_id"] / "DONE.json"
        _write(path, {"job_id": job["job_id"]})
        done_paths[role].append(path)
        source_done_inputs[role].append(
            {
                "path": str(path.resolve()),
                "sha256": lock_open.sha256_file(path),
                "shard_manifest_digest": job["shard_manifest_sha256"],
                "work_hand_indices": list(job["work_hand_indices"]),
            }
        )
    result_claim_path = receive_dir / lock_spot.RESULT_OPEN_CLAIM_NAME
    _write(result_claim_path, {"schema": lock_spot.RESULT_OPEN_CLAIM_SCHEMA})
    received_jobs = [
        {
            "job_id": job["job_id"],
            "source_role": job["source_role"],
            "work_hand_indices": list(job["work_hand_indices"]),
            "done_path": f"jobs/{job['job_id']}/DONE.json",
            "done_sha256": lock_open.sha256_file(
                receive_dir / "jobs" / job["job_id"] / "DONE.json"
            ),
        }
        for job in plan["jobs"]
    ]
    receipt = {
        "schema": lock_spot.RECEIVE_SCHEMA,
        "status": "exact_performance_lock_source_shards_received_and_validated",
        "run_name": "test-one-shot-lock",
        "package_manifest_sha256": _sha(4000),
        "launch_authorization_sha256": _sha(4001),
        "launch_claim_sha256": _sha(4002),
        "launch_result_sha256": _sha(4003),
        "resume_claim_sha256": None,
        "resume_result_sha256": None,
        "result_open_claim_sha256": lock_open.sha256_file(result_claim_path),
        "precontent_plan_sha256": lock_open.sha256_file(plan_path),
        "root_open_claim_sha256": lock_open.sha256_file(claim_path),
        "root_seal_sha256": lock_open.sha256_file(seal_path),
        "run_contract_digest": plan["run_contract_digest"],
        "source_roles": list(runner.SOURCE_ROLES),
        "logical_job_count": 20,
        "paired_hand_count": 100,
        "root_count": 200,
        "jobs": received_jobs,
        "candidate_done_paths": [
            row["done_path"]
            for row in received_jobs
            if row["source_role"] == "candidate"
        ],
        "reference_done_paths": [
            row["done_path"]
            for row in received_jobs
            if row["source_role"] == "reference"
        ],
        "source_isolation_validated": True,
        "root_pairing_validated": True,
        "sealed_root_set_validated": True,
        "merge_executed": False,
        "performance_lock_finalized": False,
        "quality_pilot_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
    }
    receipt_path = receive_dir / "receive_receipt.json"
    _write(receipt_path, receipt)

    def validate_receive(**kwargs: Any) -> dict[str, Any]:
        assert Path(kwargs["receive_dir"]).resolve() == receive_dir.resolve()
        assert Path(kwargs["run_dir"]).resolve() == run_dir.resolve()
        assert kwargs["project"] == "test-project"
        assert kwargs["bucket"] == "test-bucket"
        producer_calls["receive"] += 1
        return _read(receipt_path)

    monkeypatch.setattr(lock_spot, "validate_received_directory", validate_receive)
    profiles = tuple(M31_T3_BEHAVIOR_PROFILES)
    paired = [
        {
            "hand_index": index,
            "profile": profiles[index % len(profiles)],
            "root_artifact_sha256": root_hashes[index],
            "root_file_sha256": root_hashes[index],
            "paired_seat_parity_count": 2,
        }
        for index in runner.CONTRACT_HAND_INDICES
    ]
    generic = {
        "schema": base.MERGE_SCHEMA,
        "scope": base.FULL_PERFORMANCE_SCOPE,
        "run_contract": plan["run_contract"],
        "run_contract_digest": plan["run_contract_digest"],
        "hand_indices": list(runner.CONTRACT_HAND_INDICES),
        "paired_hand_count": 100,
        "root_count": 200,
        "budget": dict(plan["run_contract"]["budget"]),
        "allocation": dict(runner.ALLOCATION),
        "source_done_inputs": source_done_inputs,
        "paired_artifacts": paired,
        "integrity": {
            "candidate_hand_count": 100,
            "reference_hand_count": 100,
            "paired_hand_count": 100,
            "paired_root_count": 200,
            "unique_observation_fingerprint_count": 200,
            "paired_hand_parity_count": 100,
            "paired_root_parity_count": 200,
            "missing_hand_indices": [],
            "duplicate_hand_indices": [],
            "out_of_contract_hand_indices": [],
        },
        "performance": _performance(),
    }
    return _Fixture(
        inputs=inputs,
        receive_inputs=receive_inputs,
        plan=plan,
        generic=generic,
        claim_path=claim_path,
        materialization_path=materialization_path,
        seal_path=seal_path,
        done_paths=done_paths,
        producer_calls=producer_calls,
    )


def _patch_generic_merge(
    monkeypatch: pytest.MonkeyPatch,
    fixture: _Fixture,
    calls: list[dict[str, Any]],
) -> None:
    def merge(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        assert kwargs["scope"] == base.FULL_PERFORMANCE_SCOPE
        assert kwargs["contract_variant"] == runner.CANDIDATE02_PERFORMANCE_LOCK_VARIANT
        return deepcopy(fixture.generic)

    monkeypatch.setattr(base, "merge_performance_v2", merge)


def test_lock_merge_passes_only_fresh_quality_pilot(
    lock_fixture: _Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    _patch_generic_merge(monkeypatch, lock_fixture, calls)

    value = subject.merge_candidate02_performance_lock(
        candidate_done_paths=lock_fixture.done_paths["candidate"],
        reference_done_paths=lock_fixture.done_paths["reference"],
        lock_inputs=lock_fixture.inputs,
        receive_inputs=lock_fixture.receive_inputs,
    )

    assert len(calls) == 1
    assert lock_fixture.producer_calls == {"claim": 1, "seal": 1, "receive": 1}
    assert value["status"] == "pass"
    assert value["decision"] == subject.PASS_DECISION
    assert value["hard_qualification_passed"] is True
    assert value["quality_launch_guards_passed"] is True
    assert value["performance_lock_qualified"] is True
    assert value["quality_pilot_authorized"] is True
    assert value["candidate_finalized_no_go"] is False
    assert value["root_lineage"]["candidate_reference_root_pairing_exact"] is True
    assert value["root_lineage"]["root_artifact_count"] == 100
    assert value["root_count"] == 200
    assert value["plan_partitions"]["candidate_shard_count"] == 10
    assert value["plan_partitions"]["reference_shard_count"] == 10
    for field in (
        "rerun_authorized",
        "reseed_authorized",
        "artifact_fanout_authorized",
        "training_eligible",
        "training_authorized",
        "promotion_evidence",
        "promotion_authorized",
        "current_profile_changed",
        "named_profile_added",
        "runtime_policy_activated",
        "m31_complete",
    ):
        assert value[field] is False


@pytest.mark.parametrize(
    ("performance", "hard", "guards"),
    [
        (_performance(), True, True),
        (
            _performance(
                first_p95=160.0,
                first_p99=220.0,
                first_max=230.0,
                second_p95=5.5,
                peak_rss=900_000_000,
            ),
            True,
            False,
        ),
        (
            _performance(
                first_p95=180.0001,
                first_p99=240.0,
                first_max=240.0,
                second_p95=6.0,
                peak_rss=1_073_741_824,
            ),
            False,
            False,
        ),
    ],
)
def test_gate_groups_are_inclusive_and_separate(
    lock_fixture: _Fixture,
    performance: dict[str, Any],
    hard: bool,
    guards: bool,
) -> None:
    generic = deepcopy(lock_fixture.generic)
    generic["performance"] = performance

    _hard_gates, hard_passed, _guard_gates, guards_passed = subject._gate_groups(
        generic
    )

    assert hard_passed is hard
    assert guards_passed is guards


def test_hard_pass_guard_failure_finalizes_candidate_no_go(
    lock_fixture: _Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_fixture.generic["performance"] = _performance(
        first_p95=160.0,
        first_p99=220.0,
        first_max=230.0,
        second_p95=5.5,
        peak_rss=900_000_000,
    )
    _patch_generic_merge(monkeypatch, lock_fixture, [])

    value = subject.merge_candidate02_performance_lock(
        candidate_done_paths=lock_fixture.done_paths["candidate"],
        reference_done_paths=lock_fixture.done_paths["reference"],
        lock_inputs=lock_fixture.inputs,
        receive_inputs=lock_fixture.receive_inputs,
    )

    assert value["hard_qualification_passed"] is True
    assert value["quality_launch_guards_passed"] is False
    assert value["status"] == "no_go"
    assert value["decision"] == subject.NO_GO_DECISION
    assert value["candidate_finalized_no_go"] is True
    assert value["quality_pilot_authorized"] is False
    assert value["rerun_authorized"] is False
    assert value["reseed_authorized"] is False


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), True, "150"])
def test_gate_groups_reject_nonfinite_and_wrong_type_metrics(
    lock_fixture: _Fixture, bad_value: Any
) -> None:
    generic = deepcopy(lock_fixture.generic)
    generic["performance"]["candidate_by_seat"]["first"]["p95_seconds"] = bad_value
    with pytest.raises(ValueError):
        subject._gate_groups(generic)


def test_partition_and_root_order_tamper_fail_closed(
    lock_fixture: _Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    partition = deepcopy(lock_fixture.generic)
    partition["source_done_inputs"]["candidate"][0]["shard_manifest_digest"] = _sha(
        2000
    )
    lock_fixture.generic = partition
    _patch_generic_merge(monkeypatch, lock_fixture, [])
    with pytest.raises(ValueError, match="candidate shard partition"):
        subject.merge_candidate02_performance_lock(
            candidate_done_paths=lock_fixture.done_paths["candidate"],
            reference_done_paths=lock_fixture.done_paths["reference"],
            lock_inputs=lock_fixture.inputs,
            receive_inputs=lock_fixture.receive_inputs,
        )

    root_order = deepcopy(partition)
    root_order["source_done_inputs"]["candidate"][0]["shard_manifest_digest"] = (
        lock_fixture.plan["jobs"][0]["shard_manifest_sha256"]
    )
    root_order["paired_artifacts"][0], root_order["paired_artifacts"][1] = (
        root_order["paired_artifacts"][1],
        root_order["paired_artifacts"][0],
    )
    lock_fixture.generic = root_order
    with pytest.raises(ValueError, match="generic/root pairing"):
        subject.merge_candidate02_performance_lock(
            candidate_done_paths=lock_fixture.done_paths["candidate"],
            reference_done_paths=lock_fixture.done_paths["reference"],
            lock_inputs=lock_fixture.inputs,
            receive_inputs=lock_fixture.receive_inputs,
        )


def test_claim_and_seal_lineage_tamper_fail_closed(
    lock_fixture: _Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_generic_merge(monkeypatch, lock_fixture, [])
    claim = _read(lock_fixture.claim_path)
    claim["restrictions"]["training_authorized"] = True
    _write(lock_fixture.claim_path, claim)
    with pytest.raises(ValueError, match="claim/plan lineage"):
        subject.merge_candidate02_performance_lock(
            candidate_done_paths=lock_fixture.done_paths["candidate"],
            reference_done_paths=lock_fixture.done_paths["reference"],
            lock_inputs=lock_fixture.inputs,
            receive_inputs=lock_fixture.receive_inputs,
        )

    claim["restrictions"]["training_authorized"] = False
    _write(lock_fixture.claim_path, claim)
    materialization = _read(lock_fixture.materialization_path)
    materialization["global_claim_sha256"] = lock_open.canonical_sha256(claim)
    _write(lock_fixture.materialization_path, materialization)
    seal = _read(lock_fixture.seal_path)
    seal["global_claim_sha256"] = lock_open.canonical_sha256(claim)
    seal["materialization_sha256"] = lock_open.canonical_sha256(materialization)
    seal["selection_inputs"]["timing_used"] = True
    _write(lock_fixture.seal_path, seal)
    with pytest.raises(ValueError, match="materialization/root-seal lineage"):
        subject.merge_candidate02_performance_lock(
            candidate_done_paths=lock_fixture.done_paths["candidate"],
            reference_done_paths=lock_fixture.done_paths["reference"],
            lock_inputs=lock_fixture.inputs,
            receive_inputs=lock_fixture.receive_inputs,
        )


def test_write_once_validation_replays_sources_and_lineage(
    lock_fixture: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []
    _patch_generic_merge(monkeypatch, lock_fixture, calls)
    summary_path = tmp_path / "merge" / "summary.json"
    validation_path = tmp_path / "merge" / "validation.json"

    summary, validation = subject.merge_and_validate_candidate02_performance_lock(
        candidate_done_paths=lock_fixture.done_paths["candidate"],
        reference_done_paths=lock_fixture.done_paths["reference"],
        lock_inputs=lock_fixture.inputs,
        receive_inputs=lock_fixture.receive_inputs,
        summary_output_path=summary_path,
        validation_output_path=validation_path,
    )

    assert len(calls) == 2
    assert lock_fixture.producer_calls == {"claim": 2, "seal": 2, "receive": 2}
    assert summary_path.read_bytes() == runner.canonical_bytes(summary)
    assert validation_path.read_bytes() == runner.canonical_bytes(validation)
    assert validation["source_artifacts_replayed"] is True
    assert validation["open_claim_replayed"] is True
    assert validation["root_seal_replayed"] is True
    assert validation["root_set_recomputed"] is True
    assert validation["quality_pilot_authorized"] is True
    with pytest.raises(FileExistsError, match="write-once"):
        subject.merge_and_validate_candidate02_performance_lock(
            candidate_done_paths=lock_fixture.done_paths["candidate"],
            reference_done_paths=lock_fixture.done_paths["reference"],
            lock_inputs=lock_fixture.inputs,
            receive_inputs=lock_fixture.receive_inputs,
            summary_output_path=summary_path,
            validation_output_path=validation_path,
        )


def test_validation_rejects_done_and_summary_tamper(
    lock_fixture: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_generic_merge(monkeypatch, lock_fixture, [])
    summary = subject.merge_candidate02_performance_lock(
        candidate_done_paths=lock_fixture.done_paths["candidate"],
        reference_done_paths=lock_fixture.done_paths["reference"],
        lock_inputs=lock_fixture.inputs,
        receive_inputs=lock_fixture.receive_inputs,
    )
    summary_path = tmp_path / "summary.json"
    _write(summary_path, summary)
    changed_done = lock_fixture.done_paths["candidate"][0]
    changed_done.write_bytes(changed_done.read_bytes() + b" ")
    with pytest.raises(ValueError, match="DONE input hash mismatch"):
        subject.validate_candidate02_performance_lock_merge(summary_path=summary_path)

    changed_done.write_bytes(runner.canonical_bytes({"job_id": "candidate-shard-00"}))
    tampered = deepcopy(summary)
    tampered["quality_pilot_authorized"] = False
    _write(summary_path, tampered)
    with pytest.raises(ValueError, match="aggregate changed"):
        subject.validate_candidate02_performance_lock_merge(summary_path=summary_path)


def test_validation_failure_leaves_consumed_summary_and_no_retry(
    lock_fixture: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_generic_merge(monkeypatch, lock_fixture, [])
    summary_path = tmp_path / "failed" / "summary.json"
    validation_path = tmp_path / "failed" / "validation.json"

    def fail_validation(**_kwargs: Any) -> dict[str, Any]:
        raise ValueError("independent replay failed")

    monkeypatch.setattr(
        subject, "validate_candidate02_performance_lock_merge", fail_validation
    )
    with pytest.raises(ValueError, match="independent replay failed"):
        subject.merge_and_validate_candidate02_performance_lock(
            candidate_done_paths=lock_fixture.done_paths["candidate"],
            reference_done_paths=lock_fixture.done_paths["reference"],
            lock_inputs=lock_fixture.inputs,
            receive_inputs=lock_fixture.receive_inputs,
            summary_output_path=summary_path,
            validation_output_path=validation_path,
        )
    assert summary_path.is_file()
    assert not validation_path.exists()
    with pytest.raises(FileExistsError, match="write-once"):
        subject.merge_and_validate_candidate02_performance_lock(
            candidate_done_paths=lock_fixture.done_paths["candidate"],
            reference_done_paths=lock_fixture.done_paths["reference"],
            lock_inputs=lock_fixture.inputs,
            receive_inputs=lock_fixture.receive_inputs,
            summary_output_path=summary_path,
            validation_output_path=validation_path,
        )


def test_summary_and_validation_paths_must_be_distinct(
    lock_fixture: _Fixture, tmp_path: Path
) -> None:
    path = tmp_path / "same.json"
    with pytest.raises(ValueError, match="must be distinct"):
        subject.merge_and_validate_candidate02_performance_lock(
            candidate_done_paths=lock_fixture.done_paths["candidate"],
            reference_done_paths=lock_fixture.done_paths["reference"],
            lock_inputs=lock_fixture.inputs,
            receive_inputs=lock_fixture.receive_inputs,
            summary_output_path=path,
            validation_output_path=path,
        )


def test_arbitrary_done_set_cannot_gain_quality_authorization(
    lock_fixture: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_generic_merge(monkeypatch, lock_fixture, [])
    replacement = tmp_path / "unclaimed-rerun" / "DONE.json"
    _write(replacement, {"job_id": "candidate-shard-00"})
    supplied = list(lock_fixture.done_paths["candidate"])
    supplied[0] = replacement

    with pytest.raises(ValueError, match="not the claimed one-shot receive set"):
        subject.merge_candidate02_performance_lock(
            candidate_done_paths=supplied,
            reference_done_paths=lock_fixture.done_paths["reference"],
            lock_inputs=lock_fixture.inputs,
            receive_inputs=lock_fixture.receive_inputs,
        )


@pytest.mark.parametrize("forbidden_kind", ["source", "receive", "run", "roots"])
def test_outputs_cannot_mutate_scientific_input_trees(
    lock_fixture: _Fixture,
    tmp_path: Path,
    forbidden_kind: str,
) -> None:
    parents = {
        "source": lock_fixture.done_paths["candidate"][0].parent,
        "receive": Path(lock_fixture.receive_inputs.receive_dir),
        "run": Path(lock_fixture.receive_inputs.run_dir),
        "roots": Path(lock_fixture.inputs.lock_output_directory),
    }
    summary = parents[forbidden_kind] / "scientific-summary.json"
    validation = tmp_path / "safe-validation.json"
    with pytest.raises(ValueError, match="outside source"):
        subject.merge_and_validate_candidate02_performance_lock(
            candidate_done_paths=lock_fixture.done_paths["candidate"],
            reference_done_paths=lock_fixture.done_paths["reference"],
            lock_inputs=lock_fixture.inputs,
            receive_inputs=lock_fixture.receive_inputs,
            summary_output_path=summary,
            validation_output_path=validation,
        )
    assert not summary.exists()
    assert not validation.exists()


@pytest.mark.parametrize("forbidden_kind", ["source", "receive", "run", "roots"])
def test_public_validator_cannot_write_into_scientific_input_trees(
    lock_fixture: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    forbidden_kind: str,
) -> None:
    _patch_generic_merge(monkeypatch, lock_fixture, [])
    summary = subject.merge_candidate02_performance_lock(
        candidate_done_paths=lock_fixture.done_paths["candidate"],
        reference_done_paths=lock_fixture.done_paths["reference"],
        lock_inputs=lock_fixture.inputs,
        receive_inputs=lock_fixture.receive_inputs,
    )
    summary_path = tmp_path / f"safe-summary-{forbidden_kind}.json"
    _write(summary_path, summary)
    parents = {
        "source": lock_fixture.done_paths["candidate"][0].parent,
        "receive": Path(lock_fixture.receive_inputs.receive_dir),
        "run": Path(lock_fixture.receive_inputs.run_dir),
        "roots": Path(lock_fixture.inputs.lock_output_directory),
    }
    validation_path = parents[forbidden_kind] / "validation.json"
    with pytest.raises(ValueError, match="outside source"):
        subject.validate_candidate02_performance_lock_merge(
            summary_path=summary_path,
            output_path=validation_path,
        )
    assert not validation_path.exists()


def test_public_validator_rejects_summary_as_output(
    lock_fixture: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_generic_merge(monkeypatch, lock_fixture, [])
    summary = subject.merge_candidate02_performance_lock(
        candidate_done_paths=lock_fixture.done_paths["candidate"],
        reference_done_paths=lock_fixture.done_paths["reference"],
        lock_inputs=lock_fixture.inputs,
        receive_inputs=lock_fixture.receive_inputs,
    )
    summary_path = tmp_path / "summary-is-not-validation.json"
    _write(summary_path, summary)
    with pytest.raises(ValueError, match="must be distinct"):
        subject.validate_candidate02_performance_lock_merge(
            summary_path=summary_path,
            output_path=summary_path,
        )


@pytest.mark.parametrize(("status", "exit_code"), [("pass", 0), ("no_go", 1)])
def test_cli_exit_code_distinguishes_pass_and_no_go(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    exit_code: int,
) -> None:
    monkeypatch.setattr(
        subject,
        "merge_and_validate_candidate02_performance_lock",
        lambda **_kwargs: ({"status": status}, {}),
    )
    argv = [
        "--candidate-done",
        "candidate/DONE.json",
        "--reference-done",
        "reference/DONE.json",
        "--summary-output",
        "summary.json",
        "--validation-output",
        "validation.json",
        "--run-dir",
        "run",
        "--receive-dir",
        "receive",
        "--lock-output",
        "roots",
        "--candidate-library",
        "candidate.so",
        "--reference-library",
        "reference.so",
        "--feature-encoder",
        "feature.py",
        "--startup-source",
        "startup.sh",
    ]
    assert subject.main(argv) == exit_code
