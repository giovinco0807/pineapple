from __future__ import annotations

import copy
from pathlib import Path

import pytest

from ofc_regular import hu_m31_t3_step6d_performance_lock_v4_gate as gate
from ofc_regular import run_hu_m31_t3_step6d_performance_v2 as runner
from ofc_regular import (
    merge_hu_m31_t3_step6d_candidate02_performance_lock_v4 as subject,
)


def _contract() -> dict:
    return runner.build_run_contract(
        candidate_library_sha256=(
            runner.CANDIDATE02_PERFORMANCE_LOCK_ACCEPTED_CANDIDATE_LIBRARY_SHA256
        ),
        reference_library_sha256=(
            runner.CANDIDATE02_PERFORMANCE_LOCK_ACCEPTED_REFERENCE_LIBRARY_SHA256
        ),
        variant=runner.CANDIDATE02_PERFORMANCE_LOCK_V4_VARIANT,
    )


def _jobs(contract: dict) -> list[dict]:
    result = []
    for role in runner.SOURCE_ROLES:
        for shard in range(10):
            work = list(range(10 * shard, 10 * shard + 10))
            manifest = runner.build_shard_manifest(
                run_contract=contract,
                source_role=role,
                work_hand_indices=work,
            )
            result.append(
                {
                    "job_id": f"{role}-shard-{shard:02d}",
                    "source_role": role,
                    "shard_index": shard,
                    "work_hand_indices": work,
                    "shard_manifest_sha256": runner.canonical_sha256(manifest),
                }
            )
    return result


def _lineage(monkeypatch: pytest.MonkeyPatch) -> tuple[dict, dict, dict]:
    contract = _contract()
    plan = {
        "run_contract": contract,
        "run_contract_digest": runner.canonical_sha256(contract),
        "run009_scientific_gate": {"receipt_sha256": "1" * 64},
        "jobs": _jobs(contract),
    }
    plan_sha = subject.canonical_sha256(plan)
    root_hashes = [f"{index + 1:064x}" for index in range(100)]
    audit = {
        "root_file_count": 100,
        "paired_hand_count": 100,
        "observation_count": 200,
        "root_hashes": root_hashes,
        "root_hash_aggregate_sha256": subject.canonical_sha256(root_hashes),
        "profile_counts": {},
        "seat_counts": {"first": 100, "second": 100},
        "seed_set_sha256": runner.CANDIDATE02_PERFORMANCE_LOCK_V4_SEED_SET_SHA256,
        "seed_overlap_counts": {"prior": 0},
        "observation_fingerprint_count": 200,
        "observation_fingerprint_aggregate_sha256": "2" * 64,
        "all_root_artifacts_valid": True,
        "current_profile_unchanged": True,
        "old_root_or_seed_reuse": False,
    }
    materialization = {
        "plan_sha256": plan_sha,
        "run_contract_digest": runner.canonical_sha256(contract),
        "claim_sha256": "3" * 64,
        "root_output_directory": "C:/immutable/v4-roots",
        **audit,
    }
    seal = {
        **copy.deepcopy(materialization),
        "materialization_receipt_sha256": subject.canonical_sha256(materialization),
    }
    monkeypatch.setattr(subject.lock_plan, "PLAN_SHA256", plan_sha)
    monkeypatch.setattr(
        subject.lock_plan,
        "RUN_CONTRACT_DIGEST",
        runner.canonical_sha256(contract),
    )
    monkeypatch.setattr(
        subject.lock_plan,
        "validate_performance_lock_v4_plan",
        lambda value: copy.deepcopy(dict(value)),
    )
    monkeypatch.setattr(
        subject.lock_plan,
        "validate_materialization_receipt",
        lambda value: copy.deepcopy(dict(value)),
    )
    monkeypatch.setattr(
        subject.lock_plan,
        "validate_root_seal",
        lambda value: copy.deepcopy(dict(value)),
    )
    return plan, materialization, seal


def _generic(plan: dict, seal: dict) -> dict:
    source_inputs = {role: [] for role in runner.SOURCE_ROLES}
    for job in plan["jobs"]:
        role = job["source_role"]
        source_inputs[role].append(
            {
                "path": f"C:/received/{role}/{job['job_id']}/DONE.json",
                "sha256": "4" * 64,
                "shard_manifest_digest": job["shard_manifest_sha256"],
                "work_hand_indices": list(job["work_hand_indices"]),
            }
        )
    paired = []
    fingerprints = []
    for hand, root_hash in enumerate(seal["root_hashes"]):
        rows = []
        for offset, seat in enumerate(("first", "second")):
            fingerprint = f"{10_000 + 2 * hand + offset:064x}"
            fingerprints.append(fingerprint)
            rows.append(
                {
                    "root_index": 2 * hand + offset,
                    "seat": seat,
                    "observation_fingerprint": fingerprint,
                }
            )
        paired.append(
            {
                "hand_index": hand,
                "root_file_sha256": root_hash,
                "root_artifact_sha256": root_hash,
                "rows": rows,
            }
        )
    seal["observation_fingerprint_aggregate_sha256"] = subject.canonical_sha256(
        sorted(fingerprints)
    )
    return {
        "run_contract": plan["run_contract"],
        "run_contract_digest": plan["run_contract_digest"],
        "source_done_inputs": source_inputs,
        "paired_artifacts": paired,
    }


def _gate_value(generic: dict, passed: bool = True) -> dict:
    return {
        "all_gates_passed": passed,
        "quality_pilot_authorized": passed,
    }


def test_merge_binds_plan_seal_partitions_and_one_shot_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, materialization, seal = _lineage(monkeypatch)
    generic = _generic(plan, seal)
    materialization["observation_fingerprint_aggregate_sha256"] = seal[
        "observation_fingerprint_aggregate_sha256"
    ]
    seal["materialization_receipt_sha256"] = subject.canonical_sha256(
        materialization
    )
    monkeypatch.setattr(
        subject.performance_v2,
        "merge_performance_v2",
        lambda **_kwargs: copy.deepcopy(generic),
    )
    monkeypatch.setattr(
        subject.lock_gate,
        "build_performance_lock_v4_gate",
        lambda *_args, **_kwargs: _gate_value(generic),
    )
    monkeypatch.setattr(
        subject.lock_gate,
        "validate_performance_lock_v4_gate_value",
        lambda value, **_kwargs: copy.deepcopy(dict(value)),
    )

    summary = subject.merge_candidate02_performance_lock_v4(
        candidate_done_paths=[Path("candidate")],
        reference_done_paths=[Path("reference")],
        plan_value=plan,
        materialization_value=materialization,
        root_seal_value=seal,
    )
    assert summary["status"] == "pass"
    assert summary["scientific_performance_gate_passed"] is True
    assert summary["transport_lineage_required"] is True
    assert summary["transport_lineage_validated"] is False
    assert summary["performance_lock_finalized"] is False
    assert summary["one_shot_lock_consumed"] is False
    assert summary["quality_pilot_authorized"] is False
    assert summary["rerun_authorized"] is False
    assert summary["reseed_authorized"] is False
    assert summary["training_authorized"] is False
    assert summary["current_profile_changed"] is False


def test_lineage_rejects_materialization_or_seal_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, materialization, seal = _lineage(monkeypatch)
    seal["claim_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="lineage changed"):
        subject._validate_lineage(
            plan_value=plan,
            materialization_value=materialization,
            root_seal_value=seal,
        )


def test_partition_rejects_candidate_reference_path_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, _materialization, seal = _lineage(monkeypatch)
    generic = _generic(plan, seal)
    generic["source_done_inputs"]["reference"][0]["path"] = generic[
        "source_done_inputs"
    ]["candidate"][0]["path"]
    with pytest.raises(ValueError, match="source isolation"):
        subject._validate_partitions(plan, generic)


def test_root_lineage_rejects_seal_hash_or_fingerprint_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, materialization, seal = _lineage(monkeypatch)
    generic = _generic(plan, seal)
    materialization["observation_fingerprint_aggregate_sha256"] = seal[
        "observation_fingerprint_aggregate_sha256"
    ]
    generic["paired_artifacts"][0]["root_file_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="root hash changed"):
        subject._validate_root_lineage(
            generic=generic, materialization=materialization, seal=seal
        )


def test_stored_boundary_tamper_is_rejected_even_with_resealed_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, materialization, seal = _lineage(monkeypatch)
    generic = _generic(plan, seal)
    materialization["observation_fingerprint_aggregate_sha256"] = seal[
        "observation_fingerprint_aggregate_sha256"
    ]
    seal["materialization_receipt_sha256"] = subject.canonical_sha256(
        materialization
    )
    monkeypatch.setattr(
        subject.performance_v2,
        "merge_performance_v2",
        lambda **_kwargs: copy.deepcopy(generic),
    )
    monkeypatch.setattr(
        subject.lock_gate,
        "build_performance_lock_v4_gate",
        lambda *_args, **_kwargs: _gate_value(generic),
    )
    monkeypatch.setattr(
        subject.lock_gate,
        "validate_performance_lock_v4_gate_value",
        lambda value, **_kwargs: copy.deepcopy(dict(value)),
    )
    summary = subject.merge_candidate02_performance_lock_v4(
        candidate_done_paths=[Path("candidate")],
        reference_done_paths=[Path("reference")],
        plan_value=plan,
        materialization_value=materialization,
        root_seal_value=seal,
    )
    tampered = copy.deepcopy(summary)
    tampered["quality_pilot_authorized"] = True
    body = {key: value for key, value in tampered.items() if key != "summary_sha256"}
    tampered["summary_sha256"] = subject.canonical_sha256(body)
    with pytest.raises(ValueError, match="boundary changed"):
        subject.validate_candidate02_performance_lock_v4_value(
            tampered, replay_sources=False
        )
