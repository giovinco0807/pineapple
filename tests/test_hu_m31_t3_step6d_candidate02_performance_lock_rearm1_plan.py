from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_candidate02_performance_lock_plan as v1_plan,
)
from ofc_regular import (
    hu_m31_t3_step6d_candidate02_performance_lock_rearm1_plan as subject,
)
from ofc_regular import run_hu_m31_t3_step6d_performance_v2 as runner
from ofc_regular.hu_m31_t3_behavior_roots import M31_T3_BEHAVIOR_PROFILES


def test_real_closeout_builds_frozen_fresh_rearm1_plan() -> None:
    plan = subject.build_precontent_plan()

    assert subject.canonical_sha256(plan) == subject.PRECONTENT_PLAN_SHA256
    assert plan["schema"] == subject.PLAN_SCHEMA
    assert plan["status"] == subject.PLAN_STATUS
    assert plan["scope"] == "performance_lock_rearm1_fresh_roots_only"
    assert plan["candidate_variant"] == (
        runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_VARIANT
    )
    assert plan["run_contract_digest"] == subject.RECOVERY_RUN_CONTRACT_DIGEST
    assert runner.contract_variant(plan["run_contract"]) == subject.CANDIDATE_VARIANT
    assert plan["root_contract"]["schema"] == subject.ROOT_SCHEMA
    assert plan["root_contract"]["seed_contract"] == (
        runner.candidate02_performance_lock_recovery_seed_contract()
    )
    assert plan["root_contract"]["fresh_root_set"] is True
    assert plan["root_contract"]["root_content_addressed"] is False
    assert plan["open_claim_required_before_root_content"] is True
    assert plan["root_content_opened"] is False
    assert plan["cloud_started"] is False
    assert plan["logical_job_count"] == 20
    assert plan["training_eligible"] is False
    assert plan["quality_evidence"] is False
    assert plan["current_profile_changed"] is False


def test_receipt_and_terminal_old_run_disposition_are_content_bound() -> None:
    plan = subject.build_precontent_plan()
    closeout = plan["startup_failure_closeout"]
    receipt = closeout["receipt"]

    assert closeout["sha256"] == subject.STARTUP_FAILURE_RECEIPT_SHA256
    assert subject.canonical_sha256(receipt) == subject.STARTUP_FAILURE_RECEIPT_SHA256
    assert receipt["incident"]["failed_jobs"] == 20
    assert receipt["control_plane_counts"] == {
        "live": 0,
        "done": 0,
        "heartbeat": 0,
        "result": 0,
        "resume": 0,
        "attempt1": 0,
    }
    assert receipt["content_access"]["hand_content_opened"] is False
    assert receipt["content_access"]["root_content_opened"] is False
    assert receipt["content_access"]["result_content_opened"] is False

    old = plan["old_run_disposition"]
    assert old["attempt0_consumed"] is True
    assert old["current_run_irrecoverable"] is True
    assert old["old_attempt1_authorized"] is False
    assert old["old_root_reuse_authorized"] is False
    assert old["old_result_reuse_authorized"] is False
    assert old["old_seed_reuse_authorized"] is False
    assert old["old_package_reuse_authorized"] is False
    assert old["old_claim_reuse_authorized"] is False
    assert old["old_cloud_execution_authorized"] is False


def test_only_one_fresh_seed_lock_is_planned_and_not_cloud_authorized() -> None:
    plan = subject.build_precontent_plan()
    authority = plan["fresh_lock_authority"]
    seed_contract = plan["root_contract"]["seed_contract"]

    assert authority["authorized_fresh_lock_count"] == 1
    assert authority["fresh_lock_plan_authorized"] is True
    assert authority["fresh_root_set_required"] is True
    assert authority["fresh_seed_set_required"] is True
    assert authority["fresh_global_claim_required_before_root_content"] is True
    assert authority["fresh_root_content_authorized_before_global_claim"] is False
    assert authority["fresh_spot_execution_authorized"] is False
    assert authority["old_attempt1_authorized"] is False
    assert authority["old_roots_authorized"] is False
    assert authority["old_seeds_authorized"] is False
    assert authority["second_fresh_lock_authorized"] is False
    assert authority["quality_pilot_authorized"] is False
    assert authority["training_eligible"] is False
    assert authority["current_profile_changed"] is False

    assert seed_contract["performance_lock_v1_overlap_count"] == 0
    assert seed_contract["candidate02_development_overlap_count"] == 0
    assert seed_contract["existing_step6d_union_overlap_count"] == 0
    assert seed_contract["all_values_unique"] is True
    assert seed_contract["seed_set_sha256"] == (
        runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SEED_SET_SHA256
    )


def test_arithmetic_twenty_job_mapping_is_balanced_and_role_identical() -> None:
    plan = subject.build_precontent_plan()
    shards = plan["shards"]
    covered = [hand for shard in shards for hand in shard["work_hand_indices"]]

    assert covered == list(range(100))
    assert all(
        shard["work_hand_indices"]
        == list(range(shard["shard_index"] * 10, shard["shard_index"] * 10 + 10))
        for shard in shards
    )
    assert all(
        shard["profile_counts"] == {profile: 2 for profile in M31_T3_BEHAVIOR_PROFILES}
        for shard in shards
    )
    candidate = [
        job["work_hand_indices"]
        for job in plan["jobs"]
        if job["source_role"] == "candidate"
    ]
    reference = [
        job["work_hand_indices"]
        for job in plan["jobs"]
        if job["source_role"] == "reference"
    ]
    assert candidate == reference == [
        shard["work_hand_indices"] for shard in plan["shards"]
    ]
    assert all(
        runner.build_shard_manifest(
            run_contract=plan["run_contract"],
            source_role=job["source_role"],
            work_hand_indices=job["work_hand_indices"],
        )["schema"]
        == subject.SHARD_MANIFEST_SCHEMA
        for job in plan["jobs"]
    )


def test_rearm1_keeps_v1_binary_image_and_allocation_identity() -> None:
    old = v1_plan.build_precontent_plan()
    new = subject.build_precontent_plan()

    assert new["development_qualification"] == old["development_qualification"]
    assert new["source_identity"] == old["source_identity"]
    assert new["image"] == old["image"]
    assert new["allocation"] == old["allocation"]
    assert new["run_contract"]["candidate_library_sha256"] == (
        old["run_contract"]["candidate_library_sha256"]
    )
    assert new["run_contract"]["reference_library_sha256"] == (
        old["run_contract"]["reference_library_sha256"]
    )


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("startup_failure_closeout", "receipt", "disposition", "attempt1_authorized"), True),
        (("old_run_disposition", "old_root_reuse_authorized"), True),
        (("fresh_lock_authority", "authorized_fresh_lock_count"), 2),
        (("fresh_lock_authority", "fresh_spot_execution_authorized"), True),
        (("fresh_lock_authority", "training_eligible"), True),
        (("root_contract", "seed_contract", "performance_lock_v1_overlap_count"), 1),
        (("jobs", 0, "work_hand_indices"), list(range(10, 20))),
        (("current_profile_changed",), True),
    ),
)
def test_receipt_mapping_seed_and_authorization_tamper_fail_closed(
    path: tuple[object, ...], value: object
) -> None:
    mutated = deepcopy(subject.build_precontent_plan())
    target: object = mutated
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]

    with pytest.raises(ValueError):
        subject.validate_precontent_plan(mutated)


def test_incident_receipt_file_tamper_fails_before_plan_build(tmp_path: Path) -> None:
    tampered = tmp_path / "closeout.json"
    tampered.write_bytes(subject.DEFAULT_INCIDENT_RECEIPT_PATH.read_bytes() + b" ")

    with pytest.raises(ValueError, match="file hash changed"):
        subject.build_precontent_plan(incident_receipt_path=tampered)


def test_freeze_candidate_is_available_but_unfrozen_build_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = subject.PRECONTENT_PLAN_SHA256
    assert subject.compute_precontent_plan_sha256() == frozen

    monkeypatch.setattr(
        subject,
        "PRECONTENT_PLAN_SHA256",
        subject.PRECONTENT_PLAN_SHA256_PLACEHOLDER,
    )
    assert subject.compute_precontent_plan_sha256() == frozen
    with pytest.raises(RuntimeError, match="not frozen"):
        subject.build_precontent_plan()


def test_plan_write_and_load_are_canonical_exclusive_and_hash_checked(
    tmp_path: Path,
) -> None:
    target = tmp_path / "rearm1-plan.json"
    plan = subject.write_precontent_plan(output_path=target)

    assert target.read_bytes() == subject.canonical_bytes(plan)
    assert subject.load_and_validate_precontent_plan(target) == plan
    with pytest.raises(FileExistsError, match="already exists"):
        subject.write_precontent_plan(output_path=target)

    changed = tmp_path / "changed.json"
    changed.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match="file hash changed"):
        subject.load_and_validate_precontent_plan(changed)
