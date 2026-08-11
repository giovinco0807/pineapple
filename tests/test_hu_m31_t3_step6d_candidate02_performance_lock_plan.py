from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_candidate02_performance_lock_plan as subject,
)
from ofc_regular import run_hu_m31_t3_step6d_performance_v2 as runner
from ofc_regular.hu_m31_t3_behavior_roots import M31_T3_BEHAVIOR_PROFILES


def test_real_development_go_builds_precontent_arithmetic_plan() -> None:
    plan = subject.build_precontent_plan()

    assert plan["schema"] == subject.PLAN_SCHEMA
    assert plan["scope"] == "performance_lock"
    assert plan["candidate_variant"] == runner.CANDIDATE02_PERFORMANCE_LOCK_VARIANT
    assert plan["run_contract_digest"] == subject.LOCK_RUN_CONTRACT_DIGEST
    assert plan["root_contract"]["seed_contract"] == (
        runner.candidate02_performance_lock_seed_contract()
    )
    assert plan["root_contract"]["root_content_addressed"] is False
    assert plan["root_content_opened"] is False
    assert plan["open_claim_required_before_root_content"] is True
    assert plan["logical_job_count"] == 20
    assert plan["current_profile_changed"] is False
    assert plan["training_eligible"] is False

    covered = [hand for shard in plan["shards"] for hand in shard["work_hand_indices"]]
    assert covered == list(range(100))
    assert all(
        shard["work_hand_indices"]
        == list(range(shard["shard_index"] * 10, shard["shard_index"] * 10 + 10))
        for shard in plan["shards"]
    )
    assert all(
        shard["profile_counts"] == {profile: 2 for profile in M31_T3_BEHAVIOR_PROFILES}
        for shard in plan["shards"]
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
    assert (
        candidate
        == reference
        == [shard["work_hand_indices"] for shard in plan["shards"]]
    )


def test_plan_build_does_not_touch_any_lock_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_open = Path.open
    original_read_bytes = Path.read_bytes
    original_stat = Path.stat

    def reject_lock_path(path: Path) -> None:
        lowered = str(path).lower()
        if "performance_lock" in lowered and "plan" not in lowered:
            raise AssertionError(f"precontent plan touched lock content: {path}")
        if "performance-lock" in lowered and "plan" not in lowered:
            raise AssertionError(f"precontent plan touched lock content: {path}")

    def guarded_open(path: Path, *args, **kwargs):
        reject_lock_path(path)
        return original_open(path, *args, **kwargs)

    def guarded_read_bytes(path: Path):
        reject_lock_path(path)
        return original_read_bytes(path)

    def guarded_stat(path: Path, *args, **kwargs):
        reject_lock_path(path)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    monkeypatch.setattr(Path, "stat", guarded_stat)

    plan = subject.build_precontent_plan()
    assert plan["assignment"]["root_content_used"] is False
    assert plan["assignment"]["topology_used"] is False
    assert plan["assignment"]["timing_used"] is False
    assert plan["assignment"]["q_values_used"] is False
    assert plan["assignment"]["ev_used"] is False


def test_plan_rejects_seed_assignment_and_authorization_tamper() -> None:
    base = subject.build_precontent_plan()
    mutations = []

    seed = deepcopy(base)
    seed["root_contract"]["seed_contract"]["namespace_bases"]["hand"] += 1
    mutations.append(seed)

    mapping = deepcopy(base)
    mapping["jobs"][0]["work_hand_indices"] = list(
        mapping["jobs"][1]["work_hand_indices"]
    )
    mutations.append(mapping)

    timing = deepcopy(base)
    timing["assignment"]["timing_used"] = True
    mutations.append(timing)

    opened = deepcopy(base)
    opened["root_content_opened"] = True
    mutations.append(opened)

    current = deepcopy(base)
    current["current_profile_changed"] = True
    mutations.append(current)

    for value in mutations:
        with pytest.raises(ValueError):
            subject.validate_precontent_plan(value)


def test_official_development_hash_and_current_hash_are_required(
    tmp_path: Path,
) -> None:
    summary = subject.DEFAULT_DEVELOPMENT_DIR / "summary.json"
    validation = subject.DEFAULT_DEVELOPMENT_DIR / "validation.json"
    bad_summary = tmp_path / "summary.json"
    bad_summary.write_bytes(summary.read_bytes() + b" ")
    with pytest.raises(ValueError, match="evidence hash changed"):
        subject.load_development_qualification(
            summary_path=bad_summary,
            validation_path=validation,
        )

    bad_registry = tmp_path / "ai_profiles.py"
    bad_registry.write_bytes(subject.DEFAULT_CURRENT_REGISTRY_PATH.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="current profile registry changed"):
        subject.build_precontent_plan(current_registry_path=bad_registry)


def test_plan_write_is_exclusive_and_canonical(tmp_path: Path) -> None:
    target = tmp_path / "plan.json"
    first = subject.write_precontent_plan(output_path=target)
    assert target.read_bytes() == subject.canonical_bytes(first)
    with pytest.raises(FileExistsError, match="already exists"):
        subject.write_precontent_plan(output_path=target)
