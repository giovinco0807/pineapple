from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from ofc_regular import hu_m31_t3_step6d_candidate02_full100_plan as subject
from ofc_regular.hu_m31_t3_behavior_roots import M31_T3_BEHAVIOR_PROFILES


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = (
    REPO_ROOT / "outputs/hu_joint_policy/m31_t3_step6d/candidate02_development/"
    "full100_plan_v1.json"
)


def _stored_plan() -> dict:
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def test_real_tail_go_and_roots_build_exact_frozen_full100_plan() -> None:
    plan = subject.build_full100_plan()

    assert plan == _stored_plan()
    assert subject.canonical_sha256(plan) == subject.FULL100_PLAN_SHA256
    assert plan["tail_qualification"]["all_gates_passed"] is True
    assert plan["tail_qualification"]["full_performance_development_authorized"] is True
    assert plan["run_contract_digest"] == subject.FULL_RUN_CONTRACT_DIGEST
    assert plan["logical_job_count"] == 20
    assert plan["spot_package_authorized"] is False
    assert plan["cloud_started"] is False
    assert plan["current_profile_changed"] is False

    covered = [hand for shard in plan["shards"] for hand in shard["work_hand_indices"]]
    assert sorted(covered) == list(range(100))
    assert len(covered) == len(set(covered)) == 100
    assert all(len(shard["work_hand_indices"]) == 10 for shard in plan["shards"])
    assert all(
        shard["profile_counts"] == {profile: 2 for profile in M31_T3_BEHAVIOR_PROFILES}
        for shard in plan["shards"]
    )
    weights = [shard["topology_weight"] for shard in plan["shards"]]
    assert max(weights) - min(weights) == 36

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


def test_plan_rejects_authorization_mapping_and_provenance_tamper() -> None:
    base = _stored_plan()
    mutations = []

    authorized = deepcopy(base)
    authorized["spot_package_authorized"] = True
    mutations.append(authorized)

    mapping = deepcopy(base)
    mapping["jobs"][0]["work_hand_indices"] = list(
        mapping["jobs"][1]["work_hand_indices"]
    )
    mutations.append(mapping)

    timing = deepcopy(base)
    timing["assignment"]["timing_used"] = True
    mutations.append(timing)

    qualifier = deepcopy(base)
    qualifier["tail_qualification"]["all_gates_passed"] = False
    mutations.append(qualifier)

    for value in mutations:
        with pytest.raises(ValueError):
            subject.validate_full100_plan(value)


def test_tail_qualification_file_hash_is_required_before_full100(
    tmp_path: Path,
) -> None:
    summary = subject.DEFAULT_TAIL_MERGE_DIR / "summary.json"
    validation = subject.DEFAULT_TAIL_MERGE_DIR / "validation.json"
    tampered = tmp_path / "validation.json"
    tampered.write_bytes(validation.read_bytes() + b" ")

    with pytest.raises(ValueError, match="artifact hash changed"):
        subject.load_tail_qualification(
            summary_path=summary,
            validation_path=tampered,
        )
