from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from ofc_regular.hu_m31_t3_behavior_roots import M31_T3_BEHAVIOR_PROFILES
from ofc_regular.hu_m31_t3_step6c_contract import (
    ACCEPTED_FEATURE_ENCODER_SHA256,
    ACCEPTED_NATIVE_LIBRARY_SHA256,
    CONFIRMATION_BUDGET,
    CONFIRMATION_HAND_INDICES,
    CONFIRMATION_ROOT_INDICES,
    EXPECTED_STEP6C_CONTRACT_BYTE_SHA256,
    EXPECTED_STEP6C_CONTRACT_CANONICAL_SHA256,
    PERCENTILE_METHOD,
    PILOT_HAND_INDICES,
    PILOT_ROOT_INDICES,
    PRODUCTION_LABEL_BUDGET,
    SEED_STRIDE,
    STEP5_CONTRACT_BYTE_SHA256,
    STEP6C_BEHAVIOR_SCHEDULE_SCHEMA,
    STEP6C_RUN_ID,
    STEP6C_SCHEDULE_ROW_SCHEMA,
    TRAIN_BEHAVIOR_SEED_BASE,
    TRAIN_CANDIDATE_SEED_BASE,
    TRAIN_CHILD_SEED_BASE,
    TRAIN_CONFIRMATION_SEED_BASE,
    TRAIN_EVALUATION_SEED_BASE,
    TRAIN_HAND_COUNT,
    TRAIN_HAND_SEED_BASE,
    behavior_profile_for_train_index,
    canonical_sha256,
    nearest_rank_percentile,
    pilot_shard_for_hand_index,
    profile_counts,
    schedule_row,
    schedule_rows,
    seed_set,
    train_seed_values,
    validate_frozen_schedule,
)
from ofc_regular.validate_hu_m31_t3_step6c_contract import (
    validate_contract_file,
    validate_contract_payload,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO_ROOT / "configs/hu_joint_policy_m31_t3_step6c_contract.json"


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_step6c_contract_hashes_are_frozen() -> None:
    raw = CONTRACT_PATH.read_bytes()
    payload = json.loads(raw)
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_STEP6C_CONTRACT_BYTE_SHA256
    assert canonical_sha256(payload) == EXPECTED_STEP6C_CONTRACT_CANONICAL_SHA256
    step5 = REPO_ROOT / "configs/hu_joint_policy_m31_t3_step5_contract.json"
    assert hashlib.sha256(step5.read_bytes()).hexdigest() == STEP5_CONTRACT_BYTE_SHA256


def test_search_budgets_and_run_id_are_exact() -> None:
    assert STEP6C_RUN_ID == "hu-m31-step6c-production-label-pilot-v1"
    assert PRODUCTION_LABEL_BUDGET.to_dict() == {
        "label": "production_label_8_32_4_0",
        "candidate_samples": 8,
        "evaluation_samples": 32,
        "downstream_t3_samples": 4,
        "downstream_t4_samples": 0,
    }
    assert CONFIRMATION_BUDGET.to_dict() == {
        "label": "independent_confirmation_8_128_4_0",
        "candidate_samples": 8,
        "evaluation_samples": 128,
        "downstream_t3_samples": 4,
        "downstream_t4_samples": 0,
    }


def test_seed_formula_and_six_namespaces_are_frozen() -> None:
    assert train_seed_values(0) == {
        "hand": TRAIN_HAND_SEED_BASE,
        "behavior": TRAIN_BEHAVIOR_SEED_BASE,
        "candidate": TRAIN_CANDIDATE_SEED_BASE,
        "evaluation": TRAIN_EVALUATION_SEED_BASE,
        "child": TRAIN_CHILD_SEED_BASE,
        "confirmation": TRAIN_CONFIRMATION_SEED_BASE,
    }
    expected_49 = {
        name: base + 49 * SEED_STRIDE
        for name, base in (
            ("hand", TRAIN_HAND_SEED_BASE),
            ("behavior", TRAIN_BEHAVIOR_SEED_BASE),
            ("candidate", TRAIN_CANDIDATE_SEED_BASE),
            ("evaluation", TRAIN_EVALUATION_SEED_BASE),
            ("child", TRAIN_CHILD_SEED_BASE),
            ("confirmation", TRAIN_CONFIRMATION_SEED_BASE),
        )
    }
    assert train_seed_values(49) == expected_49
    assert len(seed_set(PILOT_HAND_INDICES)) == 300
    assert len(seed_set(range(TRAIN_HAND_COUNT))) == TRAIN_HAND_COUNT * 6


def test_sha256_block_shuffle_has_exact_full_pilot_and_shard_quotas() -> None:
    evidence = validate_frozen_schedule()
    assert set(evidence["full_profile_counts"].values()) == {1200}
    assert set(evidence["pilot_profile_counts"].values()) == {10}
    assert all(
        set(counts.values()) == {5}
        for counts in evidence["pilot_shard_profile_counts"].values()
    )
    assert evidence["pilot_schedule_sha256"] == (
        "278f3b6d0fdef52129fe0e4c112cfa7560fc1291d2fa7355362ae54b417bd713"
    )


def test_schedule_formula_is_content_free_and_not_the_canary_cycle() -> None:
    assert STEP6C_BEHAVIOR_SCHEDULE_SCHEMA.endswith("block_shuffle_v2")
    first_block = [behavior_profile_for_train_index(index) for index in range(5)]
    assert first_block == [
        "stage9f_p2",
        "stage7_m5_r10",
        "stage3_baseline",
        "random_exact_final",
        "stage19_p0",
    ]
    assert set(first_block) == set(M31_T3_BEHAVIOR_PROFILES)
    assert first_block != list(M31_T3_BEHAVIOR_PROFILES)


def test_pilot_and_confirmation_indices_are_exact_and_balanced() -> None:
    assert PILOT_HAND_INDICES == tuple(range(50))
    assert PILOT_ROOT_INDICES == tuple(range(100))
    assert CONFIRMATION_HAND_INDICES == (5, 16, 29, 39, 45)
    assert CONFIRMATION_ROOT_INDICES == (10, 11, 32, 33, 58, 59, 78, 79, 90, 91)
    assert profile_counts(CONFIRMATION_HAND_INDICES) == {
        profile: 1 for profile in M31_T3_BEHAVIOR_PROFILES
    }
    assert [index % 2 for index in CONFIRMATION_ROOT_INDICES] == [0, 1] * 5


def test_schedule_rows_have_one_strict_schema_and_two_shards() -> None:
    rows = schedule_rows(PILOT_HAND_INDICES)
    assert len(rows) == 50
    assert {row["pilot_shard"] for row in rows} == {0, 1}
    assert sum(row["pilot_shard"] == 0 for row in rows) == 25
    assert sum(row["pilot_shard"] == 1 for row in rows) == 25
    assert sum(row["confirmation"] for row in rows) == 5
    assert all(row["schema"] == STEP6C_SCHEDULE_ROW_SCHEMA for row in rows)
    assert set(rows[0]) == {
        "schema",
        "schedule_schema",
        "split",
        "train_hand_index",
        "root_indices",
        "profile",
        "seeds",
        "pilot",
        "pilot_shard",
        "confirmation",
    }
    assert schedule_row(25)["pilot_shard"] == 1


def test_nearest_rank_tail_semantics_are_frozen_for_ten_roots() -> None:
    values = [float(value) for value in range(10)]
    assert PERCENTILE_METHOD == "nearest_rank_ceil_n_times_q_v1"
    assert nearest_rank_percentile(values, 0.5) == 4.0
    assert nearest_rank_percentile(values, 0.95) == 9.0
    assert nearest_rank_percentile(values, 0.99) == 9.0
    with pytest.raises(ValueError, match="must not be empty"):
        nearest_rank_percentile([], 0.95)
    with pytest.raises(ValueError, match="finite"):
        nearest_rank_percentile([0.0, float("nan")], 0.95)


@pytest.mark.parametrize("value", [-1, TRAIN_HAND_COUNT, True, 1.5])
def test_invalid_train_indices_fail_closed(value) -> None:
    with pytest.raises((TypeError, ValueError)):
        train_seed_values(value)


def test_nonpilot_shard_and_duplicate_schedule_indices_fail_closed() -> None:
    with pytest.raises(ValueError, match="outside"):
        pilot_shard_for_hand_index(50)
    with pytest.raises(ValueError, match="unique"):
        schedule_rows([0, 0])


def test_live_contract_and_all_anchors_validate() -> None:
    report = validate_contract_file(CONTRACT_PATH, repo_root=REPO_ROOT)
    assert report["status"] == "pass"
    assert report["contract_canonical_sha256"] == (
        EXPECTED_STEP6C_CONTRACT_CANONICAL_SHA256
    )
    assert report["spot_quality_pilot_authorized"] is False
    assert report["production_teacher_fanout_authorized"] is False
    assert report["training_authorized"] is False
    assert report["current_profile_changed"] is False
    assert report["named_profile_added"] is False
    assert report["runtime_policy_activated"] is False
    assert report["anchors"]["accepted_linux_native_library"]["sha256"] == (
        ACCEPTED_NATIVE_LIBRARY_SHA256
    )
    assert report["anchors"]["accepted_linux_feature_encoder"]["sha256"] == (
        ACCEPTED_FEATURE_ENCODER_SHA256
    )


def test_contract_mutation_is_rejected_before_any_action() -> None:
    payload = copy.deepcopy(_contract())
    payload["activation_guards"]["training_authorized"] = True
    with pytest.raises(ValueError, match="canonical contract"):
        validate_contract_payload(payload, repo_root=REPO_ROOT, verify_anchors=False)


def test_byte_different_contract_file_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "contract.json"
    target.write_text(
        CONTRACT_PATH.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="byte SHA-256"):
        validate_contract_file(target, repo_root=REPO_ROOT, verify_anchors=False)


def test_contract_records_confirmation_rng_reuse_and_disjoint_boundary() -> None:
    search = _contract()["search"]
    confirmation = _contract()["confirmation_subset"]
    assert search["confirmation_candidate_seed_source"] == "candidate"
    assert search["confirmation_evaluation_seed_source"] == "confirmation"
    assert search["confirmation_continuation_seed_source"] == "child"
    assert search["confirmation_reuses_primary_run_id"] is True
    assert search["confirmation_candidate_rng_keys_equal_primary_subset"] is True
    assert confirmation["expected_primary_candidate_rng_keys"] == 800
    assert confirmation["expected_primary_evaluation_rng_keys"] == 3200
    assert confirmation["expected_confirmation_evaluation_rng_keys"] == 1280
