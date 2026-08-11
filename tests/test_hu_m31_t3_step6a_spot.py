from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from ofc_regular.ai_profiles import ModelBundle
from ofc_regular.hu_m31_t3_behavior_roots import (
    M31_T3_BEHAVIOR_PROFILES,
    behavior_profile_for_index,
    generate_behavior_t3_roots,
)
from ofc_regular.hu_m31_t3_step6a_spot import (
    STEP6A_DRY_RUN_SCHEMA,
    _parity_golden,
    _write_once,
    build_schedule,
)
from ofc_regular.run_hu_m31_t3_step6a_shard import (
    CANDIDATE_SEED_BASE,
    EVALUATION_SEED_BASE,
    HAND_SEED_BASE,
    SEED_STRIDE,
    ShardSpec,
    _latency,
    portable_decision_payload,
    portable_decision_sha256,
    seed_values,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_behavior_population_cycles_in_frozen_equal_quota_order():
    assert [behavior_profile_for_index(index) for index in range(10)] == [
        *M31_T3_BEHAVIOR_PROFILES,
        *M31_T3_BEHAVIOR_PROFILES,
    ]
    with pytest.raises(ValueError, match="nonnegative"):
        behavior_profile_for_index(-1)


def test_random_behavior_roots_are_balanced_and_hidden_discard_safe():
    first, second = generate_behavior_t3_roots(
        hand_seed=100,
        behavior_seed=200,
        profile="random_exact_final",
        bundle=ModelBundle(),
    )
    assert (first.seat, second.seat) == ("first", "second")
    assert first.street == second.street == "T3"
    assert len(first.hero_private_discards) == len(second.hero_private_discards) == 2
    forbidden = {
        "opponent_private_discards",
        "remaining_deck",
        "world_state",
        "replay_truth",
    }
    assert not (forbidden & set(first.to_dict()))
    assert not (forbidden & set(second.to_dict()))


def test_current_is_rejected_as_behavior_profile():
    with pytest.raises(ValueError, match="explicit and frozen"):
        generate_behavior_t3_roots(
            hand_seed=1,
            behavior_seed=2,
            profile="current",
            bundle=ModelBundle(),
        )


def test_shard0_seed_formula_is_frozen_and_domains_are_distinct():
    first = seed_values(0)
    last = seed_values(24)
    assert first["hand"] == HAND_SEED_BASE
    assert first["candidate"] == CANDIDATE_SEED_BASE
    assert first["evaluation"] == EVALUATION_SEED_BASE
    assert last["hand"] == HAND_SEED_BASE + 24 * SEED_STRIDE
    values = [value for index in range(25) for value in seed_values(index).values()]
    assert len(values) == len(set(values)) == 150


def test_only_shard_zero_is_executable_in_step6a():
    assert ShardSpec("regular-hu-m31-step6a-test", 0, 0).root_count == 50
    with pytest.raises(ValueError, match="shard 0 only"):
        ShardSpec("regular-hu-m31-step6a-test", 1, 25)


def test_schedule_reserves_ten_small_shards_but_does_not_authorize_them():
    rows = build_schedule("regular-hu-m31-step6a-test")
    assert len(rows) == 10
    assert rows[0]["global_hand_start"] == 0
    assert rows[-1]["global_hand_start"] == 225
    assert {row["hand_count"] for row in rows} == {25}
    assert {row["root_count"] for row in rows} == {50}


def test_portable_digest_ignores_only_binary_identity_and_latency():
    base = {
        "selected_action_key": "rak1:test",
        "action_values": [{"action_key": "rak1:test", "evaluation_ev": 1.25}],
        "native_library_sha256": "a" * 64,
        "native_latency_ms": 1.0,
        "validation_latency_ms": 2.0,
        "total_latency_ms": 3.0,
        "semantic_result_digest": "b" * 64,
        "result_digest": "c" * 64,
    }
    changed_binary = deepcopy(base)
    changed_binary["native_library_sha256"] = "d" * 64
    changed_binary["total_latency_ms"] = 999.0
    assert portable_decision_sha256(base) == portable_decision_sha256(changed_binary)
    changed_value = deepcopy(base)
    changed_value["action_values"][0]["evaluation_ev"] = 1.5
    assert portable_decision_sha256(base) != portable_decision_sha256(changed_value)
    assert "native_library_sha256" not in portable_decision_payload(base)


def test_latency_percentiles_sort_observations_before_indexing():
    summary = _latency([30.0, 1.0, 20.0, 10.0, 40.0])
    assert summary == {
        "count": 5,
        "mean_seconds": 20.2,
        "p50_seconds": 20.0,
        "p95_seconds": 40.0,
        "p99_seconds": 40.0,
        "max_seconds": 40.0,
    }


def test_parity_golden_binds_both_step4_smoke_seats():
    golden = _parity_golden(REPO_ROOT)
    assert golden["schema"] == "hu_m31_t3_step6a_parity_golden_v1"
    assert [row["seat"] for row in golden["rows"]] == ["first", "second"]
    assert all(len(row["portable_decision_sha256"]) == 64 for row in golden["rows"])
    assert golden["runtime_config"]["downstream_t4_samples"] == 0


def test_step6a_artifact_writer_is_write_once(tmp_path):
    path = tmp_path / "receipt.json"
    _write_once(path, {"schema": STEP6A_DRY_RUN_SCHEMA})
    assert json.loads(path.read_text(encoding="utf-8"))["schema"] == (
        STEP6A_DRY_RUN_SCHEMA
    )
    with pytest.raises(FileExistsError, match="immutable"):
        _write_once(path, {"schema": "changed"})


def test_startup_contains_remote_resume_drill_and_done_last():
    startup = (REPO_ROOT / "scripts/startup_hu_m31_t3_step6a.sh").read_text(
        encoding="utf-8"
    )
    assert "--stop-after-tasks 1" in startup
    assert "progress/shard-000" in startup
    assert '[[ "$RECOVERED" -lt 1 ]]' in startup
    assert 'upload_once "$RESULT/DONE.json"' in startup
    assert 'SHARD" != 0' in startup


def test_linux_search_workers_use_spawn_after_rust_parity_initialization():
    runner = (REPO_ROOT / "src/ofc_regular/run_hu_m31_t3_step6a_shard.py").read_text(
        encoding="utf-8"
    )
    assert 'mp_context=multiprocessing.get_context("spawn")' in runner
