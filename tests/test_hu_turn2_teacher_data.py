import numpy as np

import json
import hashlib
import random
from collections import defaultdict
from itertools import permutations

import pytest

import ofc_regular.hu_turn2_teacher_data as t2_teacher

from ofc_regular.action_key import action_key
from ofc_regular.counter_rng import COUNTER_RNG_SCHEMA
from ofc_regular.cards import ALL_CARDS
from ofc_regular.hu_belief import sample_hidden_card_particles, turn2_actor_observation
from ofc_regular.final_turn_decision_cache import (
    FinalTurnDecisionCache,
    decide_final_turn_exact,
    final_turn_canonical_key,
)
from ofc_regular.hu_turn2_teacher_data import (
    _ActionRolloutAggregate,
    _candidate_pool_record,
    _cheap_no_rollout_proxy,
    _merge_t3_batch_profile,
    _paired_delta_stats,
    _passes_no_rollout_pool,
    _passes_source_bucket,
    _rollout_after_hero_t2_action,
    _stage3_feature_teacher_run_hash,
    _t3_continuation_metadata,
    build_hu_turn2_sample,
    evaluate_hu_turn2_actions,
    write_hu_turn2_final_turn_cache_stats,
    write_hu_turn2_final_turn_profile_csv,
    write_hu_turn2_final_turn_slow_states,
)
from ofc_regular.hu_late_street_teacher import T4SearchConfig
from ofc_regular.policy import RegularAiPolicy
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.state import Board
from ofc_regular.teacher import DEFAULT_FL_EV, terminal_score


class FirstActionModel:
    def choose_action_index(self, sample):
        return 0

    def predict_sample(self, sample):
        values = np.zeros(len(sample["actions"]), dtype=np.float64)
        values[0] = 1.0
        return values


def _hero_t2_board():
    return Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd", "6c"],
        bottom=["9c", "9d", "9s"],
    )


def _opponent_t2_board():
    return Board.from_rows(
        top=["2h"],
        middle=["3h", "4h", "5h"],
        bottom=["7h", "8h", "Th"],
    )


def _t2_belief_kwargs(*, future_samples: int, seed: int):
    observation = turn2_actor_observation(
        hero_board=_hero_t2_board(),
        opponent_public_board=_opponent_t2_board(),
        dealt_cards=("Qs", "Ah", "7d"),
        hero_seat="first",
        hero_private_discards=("2c",),
    )
    return {
        "observation": observation,
        "belief_batch": sample_hidden_card_particles(
            observation,
            base_seed=seed,
            run_id=f"t2_teacher_test|samples={future_samples}",
            sample_count=future_samples,
        ),
    }


def test_paired_delta_stats_use_aligned_future_scores():
    rollout_by_action = {
        0: _ActionRolloutAggregate(scores=[10.0, 12.0, 14.0, 16.0]),
        3: _ActionRolloutAggregate(scores=[11.0, 13.0, 18.0, 20.0]),
    }

    stats = _paired_delta_stats(rollout_by_action, baseline_index=0, action_indices={0, 3})

    assert stats["paired_delta_candidate_index"] == 3
    assert stats["paired_delta_baseline_index"] == 0
    assert stats["paired_delta_count"] == 4
    assert stats["paired_delta_mean"] == 2.5
    assert stats["paired_delta_standard_error"] > 0.0
    assert stats["paired_delta_p50"] == 2.5
    assert stats["paired_delta_by_action"][0]["candidate_index"] == 3


def test_paired_delta_stats_preserve_negative_candidate_delta():
    rollout_by_action = {
        0: _ActionRolloutAggregate(scores=[10.0, 12.0, 14.0, 16.0]),
        3: _ActionRolloutAggregate(scores=[9.0, 11.0, 12.0, 15.0]),
    }

    stats = _paired_delta_stats(rollout_by_action, baseline_index=0, action_indices={0, 3})

    assert stats["paired_delta_candidate_index"] == 3
    assert stats["paired_delta_count"] == 4
    assert stats["paired_delta_mean"] == -1.25
    assert stats["paired_delta_max"] == -1.0
    assert stats["paired_delta_lt0_rate"] == 1.0


def test_paired_delta_stats_include_component_tail_summaries():
    rollout_by_action = {
        0: _ActionRolloutAggregate(
            scores=[10.0, 12.0, 14.0, 16.0],
            component_values={
                "royalty_delta": [1.0, 1.0, 2.0, 2.0],
                "fl_delta": [0.0, 0.0, 0.0, 0.0],
            },
        ),
        3: _ActionRolloutAggregate(
            scores=[11.0, 13.0, 18.0, 20.0],
            component_values={
                "royalty_delta": [4.0, 3.0, 6.0, 1.0],
                "fl_delta": [0.0, -10.0, 5.0, -25.0],
            },
        ),
    }

    stats = _paired_delta_stats(rollout_by_action, baseline_index=0, action_indices={0, 3})
    summary = stats["paired_delta_by_action"][0]["component_delta_summaries"]

    assert summary["royalty_delta"]["mean"] == 2.0
    assert summary["royalty_delta"]["lt0_rate"] == 0.25
    assert summary["fl_delta"]["mean"] == -7.5
    assert summary["fl_delta"]["le_neg20_rate"] == 0.25


def test_evaluate_hu_turn2_actions_writes_common_future_teacher_fields():
    sample = evaluate_hu_turn2_actions(
        board=_hero_t2_board(),
        dealt_cards=["Qs", "Ah", "7d"],
        opponent_board=_opponent_t2_board(),
        hero_seat="first",
        continuation_policy=RegularAiPolicy(seed=1),
        opponent_policy=RegularAiPolicy(seed=2),
        baseline_turn2_model=FirstActionModel(),
        future_samples=2,
        future_rollout_seed=99,
        **_t2_belief_kwargs(future_samples=2, seed=99),
    )

    assert sample is not None
    assert sample["schema"] == "hu_turn2_stage1"
    assert sample["phase"] == "hu_turn2_7card"
    assert sample["turn"] == "T2"
    assert sample["common_random_futures"] is True
    assert sample["future_rollout_seed"] == 99
    assert len(sample["common_random_future_digest"]) == 64
    assert sample["continuation_policy_T3"] == "Stage3_HU_reference_default"
    assert sample["continuation"]["stage7_enabled"] is False
    assert sample["actions"]
    assert sample["legal_actions"] == sample["actions"]
    assert all(action["rollout_count"] == 2 for action in sample["actions"])
    assert all(action["common_random_future_digest"] == sample["common_random_future_digest"] for action in sample["actions"])
    assert all("ev_standard_error" in action for action in sample["actions"])
    assert "delta_best_vs_baseline" in sample
    assert "SE_delta_best_vs_baseline" in sample
    assert sample["profiling"]["seconds_total"] > 0.0
    assert "continuation_decision_seconds" in sample["profiling"]
    assert "t3_continuation_override_rate" in sample["downstream_features"]
    assert "t3_stage7_override_rate" in sample["downstream_features"]


def test_stage7_continuation_metadata_remains_explicit_opt_in():
    metadata = _t3_continuation_metadata("stage7_m5_r10")
    sample = evaluate_hu_turn2_actions(
        board=_hero_t2_board(),
        dealt_cards=["Qs", "Ah", "7d"],
        opponent_board=_opponent_t2_board(),
        hero_seat="first",
        continuation_policy=RegularAiPolicy(seed=1),
        opponent_policy=RegularAiPolicy(seed=2),
        continuation_policy_name="Stage7_candidate_A_m5_r10",
        continuation_metadata=metadata,
        baseline_turn2_model=FirstActionModel(),
        future_samples=1,
        future_rollout_seed=99,
        **_t2_belief_kwargs(future_samples=1, seed=99),
    )

    assert sample is not None
    assert sample["continuation_policy_T3"] == "Stage7_candidate_A_m5_r10"
    assert sample["continuation"]["mode"] == "stage7_m5_r10"
    assert sample["continuation"]["stage7_enabled"] is True


def test_stage9d_continuation_metadata_includes_support_and_gate():
    metadata = _t3_continuation_metadata("stage9d_p07_relaxed_both")

    assert metadata["policy"] == "Stage9d_second3seed_gate_p07_relaxed_both"
    assert metadata["mode"] == "stage9d_p07_relaxed_both"
    assert metadata["stage7_enabled"] is True
    assert metadata["stage7_hu_turn3_min_margin"] == 0.5
    assert metadata["stage7_hu_turn3_reference_min_margin"] == 0.0
    assert metadata["hu_turn3_min_support_margin"] == 0.5
    assert metadata["hu_turn3_min_model_score"] == 5.0
    assert metadata["hu_turn3_min_gate_probability"] == 0.7
    assert metadata["support_model_path"]
    assert metadata["gate_model_path"]


def test_teacher_run_hash_separates_t3_continuation_modes():
    common = {
        "seed": 1,
        "samples": 2,
        "future_samples": 3,
        "source_bucket": "natural",
        "stage3_feature_encoder_mode": "rust_direct",
        "disable_stage3_feature_fast_path": False,
    }

    stage3_hash = _stage3_feature_teacher_run_hash(
        **common,
        t3_continuation="stage3_reference_default",
    )
    stage7_hash = _stage3_feature_teacher_run_hash(
        **common,
        t3_continuation="stage7_m5_r10",
    )
    stage9d_hash = _stage3_feature_teacher_run_hash(
        **common,
        t3_continuation="stage9d_p07_relaxed_both",
    )

    assert stage3_hash != stage7_hash
    assert stage3_hash != stage9d_hash
    assert stage7_hash != stage9d_hash


def test_t3_batch_profile_merge_preserves_string_metadata():
    profile = {
        "continuation_decision_seconds": 0.0,
        "summary_aggregation_version": "hu_turn2_batch6_5_v1",
        "stage3_feature_mode": "scalar",
        "feature_dtype": "",
    }
    batch_profile = {
        "summary_aggregation_version": "hu_turn2_batch6_5_v1",
        "stage3_feature_mode": "rust_direct",
        "feature_dtype": "float32",
        "t3_continuation_total_seconds": 1.25,
        "stage3_feature_generation_time": 2.0,
        "feature_column_count": 1076,
        "memory_peak_mb": 128.0,
    }

    _merge_t3_batch_profile(profile, batch_profile)

    assert profile["summary_aggregation_version"] == "hu_turn2_batch6_5_v1"
    assert profile["stage3_feature_mode"] == "rust_direct"
    assert profile["feature_dtype"] == "float32"
    assert profile["stage3_feature_generation_time"] == 2.0
    assert profile["feature_column_count"] == 1076.0
    assert profile["memory_peak_mb"] == 128.0
    assert profile["continuation_decision_seconds"] == 1.25


def test_build_hu_turn2_sample_adds_identity_and_distribution_metadata():
    sample = build_hu_turn2_sample(
        sample_id=3,
        state_id=5,
        seed=7,
        hand_seed=11,
        hand_index=2,
        player=0,
        board=_hero_t2_board(),
        dealt_cards=["Qs", "Ah", "7d"],
        opponent_board=_opponent_t2_board(),
        dead_cards=["2c"],
        hero_private_discards=["2c"],
        hero_seat="first",
        continuation_policy=RegularAiPolicy(seed=1),
        opponent_policy=RegularAiPolicy(seed=2),
        baseline_turn2_model=FirstActionModel(),
        future_samples=2,
        future_rollout_seed=99,
        source_bucket="natural",
    )

    assert sample is not None
    assert sample["sample_id"] == 3
    assert sample["state_id"] == 5
    assert sample["hand_id"] == 11
    assert sample["source_bucket"] == "natural"
    assert sample["cards_to_place"] == ["Qs", "Ah", "7d"]
    assert sample["missing"] == 0
    assert sample["teacher_label"] in {"positive", "negative", "gray"}
    assert sample["fl_ev_14"] == DEFAULT_FL_EV[14]
    assert sample["fl_ev"]["14"] == DEFAULT_FL_EV[14]
    assert sample["rng_schema"] == COUNTER_RNG_SCHEMA
    assert len(sample["legal_action_set_digest"]) == 64
    assert len(sample["legal_action_order_digest"]) == 64
    assert all(
        action["canonical_action_key"].startswith("rak1:")
        for action in sample["actions"]
    )


def test_build_hu_turn2_sample_separates_true_and_visible_dead_cards():
    opponent = _opponent_t2_board()
    visible_dead = [*opponent.all_cards(), "2c"]
    sample = build_hu_turn2_sample(
        sample_id=3,
        state_id=5,
        seed=7,
        hand_seed=11,
        hand_index=2,
        player=0,
        board=_hero_t2_board(),
        dealt_cards=["Qs", "Ah", "7d"],
        opponent_board=opponent,
        dead_cards=["2c", "3c"],
        visible_dead_cards=visible_dead,
        hero_private_discards=["2c"],
        opponent_private_discards=["3c"],
        hero_seat="first",
        continuation_policy=RegularAiPolicy(seed=1),
        opponent_policy=RegularAiPolicy(seed=2),
        baseline_turn2_model=FirstActionModel(),
        future_samples=2,
        future_rollout_seed=99,
        source_bucket="natural",
    )

    assert sample is not None
    assert sample["dead_cards"] == visible_dead
    assert sample["visible_dead_cards"] == visible_dead
    assert "2c" in sample["visible_dead_cards"]
    assert "3c" not in sample["visible_dead_cards"]
    assert sample["hero_private_discards"] == ["2c"]
    assert "opponent_private_discards" not in sample
    assert sample["true_dead_cards"] == ["2c", "3c"]
    assert sample["true_opponent_private_discards"] == ["3c"]
    assert sample["replay_truth"]["opponent_private_discards"] == ["3c"]
    assert sample["replay_ready"] is True
    assert sample["belief_conditioned"] is True
    assert len(sample["belief_batch_digest"]) == 64


def test_build_hu_turn2_sample_fallback_hides_opponent_private_discards():
    sample = build_hu_turn2_sample(
        sample_id=3,
        state_id=5,
        seed=7,
        hand_seed=11,
        hand_index=2,
        player=0,
        board=_hero_t2_board(),
        dealt_cards=["Qs", "Ah", "7d"],
        opponent_board=_opponent_t2_board(),
        dead_cards=["2c", "3c"],
        hero_private_discards=["2c"],
        opponent_private_discards=["3c"],
        hero_seat="first",
        continuation_policy=RegularAiPolicy(seed=1),
        opponent_policy=RegularAiPolicy(seed=2),
        baseline_turn2_model=FirstActionModel(),
        future_samples=2,
        future_rollout_seed=99,
        source_bucket="natural",
    )

    assert sample is not None
    assert sample["dead_cards"] == sample["visible_dead_cards"]
    assert "2c" in sample["visible_dead_cards"]
    assert "3c" not in sample["visible_dead_cards"]


def test_t2_belief_values_ignore_realized_opponent_discard_identity():
    opponent = _opponent_t2_board()
    visible_dead = [*opponent.all_cards(), "2c"]

    def build(opponent_discard: str):
        return build_hu_turn2_sample(
            sample_id=3,
            state_id=5,
            seed=7,
            hand_seed=11,
            hand_index=2,
            player=0,
            board=_hero_t2_board(),
            dealt_cards=["Qs", "Ah", "7d"],
            opponent_board=opponent,
            dead_cards=["2c", opponent_discard],
            visible_dead_cards=visible_dead,
            hero_private_discards=["2c"],
            opponent_private_discards=[opponent_discard],
            hero_seat="first",
            continuation_policy=RegularAiPolicy(seed=1),
            opponent_policy=RegularAiPolicy(seed=2),
            baseline_turn2_model=FirstActionModel(),
            future_samples=2,
            future_rollout_seed=99,
            source_bucket="natural",
        )

    first = build("3c")
    second = build("4c")
    assert first is not None and second is not None
    assert first["belief_batch_digest"] == second["belief_batch_digest"]
    assert first["common_random_future_digest"] == second["common_random_future_digest"]
    assert [row["canonical_action_key"] for row in first["actions"]] == [
        row["canonical_action_key"] for row in second["actions"]
    ]
    assert [row["score"] for row in first["actions"]] == [
        row["score"] for row in second["actions"]
    ]


def test_t2_all_dealt_permutations_preserve_belief_futures_and_action_evs():
    class TieModel:
        def predict_sample(self, sample):
            return np.zeros(len(sample["actions"]), dtype=np.float64)

    def policy(*, seat: str, seed: int) -> RegularAiPolicy:
        return RegularAiPolicy(
            turn1_model=TieModel(),
            turn2_model=TieModel(),
            turn3_model=TieModel(),
            seat=seat,
            seed=seed,
        )

    results = []
    for dealt in permutations(("Qs", "Ah", "7d")):
        observation = turn2_actor_observation(
            hero_board=_hero_t2_board(),
            opponent_public_board=_opponent_t2_board(),
            dealt_cards=dealt,
            hero_seat="first",
            hero_private_discards=("2c",),
        )
        belief = sample_hidden_card_particles(
            observation,
            base_seed=20260712,
            run_id="t2-dealt-permutation-parity",
            sample_count=1,
        )
        sample = evaluate_hu_turn2_actions(
            board=_hero_t2_board(),
            dealt_cards=dealt,
            opponent_board=_opponent_t2_board(),
            hero_seat="first",
            continuation_policy=policy(seat="first", seed=1),
            opponent_policy=policy(seat="second", seed=2),
            baseline_turn2_model=TieModel(),
            future_samples=1,
            future_rollout_seed=20260712,
            use_final_turn_cache=False,
            observation=observation,
            belief_batch=belief,
        )
        assert sample is not None
        results.append(
            {
                "observation_fingerprint": observation.fingerprint(),
                "belief_batch_digest": sample["belief_batch_digest"],
                "common_random_future_digest": sample[
                    "common_random_future_digest"
                ],
                "legal_action_set_digest": sample["legal_action_set_digest"],
                "action_evs": {
                    row["canonical_action_key"]: (
                        row["score"],
                        row["rollout_count"],
                    )
                    for row in sample["actions"]
                },
            }
        )

    first = results[0]
    assert len(first["action_evs"]) == len(
        generate_turn_actions(_hero_t2_board(), ("Qs", "Ah", "7d"))
    )
    assert all(result == first for result in results[1:])


class RecordingTurnPolicy:
    def __init__(self, *, seat: str) -> None:
        self.seat = seat
        self.calls = []

    def choose_action(self, board, dealt_cards, *, dead_cards=(), opponent_board=None, **_kwargs):
        actions = generate_turn_actions(board, dealt_cards)
        action = actions[0]
        self.calls.append(
            {
                "board_count": board.card_count(),
                "dealt": tuple(dealt_cards),
                "dead_cards": tuple(dead_cards),
                "opponent_board": opponent_board,
                "action": action,
            }
        )
        return action


def _t4_rollout_root():
    hero = Board.from_rows(
        top=("4d", "5c"),
        middle=("Jh", "Td", "6h", "8h", "As"),
        bottom=("2c", "Jd", "Kd", "Th"),
    )
    opponent = Board.from_rows(
        top=("Tc", "Qc"),
        middle=("7d", "3s", "6c", "8s", "9h"),
        bottom=("9d", "7c", "Qs", "7s"),
    )
    return {
        "hero": hero,
        "opponent": opponent,
        "hero_deal": ("9s", "Ah", "4s"),
        "opponent_deal": ("3d", "Kh", "2h"),
        "hero_private": ("Ac", "2d", "5d"),
        "opponent_private": ("2s", "3h", "9c"),
    }


def _record_t4_selector_calls(monkeypatch):
    safe_observations = []
    exact_orders = []
    real_safe = t2_teacher.select_t4_action
    real_exact = t2_teacher.decide_final_turn_exact

    def safe_wrapper(observation, **kwargs):
        safe_observations.append(observation)
        return real_safe(observation, **kwargs)

    def exact_wrapper(**kwargs):
        exact_orders.append(kwargs["to_act_order"])
        return real_exact(**kwargs)

    monkeypatch.setattr(t2_teacher, "select_t4_action", safe_wrapper)
    monkeypatch.setattr(t2_teacher, "decide_final_turn_exact", exact_wrapper)
    return safe_observations, exact_orders


def _assert_t4_selector_calls(root, safe_observations, exact_orders):
    assert len(safe_observations) == 1
    observation = safe_observations[0]
    assert observation.to_act_order == "first"
    assert observation.dealt_cards == root["hero_deal"]
    assert observation.hero_private_discards == root["hero_private"]
    actor_visible = {
        *observation.hero_board.all_cards(),
        *observation.opponent_public_board.all_cards(),
        *observation.dealt_cards,
        *observation.hero_private_discards,
    }
    assert actor_visible.isdisjoint(root["opponent_deal"])
    assert actor_visible.isdisjoint(root["opponent_private"])
    assert exact_orders == ["second"]


def test_scalar_t2_rollout_uses_infoset_safe_first_t4_and_exact_second(
    monkeypatch,
):
    root = _t4_rollout_root()
    safe_observations, exact_orders = _record_t4_selector_calls(monkeypatch)
    config = T4SearchConfig(
        candidate_samples=1,
        evaluation_samples=1,
        seed=20260713,
        run_id="t2-scalar-safe-t4",
    )

    score = _rollout_after_hero_t2_action(
        hero_board=root["hero"],
        opponent_board=root["opponent"],
        dead_cards=(*root["hero_private"], *root["opponent_private"]),
        hero_private_discards=root["hero_private"],
        opponent_private_discards=root["opponent_private"],
        hero_seat="first",
        future_cards=[*root["hero_deal"], *root["opponent_deal"]],
        hero_policy=RegularAiPolicy(seat="first", seed=1),
        opponent_policy=RegularAiPolicy(seat="second", seed=2),
        t4_search_config=config,
    )

    assert score is not None
    _assert_t4_selector_calls(root, safe_observations, exact_orders)


def test_batched_t2_rollout_uses_infoset_safe_first_t4_and_exact_second(
    monkeypatch,
):
    root = _t4_rollout_root()
    safe_observations, exact_orders = _record_t4_selector_calls(monkeypatch)
    item = t2_teacher._BatchedRollout(
        action_index=0,
        hero=root["hero"],
        opponent=root["opponent"],
        dead=[*root["hero_private"], *root["opponent_private"]],
        hero_private_discards=list(root["hero_private"]),
        opponent_private_discards=list(root["opponent_private"]),
        future_cards=[*root["hero_deal"], *root["opponent_deal"]],
        future_index=0,
        future_rollout_seed=20260713,
        root_fingerprint="t2-batched-safe-t4",
    )
    config = T4SearchConfig(
        candidate_samples=1,
        evaluation_samples=1,
        seed=20260713,
        run_id="t2-batched-safe-t4",
    )

    result = t2_teacher._finish_rollout_after_t3(
        item,
        hero_seat="first",
        hero_policy=RegularAiPolicy(seat="first", seed=1),
        opponent_policy=RegularAiPolicy(seat="second", seed=2),
        profile=defaultdict(float),
        t4_search_config=config,
    )

    assert result is not None
    _assert_t4_selector_calls(root, safe_observations, exact_orders)


def test_hu_turn2_rollout_hides_opponent_private_discards_from_actor_policy():
    hero_after_t2 = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd", "6c", "Qs"],
        bottom=["9c", "9d", "9s", "Ah"],
    )
    opponent = _opponent_t2_board()
    hero_policy = RecordingTurnPolicy(seat="first")
    opponent_policy = RecordingTurnPolicy(seat="second")

    score = _rollout_after_hero_t2_action(
        hero_board=hero_after_t2,
        opponent_board=opponent,
        dead_cards=["2c", "3c"],
        hero_private_discards=["2c"],
        opponent_private_discards=["3c"],
        hero_seat="first",
        future_cards=[
            "4c",
            "5c",
            "6d",
            "7c",
            "8c",
            "Td",
            "Jc",
            "Qc",
            "Kc",
            "Ad",
            "2d",
            "3d",
            "4d",
            "5d",
            "6s",
        ],
        hero_policy=hero_policy,
        opponent_policy=opponent_policy,
    )

    assert score is not None
    assert opponent_policy.calls
    assert "3c" in opponent_policy.calls[0]["dead_cards"]
    assert "2c" not in opponent_policy.calls[0]["dead_cards"]
    assert hero_policy.calls
    assert "2c" in hero_policy.calls[0]["dead_cards"]
    assert "3c" not in hero_policy.calls[0]["dead_cards"]


def test_hu_turn2_rollout_default_private_discards_do_not_leak_true_dead_cards():
    hero_after_t2 = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd", "6c", "Qs"],
        bottom=["9c", "9d", "9s", "Ah"],
    )
    opponent = _opponent_t2_board()
    hero_policy = RecordingTurnPolicy(seat="first")
    opponent_policy = RecordingTurnPolicy(seat="second")

    score = _rollout_after_hero_t2_action(
        hero_board=hero_after_t2,
        opponent_board=opponent,
        dead_cards=["2c", "3c"],
        hero_seat="first",
        future_cards=[
            "4c",
            "5c",
            "6d",
            "7c",
            "8c",
            "Td",
            "Jc",
            "Qc",
            "Kc",
            "Ad",
            "2d",
            "3d",
            "4d",
            "5d",
            "6s",
        ],
        hero_policy=hero_policy,
        opponent_policy=opponent_policy,
    )

    assert score is not None
    assert opponent_policy.calls
    assert "2c" not in opponent_policy.calls[0]["dead_cards"]
    assert "3c" not in opponent_policy.calls[0]["dead_cards"]


def test_from_pool_source_bucket_accepts_without_actual_filtering():
    sample = {
        "delta_best_vs_baseline": 0.0,
        "best_margin": 99.0,
        "teacher_distribution_metrics": {"baseline_disagreement": False},
    }

    assert _passes_source_bucket(sample, "from_pool") is True
    assert _passes_source_bucket(sample, "high_regret") is False
    assert _passes_source_bucket(sample, "low_margin") is False


def test_cheap_no_rollout_candidate_pool_record_has_predicted_proxy_fields():
    proxy = _cheap_no_rollout_proxy(
        board=_hero_t2_board(),
        dealt_cards=["Qs", "Ah", "7d"],
        opponent_board=_opponent_t2_board(),
        dead_cards=["2c"],
        hero_private_discards=["2c"],
        hero_seat="first",
        baseline_turn2_model=FirstActionModel(),
    )
    record = _candidate_pool_record(
        seed=1,
        hand_seed=2,
        hand_index=3,
        state_id=4,
        player=0,
        board=_hero_t2_board(),
        dealt_cards=["Qs", "Ah", "7d"],
        opponent_board=_opponent_t2_board(),
        dead_cards=["2c", "3c"],
        hero_private_discards=["2c"],
        opponent_private_discards=["3c"],
        hero_seat="first",
        source_bucket="high_regret",
        cheap_sample=proxy,
        accept_reason="unit",
        predicted_bucket="predicted_high_regret",
        prefilter_version="cheap_no_rollout_v1",
    )

    assert record["prefilter_version"] == "cheap_no_rollout_v1"
    assert record["predicted_bucket"] == "predicted_high_regret"
    assert record["legal_action_count"] == len(proxy["actions"])
    assert record["state_hash"]
    assert "predicted_margin" in record
    assert "foul_risk_heuristic" in record
    assert "2c" in record["visible_dead_cards"]
    assert "3c" not in record["visible_dead_cards"]
    assert record["hero_private_discards"] == ["2c"]
    assert "opponent_private_discards" not in record
    assert record["true_opponent_private_discards"] == ["3c"]
    assert record["replay_truth"]["opponent_private_discards"] == ["3c"]
    assert record["policy_observation"]["street"] == "T2"


def test_cheap_no_rollout_ties_select_same_keys_for_all_dealt_permutations():
    class TieModel:
        def predict_sample(self, sample):
            return np.zeros(len(sample["actions"]), dtype=np.float64)

    dealt = ("Qs", "Ah", "7d")
    proxies = [
        _cheap_no_rollout_proxy(
            board=_hero_t2_board(),
            dealt_cards=permuted,
            opponent_board=_opponent_t2_board(),
            dead_cards=["2c"],
            hero_private_discards=["2c"],
            hero_seat="first",
            baseline_turn2_model=TieModel(),
        )
        for permuted in permutations(dealt)
    ]

    assert {proxy["baseline_action_key"] for proxy in proxies} == {
        proxies[0]["baseline_action_key"]
    }
    assert {proxy["reference_action_key"] for proxy in proxies} == {
        proxies[0]["reference_action_key"]
    }
    assert {proxy["legal_action_set_digest"] for proxy in proxies} == {
        proxies[0]["legal_action_set_digest"]
    }
    assert all(
        proxy["baseline_action"]["canonical_action_key"]
        == proxy["baseline_action_key"]
        for proxy in proxies
    )


def test_t2_teacher_and_proxy_reject_ambiguous_unsplit_dead_cards():
    common = {
        "board": _hero_t2_board(),
        "dealt_cards": ["Qs", "Ah", "7d"],
        "opponent_board": _opponent_t2_board(),
        "dead_cards": ["2c", "3c"],
    }
    with pytest.raises(ValueError, match="explicit hero_private_discards"):
        _cheap_no_rollout_proxy(
            **common,
            hero_seat="first",
            baseline_turn2_model=FirstActionModel(),
        )

    with pytest.raises(ValueError, match="explicit hero_private_discards"):
        build_hu_turn2_sample(
            sample_id=3,
            state_id=5,
            seed=7,
            hand_seed=11,
            hand_index=2,
            player=0,
            **common,
            hero_seat="first",
            continuation_policy=RegularAiPolicy(seed=1),
            opponent_policy=RegularAiPolicy(seed=2),
            baseline_turn2_model=FirstActionModel(),
            future_samples=1,
            future_rollout_seed=99,
            source_bucket="natural",
        )


def test_cheap_no_rollout_proxy_fallback_hides_opponent_private_discards():
    opponent = _opponent_t2_board()
    explicit_visible = [*opponent.all_cards(), "2c"]
    fallback_proxy = _cheap_no_rollout_proxy(
        board=_hero_t2_board(),
        dealt_cards=["Qs", "Ah", "7d"],
        opponent_board=opponent,
        dead_cards=["2c", "3c"],
        hero_private_discards=["2c"],
        hero_seat="first",
        baseline_turn2_model=FirstActionModel(),
    )
    explicit_proxy = _cheap_no_rollout_proxy(
        board=_hero_t2_board(),
        dealt_cards=["Qs", "Ah", "7d"],
        opponent_board=opponent,
        dead_cards=["2c", "3c"],
        visible_dead_cards=explicit_visible,
        hero_seat="first",
        baseline_turn2_model=FirstActionModel(),
    )

    assert fallback_proxy["dead_card_pressure"] == explicit_proxy["dead_card_pressure"]


def test_no_rollout_pool_predicates_are_bucket_specific():
    proxy = {
        "legal_action_count": 24,
        "baseline_reference_disagree": True,
        "predicted_margin": 0.10,
        "predicted_score_spread": 1.0,
        "baseline_reference_score_gap": 0.75,
        "reference_margin": 1.25,
        "royalty_potential_heuristic": 2.0,
        "foul_risk_heuristic": 0.0,
        "fl_distance_heuristic": 2.0,
    }

    assert _passes_no_rollout_pool(proxy, "predicted_teacher_disagreement")[0] is True
    assert _passes_no_rollout_pool(proxy, "predicted_low_margin")[0] is True
    assert _passes_no_rollout_pool(proxy, "predicted_high_regret")[0] is True


def test_final_turn_canonical_key_is_deterministic_for_card_order():
    board_a = Board.from_rows(
        top=["Qh", "Kc"],
        middle=["Ah", "Ac", "4s", "5s"],
        bottom=["Td", "7h", "7s", "7c", "Th"],
    )
    board_b = Board.from_rows(
        top=["Kc", "Qh"],
        middle=["5s", "4s", "Ac", "Ah"],
        bottom=["Th", "7c", "7s", "7h", "Td"],
    )
    opponent_a = Board.from_rows(
        top=["2h", "3h", "4h"],
        middle=["5c", "2s", "6h", "4c", "8c"],
        bottom=["7d", "Jd", "9d", "Qd", "Kd"],
    )
    opponent_b = Board.from_rows(
        top=["4h", "2h", "3h"],
        middle=["8c", "4c", "6h", "2s", "5c"],
        bottom=["Kd", "Qd", "9d", "Jd", "7d"],
    )

    key_a = final_turn_canonical_key(
        board=board_a,
        opponent_board=opponent_a,
        dealt_cards=["2c", "3c", "4d"],
        dead_cards=["6c", "8h"],
        actor="hero",
        seat="first",
        to_act_order="first",
    )
    key_b = final_turn_canonical_key(
        board=board_b,
        opponent_board=opponent_b,
        dealt_cards=["4d", "2c", "3c"],
        dead_cards=["8h", "6c"],
        actor="hero",
        seat="first",
        to_act_order="first",
    )

    assert key_a == key_b


def test_final_turn_cache_on_off_parity_and_profile_fields():
    board = Board.from_rows(
        top=["Qh", "Kc"],
        middle=["Ah", "Ac", "4s", "5s"],
        bottom=["Td", "7h", "7s", "7c", "Th"],
    )
    opponent = Board.from_rows(
        top=["2h", "3h", "4h"],
        middle=["5c", "2s", "6h", "4c", "8c"],
        bottom=["7d", "Jd", "9d", "Qd", "Kd"],
    )
    cache = FinalTurnDecisionCache(max_size=8)

    uncached, uncached_profile, _ = decide_final_turn_exact(
        board=board,
        dealt_cards=["2c", "3c", "4d"],
        opponent_board=opponent,
        dead_cards=["6c", "8h"],
        actor="hero",
        seat="first",
        to_act_order="second",
        cache=None,
        use_cache=False,
    )
    cached, cached_profile, _ = decide_final_turn_exact(
        board=board,
        dealt_cards=["2c", "3c", "4d"],
        opponent_board=opponent,
        dead_cards=["6c", "8h"],
        actor="hero",
        seat="first",
        to_act_order="second",
        cache=cache,
        use_cache=True,
    )
    cached_again, cached_again_profile, _ = decide_final_turn_exact(
        board=board,
        dealt_cards=["4d", "2c", "3c"],
        opponent_board=opponent,
        dead_cards=["8h", "6c"],
        actor="hero",
        seat="first",
        to_act_order="second",
        cache=cache,
        use_cache=True,
    )

    assert uncached is not None and cached is not None and cached_again is not None
    assert uncached.action == cached.action == cached_again.action
    assert uncached.score == cached.score == cached_again.score
    assert cached_profile.cache_hit is False
    assert cached_again_profile.cache_hit is True
    assert uncached_profile.legal_action_generation_seconds >= 0.0
    assert uncached_profile.exact_scoring_seconds >= 0.0


def test_final_turn_exact_matches_independent_terminal_score_for_all_legal_actions():
    rng = random.Random(20260712)
    hero_row_lengths = (
        (1, 5, 5),
        (2, 4, 5),
        (2, 5, 4),
        (3, 3, 5),
        (3, 4, 4),
        (3, 5, 3),
    )

    for state_index in range(12):
        deck = list(ALL_CARDS)
        rng.shuffle(deck)
        cursor = 0

        top_count, middle_count, bottom_count = hero_row_lengths[
            state_index % len(hero_row_lengths)
        ]
        board = Board.from_rows(
            top=deck[cursor : cursor + top_count],
            middle=deck[
                cursor + top_count : cursor + top_count + middle_count
            ],
            bottom=deck[
                cursor
                + top_count
                + middle_count : cursor
                + top_count
                + middle_count
                + bottom_count
            ],
        )
        cursor += 11

        seat = "first" if state_index % 2 == 0 else "second"
        if seat == "first":
            opponent_lengths = hero_row_lengths[
                (state_index + 1) % len(hero_row_lengths)
            ]
        else:
            opponent_lengths = (3, 5, 5)
        opp_top, opp_middle, opp_bottom = opponent_lengths
        opponent_board = Board.from_rows(
            top=deck[cursor : cursor + opp_top],
            middle=deck[cursor + opp_top : cursor + opp_top + opp_middle],
            bottom=deck[
                cursor
                + opp_top
                + opp_middle : cursor
                + opp_top
                + opp_middle
                + opp_bottom
            ],
        )
        cursor += sum(opponent_lengths)
        dealt = tuple(deck[cursor : cursor + 3])
        dead = tuple(deck[cursor + 3 : cursor + 6])

        actions = generate_turn_actions(board, dealt)
        assert actions
        independently_scored = []
        terminal_opponent = opponent_board if opponent_board.is_complete() else None
        for action in actions:
            final_board = board.place(action.placements)
            assert final_board.is_complete()
            score, _board_score = terminal_score(final_board, terminal_opponent)
            independently_scored.append((float(score), action, final_board))
        independently_scored.sort(
            key=lambda item: (-item[0], action_key(item[1]).sort_key())
        )
        expected_score, expected_action, expected_board = independently_scored[0]

        decision, _profile, metadata = decide_final_turn_exact(
            board=board,
            dealt_cards=dealt,
            opponent_board=opponent_board,
            dead_cards=dead,
            actor="hero",
            seat=seat,
            to_act_order=seat,
            cache=None,
            use_cache=False,
        )

        assert decision is not None
        assert decision.legal_action_count == len(actions)
        assert decision.score == expected_score
        assert action_key(decision.action) == action_key(expected_action)
        assert decision.final_board == expected_board
        assert metadata["final_action_key"] == action_key(expected_action).to_token()


def test_final_turn_exact_action_key_is_invariant_to_all_dealt_permutations():
    board = Board.from_rows(
        top=["Qh", "Kc"],
        middle=["Ah", "Ac", "4s", "5s"],
        bottom=["Td", "7h", "7s", "7c", "Th"],
    )
    opponent = Board.from_rows(
        top=["2h", "3h", "4h"],
        middle=["5c", "2s", "6h", "4c", "8c"],
        bottom=["7d", "Jd", "9d", "Qd", "Kd"],
    )
    cache = FinalTurnDecisionCache(max_size=8)
    uncached_keys = set()
    cached_keys = set()
    scores = set()

    for dealt in permutations(("2c", "3c", "4d")):
        fresh, _, _ = decide_final_turn_exact(
            board=board,
            dealt_cards=dealt,
            opponent_board=opponent,
            dead_cards=("6c", "8h"),
            actor="hero",
            seat="first",
            to_act_order="second",
            cache=None,
            use_cache=False,
        )
        warm, _, _ = decide_final_turn_exact(
            board=board,
            dealt_cards=dealt,
            opponent_board=opponent,
            dead_cards=("8h", "6c"),
            actor="hero",
            seat="first",
            to_act_order="second",
            cache=cache,
            use_cache=True,
        )
        assert fresh is not None and warm is not None
        uncached_keys.add(action_key(fresh.action))
        cached_keys.add(action_key(warm.action))
        scores.add(fresh.score)

    assert len(scores) == 1
    assert len(uncached_keys) == 1
    assert cached_keys == uncached_keys


def test_final_turn_direct_exact_matches_regular_policy():
    board = Board.from_rows(
        top=["Qh", "Kc"],
        middle=["Ah", "Ac", "4s", "5s"],
        bottom=["Td", "7h", "7s", "7c", "Th"],
    )
    opponent = Board.from_rows(
        top=["2h", "3h", "4h"],
        middle=["5c", "2s", "6h", "4c", "8c"],
        bottom=["7d", "Jd", "9d", "Qd", "Kd"],
    )
    dealt = ["2c", "3c", "4d"]

    decision, _, _ = decide_final_turn_exact(
        board=board,
        dealt_cards=dealt,
        opponent_board=opponent,
        dead_cards=["6c", "8h"],
        actor="hero",
        seat="first",
        to_act_order="second",
        cache=None,
        use_cache=False,
    )
    policy_action = RegularAiPolicy(seed=123).choose_action(
        board,
        dealt,
        dead_cards=["6c", "8h"],
        opponent_board=opponent,
    )

    assert decision is not None
    assert decision.action == policy_action


def test_final_turn_cache_preserves_teacher_hash_and_digest():
    def kwargs():
        return dict(
            board=_hero_t2_board(),
            dealt_cards=["Qs", "Ah", "7d"],
            opponent_board=_opponent_t2_board(),
            hero_seat="first",
            continuation_policy=RegularAiPolicy(seed=1),
            opponent_policy=RegularAiPolicy(seed=2),
            baseline_turn2_model=FirstActionModel(),
            future_samples=2,
            future_rollout_seed=99,
            use_batched_continuation=True,
            **_t2_belief_kwargs(future_samples=2, seed=99),
        )

    no_cache = evaluate_hu_turn2_actions(**kwargs(), use_final_turn_cache=False)
    with_cache = evaluate_hu_turn2_actions(
        **kwargs(),
        final_turn_cache=FinalTurnDecisionCache(max_size=128),
        use_final_turn_cache=True,
    )

    def canonical(sample):
        scrubbed = {key: value for key, value in sample.items() if key != "profiling"}
        return hashlib.sha256(json.dumps(scrubbed, sort_keys=True).encode("utf-8")).hexdigest()

    assert no_cache is not None and with_cache is not None
    assert canonical(no_cache) == canonical(with_cache)
    assert no_cache["common_random_future_digest"] == with_cache["common_random_future_digest"]
    assert all(
        action["common_random_future_digest"] == with_cache["common_random_future_digest"]
        for action in with_cache["actions"]
    )
    assert with_cache["profiling"]["final_turn_state_count"] > 0
    assert "final_turn_exact_scoring_time" in with_cache["profiling"]
    assert "final_turn_cache_hit_rate" in with_cache["profiling"]


def test_final_turn_artifact_writers_smoke(tmp_path):
    profiles = [
        {
            "final_turn_state_count": 2.0,
            "final_turn_cache_hit": 1.0,
            "final_turn_cache_miss": 1.0,
            "final_turn_decision_seconds": 0.2,
            "final_turn_exact_scoring_time": 0.1,
            "final_turn_legal_action_generation_seconds": 0.01,
            "_final_turn_slow_states": [{"seconds": 0.2, "key_hash": "abc"}],
        }
    ]
    profile_path = tmp_path / "final_turn_profile.csv"
    slow_path = tmp_path / "slow.jsonl"
    cache_path = tmp_path / "cache.json"

    write_hu_turn2_final_turn_profile_csv(profiles, profile_path)
    write_hu_turn2_final_turn_slow_states(profiles, slow_path)
    write_hu_turn2_final_turn_cache_stats(
        profiles,
        cache_path,
        cache=FinalTurnDecisionCache(max_size=8),
        cache_enabled=True,
    )

    assert "final_turn_cache_hit_rate" in profile_path.read_text(encoding="utf-8").splitlines()[0]
    assert json.loads(slow_path.read_text(encoding="utf-8").splitlines()[0])["key_hash"] == "abc"
    assert json.loads(cache_path.read_text(encoding="utf-8"))["cache_enabled"] is True
