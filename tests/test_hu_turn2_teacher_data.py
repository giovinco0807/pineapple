import numpy as np

import json
import hashlib

from ofc_regular.final_turn_decision_cache import (
    FinalTurnDecisionCache,
    decide_final_turn_exact,
    final_turn_canonical_key,
)
from ofc_regular.hu_turn2_teacher_data import (
    _candidate_pool_record,
    _cheap_no_rollout_proxy,
    _passes_no_rollout_pool,
    _passes_source_bucket,
    build_hu_turn2_sample,
    evaluate_hu_turn2_actions,
    write_hu_turn2_final_turn_cache_stats,
    write_hu_turn2_final_turn_profile_csv,
    write_hu_turn2_final_turn_slow_states,
)
from ofc_regular.policy import RegularAiPolicy
from ofc_regular.state import Board


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


def test_evaluate_hu_turn2_actions_writes_common_future_teacher_fields():
    sample = evaluate_hu_turn2_actions(
        board=_hero_t2_board(),
        dealt_cards=["Qs", "Ah", "7d"],
        opponent_board=_opponent_t2_board(),
        dead_cards=["2c"],
        hero_seat="first",
        continuation_policy=RegularAiPolicy(seed=1),
        opponent_policy=RegularAiPolicy(seed=2),
        baseline_turn2_model=FirstActionModel(),
        future_samples=2,
        future_rollout_seed=99,
    )

    assert sample is not None
    assert sample["schema"] == "hu_turn2_stage1"
    assert sample["phase"] == "hu_turn2_7card"
    assert sample["turn"] == "T2"
    assert sample["common_random_futures"] is True
    assert sample["future_rollout_seed"] == 99
    assert len(sample["common_random_future_digest"]) == 64
    assert sample["continuation_policy_T3"] == "Stage7_candidate_A_m5_r10"
    assert sample["actions"]
    assert sample["legal_actions"] == sample["actions"]
    assert all(action["rollout_count"] == 2 for action in sample["actions"])
    assert all(action["common_random_future_digest"] == sample["common_random_future_digest"] for action in sample["actions"])
    assert all("ev_standard_error" in action for action in sample["actions"])
    assert "delta_best_vs_baseline" in sample
    assert "SE_delta_best_vs_baseline" in sample
    assert sample["profiling"]["seconds_total"] > 0.0
    assert "continuation_decision_seconds" in sample["profiling"]
    assert "t3_stage7_override_rate" in sample["downstream_features"]


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
        dead_cards=["2c"],
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
            dead_cards=["2c"],
            hero_seat="first",
            continuation_policy=RegularAiPolicy(seed=1),
            opponent_policy=RegularAiPolicy(seed=2),
            baseline_turn2_model=FirstActionModel(),
            future_samples=2,
            future_rollout_seed=99,
            use_batched_continuation=True,
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
