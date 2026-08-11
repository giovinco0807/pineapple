import numpy as np

from ofc_regular.action_space import generate_turn_actions
from ofc_regular.decision_trace import attach_replay_truth, capture_decision_log_positions
from ofc_regular.evaluate_hu_turn2_stage8b_topk_mc_rerank import (
    HuTurn2Stage8bTopKMcRerankPolicy,
    TopKMcRerankConfig,
)
from ofc_regular.hu_turn1_topk_confirm import (
    HuTurn1TopKConfirmConfig,
    HuTurn1TopKConfirmPolicy,
    parse_hu_turn1_topk_confirm_configs,
)
from ofc_regular.hu_infoset import ActorObservation, ReplayTruth
from ofc_regular.state import Board


class BaselineTurn1Model:
    def choose_action_index(self, sample):
        return min(1, len(sample["actions"]) - 1)

    def predict_sample(self, sample):
        values = np.zeros(len(sample["actions"]), dtype=np.float64)
        values[min(1, len(values) - 1)] = 1.0
        return values


class CandidateTurn1Model:
    def predict_sample(self, sample):
        values = np.zeros(len(sample["actions"]), dtype=np.float64)
        values[0] = 5.0
        return values


class IndexedCandidateTurn1Model:
    def __init__(self, index, score=5.0):
        self.index = index
        self.score = score

    def predict_sample(self, sample):
        values = np.zeros(len(sample["actions"]), dtype=np.float64)
        if self.index < len(values):
            values[self.index] = self.score
        return values


class AlwaysSafeEstimator:
    def predict_proba(self, features):
        return np.tile(np.asarray([[0.0, 1.0]], dtype=np.float64), (len(features), 1))


def test_hu_turn1_wrapper_preserves_parent_t2_future_seed():
    policy = HuTurn1TopKConfirmPolicy(
        turn1_model=BaselineTurn1Model(),
        hu_turn2_stage8b_model=None,
        topk_rerank_config=TopKMcRerankConfig(top_k=1, mc_samples=1, min_delta=0.0),
        hu_turn1_candidate_model=CandidateTurn1Model(),
        topk_confirm_config=HuTurn1TopKConfirmConfig(top_k=1, mc_samples=1, min_delta=0.0),
        seed=11,
        seat="second",
    )
    board = Board.from_rows(top=["Ah"], middle=["7c"], bottom=["Th", "Jc", "Tc"])
    opponent = Board.from_rows(top=["Ks"], middle=["Jh", "As"], bottom=["9c", "6d"])
    kwargs = {
        "board": board,
        "opponent_board": opponent,
        "dealt": ("Jd", "3d", "Js"),
        "decision_seed": 456,
        "hand_id": 123,
        "game_id": 123,
        "phase": "stage_a",
    }

    inherited_seed = policy._future_rollout_seed(**kwargs)
    expected_seed = HuTurn2Stage8bTopKMcRerankPolicy._future_rollout_seed(policy, **kwargs)
    t1_seed = policy._hu_turn1_future_rollout_seed(**kwargs)

    assert inherited_seed == expected_seed
    assert t1_seed != inherited_seed


def test_hu_turn1_topk_confirm_config_parses_candidate_union_options():
    config = parse_hu_turn1_topk_confirm_configs(
        "k25/mc8/d0/confirm16/cse1/pd0/seat=first+second/ctopk20/cap25/union=max_z_score"
    )[0]

    assert config.top_k == 25
    assert config.candidate_topk == 20
    assert config.candidate_union_cap == 25
    assert config.candidate_union_mode == "max_z_score"
    assert "ctopk20" in config.config_id
    assert "cap25" in config.config_id
    assert "union_max_z_score" in config.config_id


def test_hu_turn1_topk_confirm_config_parses_seat_specific_safe_thresholds():
    config = parse_hu_turn1_topk_confirm_configs(
        "k25/mc8/d0/confirm16/cse1/safe0.3/safefirst0.4/safesecond0.7/seat=first+second"
    )[0]

    assert config.safe_selector_threshold == 0.3
    assert config.safe_selector_threshold_first == 0.4
    assert config.safe_selector_threshold_second == 0.7
    assert config.safe_selector_threshold_for_seat("first") == 0.4
    assert config.safe_selector_threshold_for_seat("second") == 0.7
    assert "safefirst0.4" in config.config_id
    assert "safesecond0.7" in config.config_id


def test_stage18_p1_second_seat_keeps_stage9f_p2_fallback():
    board = Board.from_rows(top=["Ah"], middle=["7c"], bottom=["Th", "Jc", "Tc"])
    opponent = Board.from_rows(top=["Ks"], middle=["Jh", "As"], bottom=["9c", "6d"])
    dealt = ["Jd", "3d", "Js"]
    actions = generate_turn_actions(board, dealt)
    log_rows = []
    policy = HuTurn1TopKConfirmPolicy(
        turn1_model=BaselineTurn1Model(),
        hu_turn2_stage8b_model=None,
        topk_rerank_config=TopKMcRerankConfig(top_k=1, mc_samples=1, min_delta=0.0),
        hu_turn1_candidate_model=CandidateTurn1Model(),
        hu_turn1_safe_selector_model=object(),
        topk_confirm_config=parse_hu_turn1_topk_confirm_configs(
            "k5/mc8/d3/confirm32/cse1.5/pd1.5/seat=first/safe0.7"
        )[0],
        hu_turn1_decision_log=log_rows,
        seed=11,
        seat="second",
    )

    action = policy.choose_action(
        board,
        dealt,
        dead_cards=["Ks", "Jh", "As", "9c", "6d"],
        opponent_board=opponent,
        hand_id=123,
        game_id=123,
        decision_seed=456,
        street="T1",
    )

    assert action == actions[1]
    assert log_rows[0]["override_fired"] is False
    assert log_rows[0]["no_override_reason"] == "seat_not_allowed"


def test_hu_turn1_topk_confirm_uses_seat_specific_safe_threshold():
    board = Board.from_rows(
        top=["Ah"],
        middle=["7c"],
        bottom=["Th", "Jc", "Tc"],
    )
    opponent = Board.from_rows(
        top=["Ks"],
        middle=["Jh", "As"],
        bottom=["9c", "6d"],
    )
    dealt = ["Jd", "3d", "Js"]
    actions = generate_turn_actions(board, dealt)
    log_rows = []
    policy = HuTurn1TopKConfirmPolicy(
        turn1_model=BaselineTurn1Model(),
        hu_turn2_stage8b_model=None,
        topk_rerank_config=TopKMcRerankConfig(top_k=1, mc_samples=1, min_delta=0.0),
        hu_turn1_candidate_model=CandidateTurn1Model(),
        hu_turn1_safe_selector_model=object(),
        topk_confirm_config=HuTurn1TopKConfirmConfig(
            top_k=1,
            mc_samples=1,
            min_delta=0.0,
            safe_selector_threshold=0.3,
            safe_selector_threshold_first=0.4,
            safe_selector_threshold_second=0.7,
        ),
        hu_turn1_decision_log=log_rows,
        seed=11,
        seat="second",
    )

    def fake_rerank_sample(**kwargs):
        return {
            "evaluated_action_count": 2,
            "actions": [
                {"original_index": 0, "score": 5.0, "se": 0.0},
                {"original_index": 1, "score": 0.0, "se": 0.0},
            ],
        }

    policy._rerank_turn1_sample = fake_rerank_sample
    policy._safe_selector_score = lambda **_kwargs: 0.5

    action = policy.choose_action(
        board,
        dealt,
        dead_cards=["Ks", "Jh", "As", "9c", "6d"],
        opponent_board=opponent,
        hand_id=123,
        game_id=123,
        decision_seed=456,
        street="T1",
    )

    assert action == actions[1]
    assert len(log_rows) == 1
    record = log_rows[0]
    assert record["override_fired"] is False
    assert record["no_override_reason"] == "below_safe_selector"
    assert record["safe_selector_score"] == 0.5
    assert record["safe_selector_threshold"] == 0.7
    assert record["safe_selector_threshold_global"] == 0.3
    assert record["safe_selector_threshold_first"] == 0.4
    assert record["safe_selector_threshold_second"] == 0.7


def test_hu_turn1_safe_selector_receives_observation_and_can_fire():
    board = Board.from_rows(
        top=["Ah"],
        middle=["7c"],
        bottom=["Th", "Jc", "Tc"],
    )
    opponent = Board.from_rows(
        top=["Ks"],
        middle=["Jh", "As"],
        bottom=["9c", "6d"],
    )
    dealt = ("Jd", "3d", "Js")
    actions = generate_turn_actions(board, dealt)
    observation = ActorObservation(
        hero_board=board,
        opponent_public_board=opponent,
        dealt_cards=dealt,
        hero_private_discards=(),
        seat="first",
        street="T1",
        to_act_order="first",
    )
    log_rows = []
    policy = HuTurn1TopKConfirmPolicy(
        turn1_model=BaselineTurn1Model(),
        hu_turn2_stage8b_model=None,
        topk_rerank_config=TopKMcRerankConfig(top_k=1, mc_samples=1, min_delta=0.0),
        hu_turn1_candidate_model=CandidateTurn1Model(),
        hu_turn1_safe_selector_model={
            "model_kind": "hu_turn1_safe_override_selector_sklearn",
            "feature_mode": "meta_only",
            "estimator": AlwaysSafeEstimator(),
        },
        topk_confirm_config=HuTurn1TopKConfirmConfig(
            top_k=1,
            mc_samples=1,
            min_delta=0.0,
            safe_selector_threshold=0.5,
            allowed_seats=("first",),
        ),
        hu_turn1_decision_log=log_rows,
        seed=11,
        seat="first",
    )
    policy._rerank_turn1_sample = lambda **_kwargs: {
        "evaluated_action_count": 2,
        "actions": [
            {"original_index": 0, "score": 5.0, "se": 0.0},
            {"original_index": 1, "score": 0.0, "se": 0.0},
        ],
    }

    action = policy.choose_action_observation(
        observation,
        hand_id=123,
        game_id=123,
        decision_seed=456,
    )

    assert action == actions[0]
    assert len(log_rows) == 1
    record = log_rows[0]
    assert record["override_fired"] is True
    assert record["safe_selector_score"] == 1.0
    assert record["policy_observation"] == observation.to_dict()
    assert record["observation_fingerprint"] == observation.fingerprint()
    assert "opponent_private_discards" not in record["policy_observation"]


def test_stage18_p1_missing_safe_selector_keeps_stage9f_p2_fallback():
    board = Board.from_rows(top=["Ah"], middle=["7c"], bottom=["Th", "Jc", "Tc"])
    opponent = Board.from_rows(top=["Ks"], middle=["Jh", "As"], bottom=["9c", "6d"])
    dealt = ["Jd", "3d", "Js"]
    actions = generate_turn_actions(board, dealt)
    log_rows = []
    policy = HuTurn1TopKConfirmPolicy(
        turn1_model=BaselineTurn1Model(),
        hu_turn2_stage8b_model=None,
        topk_rerank_config=TopKMcRerankConfig(top_k=1, mc_samples=1, min_delta=0.0),
        hu_turn1_candidate_model=CandidateTurn1Model(),
        hu_turn1_safe_selector_model=None,
        topk_confirm_config=HuTurn1TopKConfirmConfig(
            top_k=1,
            mc_samples=1,
            min_delta=0.0,
            safe_selector_threshold=0.7,
            allowed_seats=("first",),
        ),
        hu_turn1_decision_log=log_rows,
        seed=11,
        seat="first",
    )

    policy._rerank_turn1_sample = lambda **_kwargs: {
        "evaluated_action_count": 2,
        "actions": [
            {"original_index": 0, "score": 5.0, "se": 0.0},
            {"original_index": 1, "score": 0.0, "se": 0.0},
        ],
    }

    action = policy.choose_action(
        board,
        dealt,
        dead_cards=["Ks", "Jh", "As", "9c", "6d"],
        opponent_board=opponent,
        hand_id=123,
        game_id=123,
        decision_seed=456,
        street="T1",
    )

    assert action == actions[1]
    assert log_rows[0]["override_fired"] is False
    assert log_rows[0]["no_override_reason"] == "safe_selector_unavailable"


def test_hu_turn1_topk_confirm_logs_safe_score_when_confirm_gate_blocks():
    board = Board.from_rows(
        top=["Ah"],
        middle=["7c"],
        bottom=["Th", "Jc", "Tc"],
    )
    opponent = Board.from_rows(
        top=["Ks"],
        middle=["Jh", "As"],
        bottom=["9c", "6d"],
    )
    dealt = ["Jd", "3d", "Js"]
    actions = generate_turn_actions(board, dealt)
    log_rows = []
    policy = HuTurn1TopKConfirmPolicy(
        turn1_model=BaselineTurn1Model(),
        hu_turn2_stage8b_model=None,
        topk_rerank_config=TopKMcRerankConfig(top_k=1, mc_samples=1, min_delta=0.0),
        hu_turn1_candidate_model=CandidateTurn1Model(),
        hu_turn1_safe_selector_model=object(),
        topk_confirm_config=HuTurn1TopKConfirmConfig(
            top_k=1,
            mc_samples=1,
            min_delta=0.0,
            confirm_mc_samples=16,
            confirm_se_multiplier=1.0,
            safe_selector_threshold=0.4,
        ),
        hu_turn1_decision_log=log_rows,
        seed=11,
        seat="second",
    )

    calls = []

    def fake_rerank_sample(**kwargs):
        calls.append(tuple(kwargs["action_indices"]))
        if "paired_delta_candidate_index" in kwargs:
            return {
                "paired_delta_mean": 0.5,
                "paired_delta_standard_error": 1.0,
                "paired_delta_count": 16,
            }
        return {
            "evaluated_action_count": 2,
            "actions": [
                {"original_index": 0, "score": 5.0, "se": 0.0},
                {"original_index": 1, "score": 0.0, "se": 0.0},
            ],
        }

    policy._rerank_turn1_sample = fake_rerank_sample
    policy._safe_selector_score = lambda **_kwargs: 0.8

    action = policy.choose_action(
        board,
        dealt,
        dead_cards=["Ks", "Jh", "As", "9c", "6d"],
        opponent_board=opponent,
        hand_id=123,
        game_id=123,
        decision_seed=456,
        street="T1",
    )

    assert action == actions[1]
    assert len(log_rows) == 1
    record = log_rows[0]
    assert record["override_fired"] is False
    assert record["no_override_reason"] == "below_confirm_se"
    assert record["safe_selector_score"] == 0.8
    assert record["safe_selector_threshold"] == 0.4
    assert calls == [(0, 1), (0, 1)]


def test_hu_turn1_topk_confirm_recomputes_safe_score_with_confirm_stats(monkeypatch):
    board = Board.from_rows(
        top=["Ah"],
        middle=["7c"],
        bottom=["Th", "Jc", "Tc"],
    )
    opponent = Board.from_rows(
        top=["Ks"],
        middle=["Jh", "As"],
        bottom=["9c", "6d"],
    )
    dealt = ["Jd", "3d", "Js"]
    actions = generate_turn_actions(board, dealt)
    log_rows = []
    policy = HuTurn1TopKConfirmPolicy(
        turn1_model=BaselineTurn1Model(),
        hu_turn2_stage8b_model=None,
        topk_rerank_config=TopKMcRerankConfig(top_k=1, mc_samples=1, min_delta=0.0),
        hu_turn1_candidate_model=CandidateTurn1Model(),
        hu_turn1_safe_selector_model=object(),
        topk_confirm_config=HuTurn1TopKConfirmConfig(
            top_k=1,
            mc_samples=1,
            min_delta=0.0,
            confirm_mc_samples=16,
            confirm_se_multiplier=1.0,
            safe_selector_threshold=0.8,
        ),
        hu_turn1_decision_log=log_rows,
        seed=11,
        seat="second",
    )

    def fake_rerank_sample(**kwargs):
        if "paired_delta_candidate_index" in kwargs:
            return {
                "paired_delta_mean": 4.0,
                "paired_delta_standard_error": 1.0,
                "paired_delta_count": 16,
            }
        return {
            "evaluated_action_count": 2,
            "actions": [
                {"original_index": 0, "score": 5.0, "se": 0.0},
                {"original_index": 1, "score": 0.0, "se": 0.0},
            ],
        }

    seen_rows = []

    def fake_safe_selector(_model, row):
        seen_rows.append(dict(row))
        return 0.9 if row.get("confirm_delta") == 4.0 else 0.1

    monkeypatch.setattr(
        "ofc_regular.hu_turn1_topk_confirm.score_hu_turn1_safe_selector",
        fake_safe_selector,
    )
    policy._rerank_turn1_sample = fake_rerank_sample

    action = policy.choose_action(
        board,
        dealt,
        dead_cards=["Ks", "Jh", "As", "9c", "6d"],
        opponent_board=opponent,
        hand_id=123,
        game_id=123,
        decision_seed=456,
        street="T1",
    )

    assert action == actions[0]
    assert len(log_rows) == 1
    record = log_rows[0]
    assert record["override_fired"] is True
    assert record["safe_selector_score"] == 0.9
    assert seen_rows[-1]["stage_a_delta"] == 5.0
    assert seen_rows[-1]["confirm_delta"] == 4.0
    assert seen_rows[-1]["confirm_delta_se"] == 1.0
    assert seen_rows[-1]["confirm_delta_count"] == 16


def test_hu_turn1_topk_confirm_log_marks_replay_ready_visibility_metadata():
    board = Board.from_rows(
        top=["Ah"],
        middle=["7c"],
        bottom=["Th", "Jc", "Tc"],
    )
    opponent = Board.from_rows(
        top=["Ks"],
        middle=["Jh", "As"],
        bottom=["9c", "6d"],
    )
    dealt = ["Jd", "3d", "Js"]
    actions = generate_turn_actions(board, dealt)
    log_rows = []
    policy = HuTurn1TopKConfirmPolicy(
        turn1_model=BaselineTurn1Model(),
        hu_turn2_stage8b_model=None,
        topk_rerank_config=TopKMcRerankConfig(top_k=1, mc_samples=1, min_delta=0.0),
        hu_turn1_candidate_model=CandidateTurn1Model(),
        topk_confirm_config=HuTurn1TopKConfirmConfig(top_k=1, mc_samples=1, min_delta=0.0),
        hu_turn1_decision_log=log_rows,
        seed=11,
        seat="first",
    )
    def fake_rerank_sample(**kwargs):
        return {
            "evaluated_action_count": 2,
            "actions": [
                {"original_index": 0, "score": 5.0, "se": 0.0},
                {"original_index": 1, "score": 0.0, "se": 0.0},
            ],
        }

    policy._rerank_turn1_sample = fake_rerank_sample

    positions = capture_decision_log_positions(policy)
    action = policy.choose_action(
        board,
        dealt,
        dead_cards=["Ks", "Jh", "As", "9c", "6d", "2c"],
        opponent_board=opponent,
        hand_id=123,
        game_id=123,
        decision_seed=456,
        street="T1",
    )
    assert log_rows[0]["replay_ready"] is False
    attach_replay_truth(
        positions,
        ReplayTruth(
            true_dead_cards=("2c", "3d"),
            visible_dead_cards=("Ks", "Jh", "As", "9c", "6d", "2c"),
            hero_private_discards=("2c",),
            opponent_private_discards=("3d",),
        ),
    )

    assert action == actions[0]
    assert len(log_rows) == 1
    record = log_rows[0]
    assert record["schema"] == "hu_turn1_topk_confirm_decision_v1"
    assert record["replay_ready"] is True
    assert record["visibility_model"] == "actor_observation_v1"
    assert record["discard_visibility"] == "own_private_only"
    assert record["true_dead_cards"] == ["2c", "3d"]
    assert record["visible_dead_cards"] == ["Ks", "Jh", "As", "9c", "6d", "2c"]
    assert record["hero_private_discards"] == ["2c"]
    assert "opponent_private_discards" not in record
    assert record["dead_cards"] == ["Ks", "Jh", "As", "9c", "6d", "2c"]
    assert record["true_hero_private_discards"] == ["2c"]
    assert record["true_opponent_private_discards"] == ["3d"]
    assert record["override_fired"] is True


def test_hu_turn1_topk_confirm_uses_multi_model_union_candidates():
    board = Board.from_rows(
        top=["Ah"],
        middle=["7c"],
        bottom=["Th", "Jc", "Tc"],
    )
    opponent = Board.from_rows(
        top=["Ks"],
        middle=["Jh", "As"],
        bottom=["9c", "6d"],
    )
    dealt = ["Jd", "3d", "Js"]
    actions = generate_turn_actions(board, dealt)
    log_rows = []
    policy = HuTurn1TopKConfirmPolicy(
        turn1_model=BaselineTurn1Model(),
        hu_turn2_stage8b_model=None,
        topk_rerank_config=TopKMcRerankConfig(top_k=1, mc_samples=1, min_delta=0.0),
        hu_turn1_candidate_model=None,
        hu_turn1_candidate_models=[
            IndexedCandidateTurn1Model(0, 5.0),
            IndexedCandidateTurn1Model(2, 6.0),
        ],
        topk_confirm_config=HuTurn1TopKConfirmConfig(
            top_k=2,
            mc_samples=1,
            min_delta=0.0,
            candidate_topk=1,
            candidate_union_cap=2,
            candidate_union_mode="max_z_score",
        ),
        hu_turn1_decision_log=log_rows,
        seed=11,
        seat="first",
    )

    seen_action_indices = []

    def fake_rerank_sample(**kwargs):
        seen_action_indices.append(tuple(kwargs["action_indices"]))
        return {
            "evaluated_action_count": len(kwargs["action_indices"]),
            "actions": [
                {"original_index": 2, "score": 6.0, "se": 0.0},
                {"original_index": 1, "score": 0.0, "se": 0.0},
                {"original_index": 0, "score": -1.0, "se": 0.0},
            ],
        }

    policy._rerank_turn1_sample = fake_rerank_sample

    action = policy.choose_action(
        board,
        dealt,
        dead_cards=["Ks", "Jh", "As", "9c", "6d"],
        opponent_board=opponent,
        hand_id=123,
        game_id=123,
        decision_seed=456,
        street="T1",
    )

    assert action == actions[2]
    assert seen_action_indices == [(0, 1, 2)]
    assert len(log_rows) == 1
    record = log_rows[0]
    assert record["candidate_model_count"] == 2
    assert record["candidate_union_mode"] == "max_z_score"
    assert record["candidate_topk"] == 1
    assert record["candidate_union_cap"] == 2
    assert record["candidate_union_size"] == 2
    assert set(record["selected_action_indices"]) == {0, 2}
    assert record["override_fired"] is True
