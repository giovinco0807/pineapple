import math

import numpy as np

from ofc_regular.action_space import generate_turn_actions
from ofc_regular.decision_trace import attach_replay_truth, capture_decision_log_positions
from ofc_regular.hu_infoset import ReplayTruth
from ofc_regular.hu_turn2_stage8_runtime import (
    HuTurn2Stage8RuntimeConfig,
    HuTurn2Stage8SelectiveOverridePolicy,
    sigmoid,
)
from ofc_regular.evaluate_hu_turn2_stage8_seat_swap import parse_configs
from ofc_regular.state import Board


class BaselineTurn2Model:
    def predict_sample(self, sample):
        values = np.zeros(len(sample["actions"]), dtype=np.float64)
        values[min(1, len(values) - 1)] = 1.0
        return values


class Stage8Model:
    def __init__(
        self,
        *,
        candidate_index=0,
        predicted_delta=3.0,
        predicted_ev=1.0,
        gate_logit=10.0,
        nan=False,
        ev_overrides=None,
    ):
        self.candidate_index = candidate_index
        self.predicted_delta = predicted_delta
        self.predicted_ev = predicted_ev
        self.gate_logit = gate_logit
        self.nan = nan
        self.ev_overrides = dict(ev_overrides or {})

    def predict_sample(self, sample):
        values = np.zeros((len(sample["actions"]), 5), dtype=np.float64)
        values[:, 4] = self.gate_logit
        values[self.candidate_index, 0] = self.predicted_ev
        values[self.candidate_index, 1] = self.predicted_delta
        for index, ev in self.ev_overrides.items():
            values[index, 0] = ev
        if self.nan:
            values[self.candidate_index, 1] = math.nan
        return values


def board_and_dealt():
    board = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd", "6c"],
        bottom=["9c", "9d", "9s"],
    )
    opponent = Board.from_rows(
        top=["2h"],
        middle=["3h", "4h", "5h"],
        bottom=["7h", "8h", "Th"],
    )
    dealt = ["Qs", "Ah", "7d"]
    return board, opponent, dealt


def make_policy(stage8_model, config, log, *, seat="first", context=None):
    return HuTurn2Stage8SelectiveOverridePolicy(
        turn2_model=BaselineTurn2Model(),
        hu_turn2_stage8_model=stage8_model,
        hu_turn2_stage8_config=config,
        hu_turn2_decision_log=log,
        hu_turn2_context=context,
        seed=1,
        seat=seat,
    )


def test_sigmoid_handles_positive_and_negative_values():
    assert sigmoid(10.0) > 0.99
    assert sigmoid(-10.0) < 0.01


def test_stage8_config_parser_accepts_seat_score_and_rank_guard():
    configs = parse_configs("2.75/0/0.95/seat=second/score=3/k=1")

    assert len(configs) == 1
    assert configs[0].min_margin == 2.75
    assert configs[0].gate_threshold == 0.95
    assert configs[0].allowed_seats == ("second",)
    assert configs[0].min_model_score == 3.0
    assert configs[0].candidate_ev_rank_max == 1
    assert configs[0].config_id == "m2.75_r0_g0.95_seatsecond_s3_k1"


def test_turn2_stage8_selective_override_fires_when_thresholds_pass():
    board, opponent, dealt = board_and_dealt()
    actions = generate_turn_actions(board, dealt)
    log = []
    policy = make_policy(
        Stage8Model(candidate_index=0, predicted_delta=3.0, gate_logit=10.0),
        HuTurn2Stage8RuntimeConfig(2.5, 0.0, 0.90),
        log,
    )

    action = policy.choose_action(board, dealt, opponent_board=opponent)

    assert action == actions[0]
    assert log[-1]["override_fired"] is True
    assert log[-1]["predicted_delta"] == 3.0
    assert log[-1]["gate_probability"] > 0.99
    assert log[-1]["dead_cards"] == []
    assert log[-1]["baseline_action_index"] == 1
    assert log[-1]["stage8_action_index"] == 0


def test_turn2_stage8_runtime_log_separates_true_and_visible_dead_cards():
    board, opponent, dealt = board_and_dealt()
    log = []
    policy = make_policy(
        Stage8Model(candidate_index=0, predicted_delta=3.0, gate_logit=10.0),
        HuTurn2Stage8RuntimeConfig(2.5, 0.0, 0.90),
        log,
        context={},
    )

    positions = capture_decision_log_positions(policy)
    policy.choose_action(
        board,
        dealt,
        dead_cards=[*opponent.all_cards(), "2c"],
        opponent_board=opponent,
    )
    assert log[-1]["replay_ready"] is False
    attach_replay_truth(
        positions,
        ReplayTruth(
            true_dead_cards=("2c", "3c"),
            visible_dead_cards=(*opponent.all_cards(), "2c"),
            hero_private_discards=("2c",),
            opponent_private_discards=("3c",),
        ),
    )

    record = log[-1]
    assert record["dead_cards"] == [*opponent.all_cards(), "2c"]
    assert record["visible_dead_cards"] == [*opponent.all_cards(), "2c"]
    assert "2c" in record["visible_dead_cards"]
    assert "3c" not in record["visible_dead_cards"]
    assert record["hero_private_discards"] == ["2c"]
    assert "opponent_private_discards" not in record
    assert record["true_dead_cards"] == ["2c", "3c"]
    assert record["true_hero_private_discards"] == ["2c"]
    assert record["true_opponent_private_discards"] == ["3c"]


def test_turn2_stage8_falls_back_below_margin():
    board, opponent, dealt = board_and_dealt()
    actions = generate_turn_actions(board, dealt)
    log = []
    policy = make_policy(
        Stage8Model(candidate_index=0, predicted_delta=1.0, gate_logit=10.0),
        HuTurn2Stage8RuntimeConfig(2.5, 0.0, 0.90),
        log,
    )

    action = policy.choose_action(board, dealt, opponent_board=opponent)

    assert action == actions[1]
    assert log[-1]["override_fired"] is False
    assert log[-1]["no_override_reason"] == "below_stage8_margin"


def test_turn2_stage8_falls_back_below_model_score():
    board, opponent, dealt = board_and_dealt()
    actions = generate_turn_actions(board, dealt)
    log = []
    policy = make_policy(
        Stage8Model(candidate_index=0, predicted_delta=9.0, predicted_ev=0.5, gate_logit=10.0),
        HuTurn2Stage8RuntimeConfig(2.5, 0.0, 0.90, min_model_score=1.0),
        log,
    )

    action = policy.choose_action(board, dealt, opponent_board=opponent)

    assert action == actions[1]
    assert log[-1]["override_fired"] is False
    assert log[-1]["no_override_reason"] == "below_model_score"
    assert log[-1]["hu_turn2_min_model_score"] == 1.0


def test_turn2_stage8_falls_back_when_seat_not_allowed():
    board, opponent, dealt = board_and_dealt()
    actions = generate_turn_actions(board, dealt)
    log = []
    policy = make_policy(
        Stage8Model(candidate_index=0, predicted_delta=9.0, gate_logit=10.0),
        HuTurn2Stage8RuntimeConfig(2.5, 0.0, 0.90, allowed_seats=("second",)),
        log,
        seat="first",
    )

    action = policy.choose_action(board, dealt, opponent_board=opponent)

    assert action == actions[1]
    assert log[-1]["override_fired"] is False
    assert log[-1]["no_override_reason"] == "seat_not_allowed"
    assert log[-1]["hu_turn2_allowed_seats"] == ["second"]


def test_turn2_stage8_falls_back_below_candidate_ev_rank():
    board, opponent, dealt = board_and_dealt()
    actions = generate_turn_actions(board, dealt)
    log = []
    policy = make_policy(
        Stage8Model(candidate_index=0, predicted_delta=9.0, predicted_ev=1.0, gate_logit=10.0, ev_overrides={2: 2.0}),
        HuTurn2Stage8RuntimeConfig(2.5, 0.0, 0.90, candidate_ev_rank_max=1),
        log,
    )

    action = policy.choose_action(board, dealt, opponent_board=opponent)

    assert action == actions[1]
    assert log[-1]["override_fired"] is False
    assert log[-1]["no_override_reason"] == "below_candidate_ev_rank"
    assert log[-1]["candidate_ev_rank"] == 2
    assert log[-1]["hu_turn2_candidate_ev_rank_max"] == 1


def test_turn2_stage8_disabled_matches_baseline():
    board, opponent, dealt = board_and_dealt()
    actions = generate_turn_actions(board, dealt)
    log = []
    policy = make_policy(
        Stage8Model(candidate_index=0, predicted_delta=9.0, gate_logit=10.0),
        HuTurn2Stage8RuntimeConfig(0.0, 0.0, 0.0, enabled=False),
        log,
    )

    action = policy.choose_action(board, dealt, opponent_board=opponent)

    assert action == actions[1]
    assert log[-1]["no_override_reason"] == "stage8_disabled"


def test_turn2_stage8_nan_prediction_falls_back():
    board, opponent, dealt = board_and_dealt()
    actions = generate_turn_actions(board, dealt)
    log = []
    policy = make_policy(
        Stage8Model(candidate_index=0, predicted_delta=9.0, gate_logit=10.0, nan=True),
        HuTurn2Stage8RuntimeConfig(2.5, 0.0, 0.90),
        log,
    )

    action = policy.choose_action(board, dealt, opponent_board=opponent)

    assert action == actions[1]
    assert log[-1]["override_fired"] is False
    assert log[-1]["no_override_reason"] == "nan_prediction"
