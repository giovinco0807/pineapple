import math

import numpy as np

from ofc_regular.action_space import generate_turn_actions
from ofc_regular.hu_turn2_stage8_runtime import (
    HuTurn2Stage8RuntimeConfig,
    HuTurn2Stage8SelectiveOverridePolicy,
    sigmoid,
)
from ofc_regular.state import Board


class BaselineTurn2Model:
    def predict_sample(self, sample):
        values = np.zeros(len(sample["actions"]), dtype=np.float64)
        values[min(1, len(values) - 1)] = 1.0
        return values


class Stage8Model:
    def __init__(self, *, candidate_index=0, predicted_delta=3.0, gate_logit=10.0, nan=False):
        self.candidate_index = candidate_index
        self.predicted_delta = predicted_delta
        self.gate_logit = gate_logit
        self.nan = nan

    def predict_sample(self, sample):
        values = np.zeros((len(sample["actions"]), 5), dtype=np.float64)
        values[:, 4] = self.gate_logit
        values[self.candidate_index, 0] = 1.0
        values[self.candidate_index, 1] = self.predicted_delta
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


def make_policy(stage8_model, config, log):
    return HuTurn2Stage8SelectiveOverridePolicy(
        turn2_model=BaselineTurn2Model(),
        hu_turn2_stage8_model=stage8_model,
        hu_turn2_stage8_config=config,
        hu_turn2_decision_log=log,
        seed=1,
    )


def test_sigmoid_handles_positive_and_negative_values():
    assert sigmoid(10.0) > 0.99
    assert sigmoid(-10.0) < 0.01


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
