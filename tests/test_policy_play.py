import json

import numpy as np

from ofc_regular import evaluate_matchups
from ofc_regular.decision_trace import attach_replay_truth, capture_decision_log_positions
from ofc_regular.hu_infoset import ReplayTruth
from ofc_regular.play_ai import play_hand
from ofc_regular.evaluate_matchups import evaluate_matchup, trace_hand
from ofc_regular.policy import RegularAiPolicy, action_to_json, policy_sample
from ofc_regular.state import Board
from ofc_regular.action_space import generate_actions, generate_turn_actions
from ofc_regular.teacher import terminal_score
from ofc_regular.turn3_model import train_ridge_model


def _turn3_training_sample():
    board = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd", "6c", "8s"],
        bottom=["9c", "9d", "9s", "Kc"],
    )
    actions = generate_turn_actions(board, ["Qs", "Ah", "7d"])
    sample = policy_sample(board, ["Qs", "Ah", "7d"], actions)
    for idx, action in enumerate(sample["actions"]):
        action["score"] = 10.0 if ("Qs", "top") in action["placements"] else float(idx) * 0.01
    return sample


def test_policy_sample_builds_turn3_model_input():
    sample = _turn3_training_sample()
    assert sample["phase"] == "turn3_9card"
    assert sample["actions"]
    assert "next_board" in sample["actions"][0]


def test_regular_ai_policy_uses_turn3_model_path():
    sample = _turn3_training_sample()
    model = train_ridge_model([sample, sample], l2=1.0)
    board = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd", "6c", "8s"],
        bottom=["9c", "9d", "9s", "Kc"],
    )
    policy = RegularAiPolicy(turn3_model=model, seed=1)
    action = policy.choose_action(board, ["Qs", "Ah", "7d"])
    assert action.placements


def test_regular_ai_policy_prefers_hu_turn3_model_with_opponent_board():
    class HuTurn3Model:
        def __init__(self) -> None:
            self.calls = 0
            self.last_sample = None

        def choose_action_index(self, sample):
            self.calls += 1
            self.last_sample = sample
            return len(sample["actions"]) - 1

    board = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd", "6c", "8s"],
        bottom=["9c", "9d", "9s", "Kc"],
    )
    opponent = Board.from_rows(
        top=["2h"],
        middle=["3h", "4h", "5h", "6h"],
        bottom=["7h", "8h", "Th", "Jh"],
    )
    model = HuTurn3Model()
    policy = RegularAiPolicy(hu_turn3_model=model, seed=1, seat="second")

    action = policy.choose_action(
        board,
        ["Qs", "Ah", "7d"],
        dead_cards=opponent.all_cards(),
        opponent_board=opponent,
    )

    assert action.placements
    assert model.calls == 1
    assert model.last_sample["schema"] == "hu_stage1"
    assert model.last_sample["seat"] == "second"
    assert model.last_sample["opponent_board"]["top"] == ["2h"]


def test_regular_ai_policy_hu_turn3_margin_can_fallback_to_self_board_model():
    class HuTurn3Model:
        def predict_sample(self, sample):
            predictions = np.zeros(len(sample["actions"]), dtype=np.float64)
            predictions[-1] = 1.0
            return predictions

    class SelfBoardTurn3Model:
        def choose_action_index(self, sample):
            return 0

    board = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd", "6c", "8s"],
        bottom=["9c", "9d", "9s", "Kc"],
    )
    opponent = Board.from_rows(
        top=["2h"],
        middle=["3h", "4h", "5h", "6h"],
        bottom=["7h", "8h", "Th", "Jh"],
    )
    actions = generate_turn_actions(board, ["Qs", "Ah", "7d"])
    policy = RegularAiPolicy(
        turn3_model=SelfBoardTurn3Model(),
        hu_turn3_model=HuTurn3Model(),
        hu_turn3_min_margin=2.0,
        seed=1,
    )

    action = policy.choose_action(
        board,
        ["Qs", "Ah", "7d"],
        dead_cards=opponent.all_cards(),
        opponent_board=opponent,
    )

    assert action == actions[0]


def test_regular_ai_policy_hu_turn3_reference_model_can_be_fallback_action():
    class CandidateHuModel:
        def predict_sample(self, sample):
            predictions = np.zeros(len(sample["actions"]), dtype=np.float64)
            predictions[-1] = 10.0
            predictions[1] = 8.0
            return predictions

    class ReferenceHuModel:
        def predict_sample(self, sample):
            predictions = np.zeros(len(sample["actions"]), dtype=np.float64)
            predictions[1] = 12.0
            return predictions

    class SelfBoardTurn3Model:
        def choose_action_index(self, sample):
            return 0

    board = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd", "6c", "8s"],
        bottom=["9c", "9d", "9s", "Kc"],
    )
    opponent = Board.from_rows(
        top=["2h"],
        middle=["3h", "4h", "5h", "6h"],
        bottom=["7h", "8h", "Th", "Jh"],
    )
    actions = generate_turn_actions(board, ["Qs", "Ah", "7d"])
    policy = RegularAiPolicy(
        turn3_model=SelfBoardTurn3Model(),
        hu_turn3_model=CandidateHuModel(),
        hu_turn3_reference_model=ReferenceHuModel(),
        hu_turn3_reference_min_margin=10.0,
        hu_turn3_min_margin=5.0,
        seed=1,
    )

    action = policy.choose_action(
        board,
        ["Qs", "Ah", "7d"],
        dead_cards=opponent.all_cards(),
        opponent_board=opponent,
    )

    assert action == actions[1]


def test_regular_ai_policy_hu_turn3_self_regret_can_fallback_to_self_board_model():
    class HuTurn3Model:
        def predict_sample(self, sample):
            predictions = np.zeros(len(sample["actions"]), dtype=np.float64)
            predictions[-1] = 10.0
            return predictions

    class SelfBoardTurn3Model:
        def choose_action_index(self, sample):
            return 0

        def predict_sample(self, sample):
            predictions = np.zeros(len(sample["actions"]), dtype=np.float64)
            predictions[0] = 5.0
            predictions[-1] = 1.0
            return predictions

    board = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd", "6c", "8s"],
        bottom=["9c", "9d", "9s", "Kc"],
    )
    opponent = Board.from_rows(
        top=["2h"],
        middle=["3h", "4h", "5h", "6h"],
        bottom=["7h", "8h", "Th", "Jh"],
    )
    actions = generate_turn_actions(board, ["Qs", "Ah", "7d"])
    policy = RegularAiPolicy(
        turn3_model=SelfBoardTurn3Model(),
        hu_turn3_model=HuTurn3Model(),
        hu_turn3_min_margin=2.0,
        hu_turn3_max_self_regret=1.0,
        seed=1,
    )

    action = policy.choose_action(
        board,
        ["Qs", "Ah", "7d"],
        dead_cards=opponent.all_cards(),
        opponent_board=opponent,
    )

    assert action == actions[0]


def test_regular_ai_policy_prefers_hu_turn1_model_with_opponent_board():
    class HuTurn1Model:
        def __init__(self) -> None:
            self.calls = 0
            self.last_sample = None

        def choose_action_index(self, sample):
            self.calls += 1
            self.last_sample = sample
            return len(sample["actions"]) - 1

    board = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd"],
        bottom=["9c", "9d"],
    )
    opponent = Board.from_rows(
        top=["2h"],
        middle=["3h", "4h"],
        bottom=["7h", "8h"],
    )
    actions = generate_turn_actions(board, ["Qs", "Ah", "7d"])
    model = HuTurn1Model()
    policy = RegularAiPolicy(hu_turn1_model=model, seed=1, seat="second")

    action = policy.choose_action(
        board,
        ["Qs", "Ah", "7d"],
        dead_cards=opponent.all_cards(),
        opponent_board=opponent,
    )

    assert action == actions[-1]
    assert model.calls == 1
    assert model.last_sample["schema"] == "hu_stage1"
    assert model.last_sample["seat"] == "second"
    assert model.last_sample["opponent_board"]["top"] == ["2h"]


def test_regular_ai_policy_hu_turn1_invalid_index_falls_back_to_self_board_model():
    class HuTurn1Model:
        def choose_action_index(self, sample):
            return len(sample["actions"]) + 10

    class SelfBoardTurn1Model:
        def choose_action_index(self, sample):
            return 0

    board = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd"],
        bottom=["9c", "9d"],
    )
    opponent = Board.from_rows(
        top=["2h"],
        middle=["3h", "4h"],
        bottom=["7h", "8h"],
    )
    actions = generate_turn_actions(board, ["Qs", "Ah", "7d"])
    policy = RegularAiPolicy(
        hu_turn1_model=HuTurn1Model(),
        turn1_model=SelfBoardTurn1Model(),
        seed=1,
    )

    action = policy.choose_action(
        board,
        ["Qs", "Ah", "7d"],
        dead_cards=opponent.all_cards(),
        opponent_board=opponent,
    )

    assert action == actions[0]


def test_regular_ai_policy_hu_turn1_margin_can_fallback_to_self_board_model():
    class HuTurn1Model:
        def predict_sample(self, sample):
            predictions = np.zeros(len(sample["actions"]), dtype=np.float64)
            predictions[-1] = 1.0
            return predictions

    class SelfBoardTurn1Model:
        def choose_action_index(self, sample):
            return 0

    board = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd"],
        bottom=["9c", "9d"],
    )
    opponent = Board.from_rows(
        top=["2h"],
        middle=["3h", "4h"],
        bottom=["7h", "8h"],
    )
    actions = generate_turn_actions(board, ["Qs", "Ah", "7d"])
    log_records = []
    policy = RegularAiPolicy(
        hu_turn1_model=HuTurn1Model(),
        hu_turn1_min_margin=2.0,
        turn1_model=SelfBoardTurn1Model(),
        hu_turn1_decision_log=log_records,
        seed=1,
    )

    action = policy.choose_action(
        board,
        ["Qs", "Ah", "7d"],
        dead_cards=opponent.all_cards(),
        opponent_board=opponent,
    )

    assert action == actions[0]


def test_regular_ai_policy_hu_turn1_margin_can_override_self_board_model():
    class HuTurn1Model:
        def predict_sample(self, sample):
            predictions = np.zeros(len(sample["actions"]), dtype=np.float64)
            predictions[-1] = 3.0
            return predictions

    class SelfBoardTurn1Model:
        def choose_action_index(self, sample):
            return 0

    board = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd"],
        bottom=["9c", "9d"],
    )
    opponent = Board.from_rows(
        top=["2h"],
        middle=["3h", "4h"],
        bottom=["7h", "8h"],
    )
    actions = generate_turn_actions(board, ["Qs", "Ah", "7d"])
    log_records = []
    policy = RegularAiPolicy(
        hu_turn1_model=HuTurn1Model(),
        hu_turn1_min_margin=2.0,
        turn1_model=SelfBoardTurn1Model(),
        hu_turn1_decision_log=log_records,
        seed=1,
    )

    action = policy.choose_action(
        board,
        ["Qs", "Ah", "7d"],
        dead_cards=opponent.all_cards(),
        opponent_board=opponent,
    )

    assert action == actions[-1]
    assert log_records[-1]["override_fired"] is True
    assert log_records[-1]["hu_turn1_predicted_margin"] == 3.0
    assert log_records[-1]["baseline_action"] == action_to_json(board, actions[0])
    assert log_records[-1]["final_action"] == action_to_json(board, actions[-1])
    assert log_records[-1]["dead_cards"] == list(opponent.all_cards())


def test_regular_ai_policy_hu_turn3_support_model_can_fallback_to_self_board_model():
    class HuTurn3Model:
        def predict_sample(self, sample):
            predictions = np.zeros(len(sample["actions"]), dtype=np.float64)
            predictions[-1] = 10.0
            return predictions

    class HuSupportModel:
        def predict_sample(self, sample):
            predictions = np.zeros(len(sample["actions"]), dtype=np.float64)
            predictions[0] = 5.0
            predictions[-1] = 6.0
            return predictions

    class SelfBoardTurn3Model:
        def choose_action_index(self, sample):
            return 0

    board = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd", "6c", "8s"],
        bottom=["9c", "9d", "9s", "Kc"],
    )
    opponent = Board.from_rows(
        top=["2h"],
        middle=["3h", "4h", "5h", "6h"],
        bottom=["7h", "8h", "Th", "Jh"],
    )
    actions = generate_turn_actions(board, ["Qs", "Ah", "7d"])
    policy = RegularAiPolicy(
        turn3_model=SelfBoardTurn3Model(),
        hu_turn3_model=HuTurn3Model(),
        hu_turn3_support_model=HuSupportModel(),
        hu_turn3_min_margin=2.0,
        hu_turn3_min_support_margin=2.0,
        seed=1,
    )

    action = policy.choose_action(
        board,
        ["Qs", "Ah", "7d"],
        dead_cards=opponent.all_cards(),
        opponent_board=opponent,
    )

    assert action == actions[0]


def test_regular_ai_policy_hu_turn3_model_score_floor_falls_back_to_stage3():
    class HuTurn3Model:
        def predict_sample(self, sample):
            predictions = np.zeros(len(sample["actions"]), dtype=np.float64)
            predictions[0] = 2.0
            predictions[-1] = 5.0
            return predictions

    class SelfBoardTurn3Model:
        def choose_action_index(self, sample):
            return 0

    board = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd", "6c", "8s"],
        bottom=["9c", "9d", "9s", "Kc"],
    )
    opponent = Board.from_rows(
        top=["2h"],
        middle=["3h", "4h", "5h", "6h"],
        bottom=["7h", "8h", "Th", "Jh"],
    )
    actions = generate_turn_actions(board, ["Qs", "Ah", "7d"])
    log_records = []
    policy = RegularAiPolicy(
        turn3_model=SelfBoardTurn3Model(),
        hu_turn3_model=HuTurn3Model(),
        hu_turn3_min_margin=2.0,
        hu_turn3_min_model_score=10.0,
        hu_turn3_decision_log=log_records,
        seed=1,
    )

    action = policy.choose_action(
        board,
        ["Qs", "Ah", "7d"],
        dead_cards=opponent.all_cards(),
        opponent_board=opponent,
    )

    assert action == actions[0]
    assert log_records[-1]["no_override_reason"] == "below_model_score"
    assert log_records[-1]["hu_turn3_min_model_score"] == 10.0


def test_regular_ai_policy_hu_turn3_allowed_seats_falls_back_when_seat_blocked():
    class HuTurn3Model:
        def __init__(self):
            self.calls = 0

        def predict_sample(self, sample):
            self.calls += 1
            predictions = np.zeros(len(sample["actions"]), dtype=np.float64)
            predictions[-1] = 10.0
            return predictions

    class SelfBoardTurn3Model:
        def choose_action_index(self, sample):
            return 0

    board = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd", "6c", "8s"],
        bottom=["9c", "9d", "9s", "Kc"],
    )
    opponent = Board.from_rows(
        top=["2h"],
        middle=["3h", "4h", "5h", "6h"],
        bottom=["7h", "8h", "Th", "Jh"],
    )
    actions = generate_turn_actions(board, ["Qs", "Ah", "7d"])
    hu_model = HuTurn3Model()
    log_records = []
    policy = RegularAiPolicy(
        turn3_model=SelfBoardTurn3Model(),
        hu_turn3_model=hu_model,
        hu_turn3_min_margin=2.0,
        hu_turn3_allowed_seats=("first",),
        hu_turn3_decision_log=log_records,
        seat="second",
        seed=1,
    )

    action = policy.choose_action(
        board,
        ["Qs", "Ah", "7d"],
        dead_cards=opponent.all_cards(),
        opponent_board=opponent,
    )

    assert action == actions[0]
    assert hu_model.calls == 0
    assert log_records[-1]["no_override_reason"] == "seat_not_allowed"
    assert log_records[-1]["hu_turn3_allowed_seats"] == ["first"]


def test_regular_ai_policy_hu_turn3_gate_model_can_fallback_to_self_board_model():
    class HuTurn3Model:
        def predict_sample(self, sample):
            predictions = np.zeros(len(sample["actions"]), dtype=np.float64)
            predictions[-1] = 10.0
            return predictions

    class SelfBoardTurn3Model:
        def choose_action_index(self, sample):
            return 0

        def predict_sample(self, sample):
            predictions = np.zeros(len(sample["actions"]), dtype=np.float64)
            predictions[0] = 2.0
            predictions[-1] = 1.0
            return predictions

    class RejectGateModel:
        def __init__(self):
            self.calls = 0

        def predict_accept_probability(self, sample, *, chosen_index, baseline_index, hu_predictions, self_predictions):
            self.calls += 1
            assert chosen_index == len(sample["actions"]) - 1
            assert baseline_index == 0
            return 0.25

    board = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd", "6c", "8s"],
        bottom=["9c", "9d", "9s", "Kc"],
    )
    opponent = Board.from_rows(
        top=["2h"],
        middle=["3h", "4h", "5h", "6h"],
        bottom=["7h", "8h", "Th", "Jh"],
    )
    actions = generate_turn_actions(board, ["Qs", "Ah", "7d"])
    gate = RejectGateModel()
    policy = RegularAiPolicy(
        turn3_model=SelfBoardTurn3Model(),
        hu_turn3_model=HuTurn3Model(),
        hu_turn3_gate_model=gate,
        hu_turn3_min_margin=2.0,
        hu_turn3_min_gate_probability=0.5,
        seed=1,
    )

    action = policy.choose_action(
        board,
        ["Qs", "Ah", "7d"],
        dead_cards=opponent.all_cards(),
        opponent_board=opponent,
    )

    assert action == actions[0]
    assert gate.calls == 1


def test_regular_ai_policy_stage7_off_matches_stage3_baseline():
    class HuTurn3Model:
        def __init__(self):
            self.calls = 0

        def predict_sample(self, sample):
            self.calls += 1
            predictions = np.zeros(len(sample["actions"]), dtype=np.float64)
            predictions[-1] = 99.0
            return predictions

    class SelfBoardTurn3Model:
        def choose_action_index(self, sample):
            return 0

    board = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd", "6c", "8s"],
        bottom=["9c", "9d", "9s", "Kc"],
    )
    opponent = Board.from_rows(
        top=["2h"],
        middle=["3h", "4h", "5h", "6h"],
        bottom=["7h", "8h", "Th", "Jh"],
    )
    actions = generate_turn_actions(board, ["Qs", "Ah", "7d"])
    stage7 = HuTurn3Model()
    log_records = []
    policy = RegularAiPolicy(
        turn3_model=SelfBoardTurn3Model(),
        hu_turn3_model=stage7,
        hu_turn3_stage7_enabled=False,
        hu_turn3_decision_log=log_records,
        seed=1,
    )

    action = policy.choose_action(
        board,
        ["Qs", "Ah", "7d"],
        dead_cards=opponent.all_cards(),
        opponent_board=opponent,
        hand_id="h1",
        game_id="g1",
        decision_seed=123,
        street="T3",
    )

    assert action == actions[0]
    assert stage7.calls == 0
    assert log_records[-1]["no_override_reason"] == "stage7_disabled"
    assert log_records[-1]["final_action"] == log_records[-1]["stage3_action"]
    assert log_records[-1]["dead_cards"] == list(opponent.all_cards())


def test_regular_ai_policy_stage7_off_keeps_reference_stage3_action():
    class HuTurn3Model:
        def __init__(self):
            self.calls = 0

        def predict_sample(self, sample):
            self.calls += 1
            predictions = np.zeros(len(sample["actions"]), dtype=np.float64)
            predictions[-1] = 99.0
            return predictions

    class ReferenceHuModel:
        def predict_sample(self, sample):
            predictions = np.zeros(len(sample["actions"]), dtype=np.float64)
            predictions[1] = 12.0
            return predictions

    class SelfBoardTurn3Model:
        def choose_action_index(self, sample):
            return 0

    board = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd", "6c", "8s"],
        bottom=["9c", "9d", "9s", "Kc"],
    )
    opponent = Board.from_rows(
        top=["2h"],
        middle=["3h", "4h", "5h", "6h"],
        bottom=["7h", "8h", "Th", "Jh"],
    )
    actions = generate_turn_actions(board, ["Qs", "Ah", "7d"])
    stage7 = HuTurn3Model()
    log_records = []
    policy = RegularAiPolicy(
        turn3_model=SelfBoardTurn3Model(),
        hu_turn3_model=stage7,
        hu_turn3_reference_model=ReferenceHuModel(),
        hu_turn3_stage7_enabled=False,
        hu_turn3_reference_min_margin=10.0,
        hu_turn3_decision_log=log_records,
        seed=1,
    )

    action = policy.choose_action(
        board,
        ["Qs", "Ah", "7d"],
        dead_cards=opponent.all_cards(),
        opponent_board=opponent,
    )

    assert action == actions[1]
    assert stage7.calls == 0
    assert log_records[-1]["no_override_reason"] == "stage7_disabled"
    assert log_records[-1]["reference_margin"] == 12.0


def test_regular_ai_policy_stage7_model_missing_falls_back_to_stage3():
    class SelfBoardTurn3Model:
        def choose_action_index(self, sample):
            return 0

    board = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd", "6c", "8s"],
        bottom=["9c", "9d", "9s", "Kc"],
    )
    opponent = Board.from_rows(
        top=["2h"],
        middle=["3h", "4h", "5h", "6h"],
        bottom=["7h", "8h", "Th", "Jh"],
    )
    actions = generate_turn_actions(board, ["Qs", "Ah", "7d"])
    policy = RegularAiPolicy(
        turn3_model=SelfBoardTurn3Model(),
        hu_turn3_model=None,
        hu_turn3_min_margin=5.0,
        hu_turn3_reference_min_margin=10.0,
        seed=1,
    )

    action = policy.choose_action(
        board,
        ["Qs", "Ah", "7d"],
        dead_cards=opponent.all_cards(),
        opponent_board=opponent,
    )

    assert action == actions[0]


def test_regular_ai_policy_stage7_illegal_candidate_falls_back_to_stage3():
    class HuTurn3Model:
        def predict_sample(self, sample):
            predictions = np.zeros(len(sample["actions"]) + 1, dtype=np.float64)
            predictions[-1] = 99.0
            return predictions

    class SelfBoardTurn3Model:
        def choose_action_index(self, sample):
            return 0

    board = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd", "6c", "8s"],
        bottom=["9c", "9d", "9s", "Kc"],
    )
    opponent = Board.from_rows(
        top=["2h"],
        middle=["3h", "4h", "5h", "6h"],
        bottom=["7h", "8h", "Th", "Jh"],
    )
    actions = generate_turn_actions(board, ["Qs", "Ah", "7d"])
    log_records = []
    policy = RegularAiPolicy(
        turn3_model=SelfBoardTurn3Model(),
        hu_turn3_model=HuTurn3Model(),
        hu_turn3_min_margin=5.0,
        hu_turn3_reference_min_margin=0.0,
        hu_turn3_decision_log=log_records,
        seed=1,
    )

    action = policy.choose_action(
        board,
        ["Qs", "Ah", "7d"],
        dead_cards=opponent.all_cards(),
        opponent_board=opponent,
    )

    assert action == actions[0]
    assert log_records[-1]["no_override_reason"] == "illegal_candidate"


def test_regular_ai_policy_stage7_nan_prediction_falls_back_to_stage3():
    class HuTurn3Model:
        def predict_sample(self, sample):
            predictions = np.zeros(len(sample["actions"]), dtype=np.float64)
            predictions[-1] = np.nan
            return predictions

    class SelfBoardTurn3Model:
        def choose_action_index(self, sample):
            return 0

    board = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd", "6c", "8s"],
        bottom=["9c", "9d", "9s", "Kc"],
    )
    opponent = Board.from_rows(
        top=["2h"],
        middle=["3h", "4h", "5h", "6h"],
        bottom=["7h", "8h", "Th", "Jh"],
    )
    actions = generate_turn_actions(board, ["Qs", "Ah", "7d"])
    log_records = []
    policy = RegularAiPolicy(
        turn3_model=SelfBoardTurn3Model(),
        hu_turn3_model=HuTurn3Model(),
        hu_turn3_min_margin=5.0,
        hu_turn3_reference_min_margin=0.0,
        hu_turn3_decision_log=log_records,
        seed=1,
    )

    action = policy.choose_action(
        board,
        ["Qs", "Ah", "7d"],
        dead_cards=opponent.all_cards(),
        opponent_board=opponent,
    )

    assert action == actions[0]
    assert log_records[-1]["no_override_reason"] == "nan_prediction"


def test_regular_ai_policy_uses_hu_exact_when_final_opponent_is_complete():
    board = Board.from_rows(
        top=["6d", "9d", "Qc"],
        middle=["7s", "As", "Ts"],
        bottom=["Ah", "4c", "Ks", "Jh", "6c"],
    )
    opponent = Board.from_rows(
        top=["Qs", "Jc", "9s"],
        middle=["3d", "4h", "5h", "Qh", "Kd"],
        bottom=["Jd", "3c", "2d", "9h", "8s"],
    )
    policy = RegularAiPolicy(seed=1)

    action = policy.choose_action(
        board,
        ["8d", "Th", "4s"],
        dead_cards=opponent.all_cards(),
        opponent_board=opponent,
    )

    assert ("4s", "middle") in action.placements
    assert action.discards == ("Th",)


def test_regular_ai_policy_uses_turn2_model_path():
    sample = _turn3_training_sample()
    model = train_ridge_model([sample, sample], l2=1.0)
    board = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd", "6c"],
        bottom=["9c", "9d", "9s"],
    )
    policy = RegularAiPolicy(turn2_model=model, seed=1)
    action = policy.choose_action(board, ["Qs", "Ah", "7d"])
    assert action.placements
    assert board.place(action.placements).card_count() == 9


def test_regular_ai_policy_uses_turn1_model_path():
    sample = _turn3_training_sample()
    model = train_ridge_model([sample, sample], l2=1.0)
    board = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd"],
        bottom=["9c", "9d"],
    )
    policy = RegularAiPolicy(turn1_model=model, seed=1)
    action = policy.choose_action(board, ["Qs", "Ah", "7d"])
    assert action.placements
    assert board.place(action.placements).card_count() == 7


def test_regular_ai_policy_uses_opening_model_path():
    sample = _turn3_training_sample()
    model = train_ridge_model([sample, sample], l2=1.0)
    board = Board.from_rows()
    policy = RegularAiPolicy(opening_model=model, seed=1)
    action = policy.choose_action(board, ["Qs", "Ah", "7d", "2c", "3c"])
    assert len(action.placements) == 5
    assert board.place(action.placements).card_count() == 5


def test_regular_ai_policy_prefers_opening_model_over_turn1_lookahead():
    class OpeningModel:
        def __init__(self) -> None:
            self.calls = 0

        def choose_action_index(self, sample):
            self.calls += 1
            return len(sample["actions"]) - 1

    class Turn1Model:
        def __init__(self) -> None:
            self.calls = 0

        def predict_sample(self, sample):
            self.calls += 1
            return np.zeros(len(sample["actions"]), dtype=np.float64)

    board = Board.from_rows()
    dealt = ["Qs", "Ah", "7d", "2c", "3c"]
    opening_model = OpeningModel()
    turn1_model = Turn1Model()
    policy = RegularAiPolicy(
        opening_model=opening_model,
        turn1_model=turn1_model,
        opening_lookahead_samples=2,
        seed=1,
    )

    action = policy.choose_action(board, dealt)

    assert len(action.placements) == 5
    assert opening_model.calls == 1
    assert turn1_model.calls == 0


def test_regular_ai_policy_opening_lookahead_enumerates_all_t0_actions():
    class CountingTurn1Model:
        def __init__(self) -> None:
            self.calls = 0

        def predict_sample(self, sample):
            self.calls += 1
            return np.zeros(len(sample["actions"]), dtype=np.float64)

    board = Board.from_rows()
    dealt = ["Qs", "Ah", "7d", "2c", "3c"]
    legal_openings = generate_actions(board, dealt)
    model = CountingTurn1Model()
    policy = RegularAiPolicy(
        turn1_model=model,
        opening_lookahead_samples=2,
        seed=1,
    )

    action = policy.choose_action(board, dealt, dead_cards=["4h", "5h", "6h"])

    assert len(action.placements) == 5
    assert board.place(action.placements).card_count() == 5
    assert model.calls == len(legal_openings) * 2


def test_play_hand_completes_two_boards_and_scores_hu():
    result = play_hand(
        seed=7,
        policy_p0=RegularAiPolicy(seed=1),
        policy_p1=RegularAiPolicy(seed=2, seat="second"),
    )
    assert result.board_p0.is_complete()
    assert result.board_p1.is_complete()
    assert set(result.board_p0.all_cards()).isdisjoint(result.board_p1.all_cards())
    reverse_score, _ = terminal_score(result.board_p1, result.board_p0)
    assert result.score_p0 == -reverse_score


class RecordingFirstActionPolicy:
    def __init__(self) -> None:
        self.calls = []

    def choose_action(
        self,
        board,
        dealt_cards,
        *,
        dead_cards=(),
        opponent_board=None,
        decision_seed=None,
        street=None,
        **_kwargs,
    ):
        action = generate_actions(board, dealt_cards)[0]
        self.calls.append(
            {
                "board_count": board.card_count(),
                "dealt": tuple(dealt_cards),
                "dead_cards": tuple(dead_cards),
                "opponent_board": opponent_board,
                "decision_seed": decision_seed,
                "street": street,
                "action": action,
            }
        )
        return action


def _placed_cards(*calls):
    return [card for call in calls for card, _row in call["action"].placements]


def _assert_hidden_opponent_discards(policy_p0, policy_p1):
    p0_t0, p0_t1, p0_t2 = policy_p0.calls[:3]
    p1_t0, p1_t1, _p1_t2 = policy_p1.calls[:3]

    p0_t1_discard = p0_t1["action"].discards[0]
    p1_t1_discard = p1_t1["action"].discards[0]

    assert p0_t1_discard not in p1_t1["dead_cards"]
    assert p1_t1_discard not in p0_t2["dead_cards"]
    assert p0_t1_discard in p0_t2["dead_cards"]
    assert set(_placed_cards(p0_t0, p0_t1)).issubset(set(p1_t1["dead_cards"]))
    assert set(_placed_cards(p1_t0, p1_t1)).issubset(set(p0_t2["dead_cards"]))


def test_play_hand_dead_cards_hide_opponent_private_discards():
    policy_p0 = RecordingFirstActionPolicy()
    policy_p1 = RecordingFirstActionPolicy()

    play_hand(seed=7, policy_p0=policy_p0, policy_p1=policy_p1)

    _assert_hidden_opponent_discards(policy_p0, policy_p1)


def test_matchup_trace_dead_cards_hide_opponent_private_discards():
    policy_p0 = RecordingFirstActionPolicy()
    policy_p1 = RecordingFirstActionPolicy()

    trace_hand(
        seed=7,
        profile_p0="a",
        profile_p1="b",
        policy_p0=policy_p0,
        policy_p1=policy_p1,
    )

    _assert_hidden_opponent_discards(policy_p0, policy_p1)


def test_play_and_trace_domain_separate_policy_rng_by_actor_and_street():
    play_p0 = RecordingFirstActionPolicy()
    play_p1 = RecordingFirstActionPolicy()
    play_hand(seed=37, policy_p0=play_p0, policy_p1=play_p1)
    play_coordinates = [
        (call["street"], call["decision_seed"])
        for call in (*play_p0.calls, *play_p1.calls)
    ]

    trace_p0 = RecordingFirstActionPolicy()
    trace_p1 = RecordingFirstActionPolicy()
    trace_hand(
        seed=37,
        profile_p0="a",
        profile_p1="b",
        policy_p0=trace_p0,
        policy_p1=trace_p1,
    )
    trace_coordinates = [
        (call["street"], call["decision_seed"])
        for call in (*trace_p0.calls, *trace_p1.calls)
    ]

    assert play_coordinates == trace_coordinates
    assert len({seed for _street, seed in play_coordinates}) == 10
    assert {street for street, _seed in play_coordinates} == {
        "T0",
        "T1",
        "T2",
        "T3",
        "T4",
    }


def test_evaluate_matchup_writes_topk_decision_log(monkeypatch, tmp_path):
    class LoggingPolicy(RegularAiPolicy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.topk_decision_log = None

        def choose_action(self, board, dealt_cards, **kwargs):
            actions = generate_actions(board, dealt_cards)
            if self.topk_decision_log is not None and board.card_count() == 7:
                chosen = actions[0]
                self.topk_decision_log.append(
                    {
                        "seat": self.seat,
                        "seed": kwargs.get("decision_seed"),
                        "street": kwargs.get("street"),
                        "override_fired": False,
                        "no_override_reason": "test_log",
                        "baseline_action": action_to_json(board, chosen),
                        "final_action": action_to_json(board, chosen),
                    }
                )
            return actions[0]

    def fake_build_policy(profile, bundle, *, seed, seat, opening_lookahead_samples):
        return LoggingPolicy(seed=seed, seat=seat, opening_lookahead_samples=opening_lookahead_samples)

    monkeypatch.setattr(evaluate_matchups, "build_policy", fake_build_policy)

    output = tmp_path / "topk_decisions.jsonl"
    summary = evaluate_matchup(
        profile_a="stage9f_cse1p5_firstseat",
        profile_b="stage7_m5_r10",
        games=2,
        seed=17,
        bundle=object(),
        opening_lookahead_samples=1,
        topk_decision_output=output,
    )

    lines = output.read_text(encoding="utf-8").splitlines()
    assert summary["topk_decision_output"] == str(output)
    assert summary["topk_decisions_written"] == 8
    assert summary["topk_realized_delta_count"] == 8
    assert summary["topk_realized_override_count"] == 0
    assert summary["topk_non_fired_nonzero_count"] == 0
    assert len(lines) == 8
    assert all('"no_override_reason":"test_log"' in line for line in lines)
    records = [json.loads(line) for line in lines]
    assert {record["seat_swap"] for record in records} == {"ab", "ba"}
    assert all(record["realized_delta_valid"] is True for record in records)
    assert all(
        record["realized_delta_basis"]
        == "same_snapshot_nonfire_trajectory_identity_v1"
        for record in records
    )


def test_evaluate_matchup_writes_hu_turn1_decision_log(monkeypatch, tmp_path):
    class LoggingPolicy(RegularAiPolicy):
        def choose_action(self, board, dealt_cards, **kwargs):
            actions = generate_actions(board, dealt_cards)
            if self.hu_turn1_decision_log is not None and board.card_count() == 5:
                chosen = actions[0]
                self.hu_turn1_decision_log.append(
                    {
                        "seat": self.seat,
                        "seed": kwargs.get("decision_seed"),
                        "street": kwargs.get("street"),
                        "override_fired": False,
                        "no_override_reason": "test_t1_log",
                        "baseline_action": action_to_json(board, chosen),
                        "final_action": action_to_json(board, chosen),
                    }
                )
            return actions[0]

    def fake_build_policy(profile, bundle, *, seed, seat, opening_lookahead_samples):
        return LoggingPolicy(seed=seed, seat=seat, opening_lookahead_samples=opening_lookahead_samples)

    monkeypatch.setattr(evaluate_matchups, "build_policy", fake_build_policy)

    output = tmp_path / "hu_turn1_decisions.jsonl"
    summary = evaluate_matchup(
        profile_a="stage9f_p2_hu_t1_stage1",
        profile_b="stage9f_p2",
        games=2,
        seed=17,
        bundle=object(),
        opening_lookahead_samples=1,
        hu_turn1_decision_output=output,
    )

    lines = output.read_text(encoding="utf-8").splitlines()
    assert summary["hu_turn1_decision_output"] == str(output)
    assert summary["hu_turn1_decisions_written"] == 8
    assert summary["hu_turn1_realized_delta_count"] == 8
    assert summary["hu_turn1_realized_override_count"] == 0
    assert summary["hu_turn1_non_fired_nonzero_count"] == 0
    assert len(lines) == 8
    records = [json.loads(line) for line in lines]
    assert all(record["no_override_reason"] == "test_t1_log" for record in records)
    assert {record["seat_swap"] for record in records} == {"ab", "ba"}
    assert all(record["realized_delta_valid"] is True for record in records)
    assert all(
        record["realized_delta_basis"]
        == "same_snapshot_nonfire_trajectory_identity_v1"
        for record in records
    )


def test_evaluate_matchup_uses_same_policy_seed_for_same_seat_across_swap(monkeypatch):
    calls = []

    def fake_build_policy(profile, bundle, *, seed, seat, opening_lookahead_samples):
        calls.append((profile, seat, seed))
        return RecordingFirstActionPolicy()

    monkeypatch.setattr(evaluate_matchups, "build_policy", fake_build_policy)

    summary = evaluate_matchup(
        profile_a="stage9f_p2_hu_t1_stage1",
        profile_b="stage9f_p2",
        games=1,
        seed=17,
        bundle=object(),
        opening_lookahead_samples=1,
    )

    assert calls == [
        ("stage9f_p2_hu_t1_stage1", "first", 68),
        ("stage9f_p2", "second", 69),
        ("stage9f_p2_hu_t1_stage1", "first", 68),
        ("stage9f_p2", "second", 69),
        ("stage9f_p2", "first", 68),
        ("stage9f_p2_hu_t1_stage1", "second", 69),
        ("stage9f_p2", "first", 68),
        ("stage9f_p2_hu_t1_stage1", "second", 69),
    ]
    assert summary["policy_seed_pairing"] == "seat_stable"


def test_hu_turn1_decision_log_includes_replay_context():
    class BaselineModel:
        def choose_action_index(self, sample):
            return 0

    class HuTurn1Model:
        def predict_sample(self, sample):
            predictions = np.zeros(len(sample["actions"]), dtype=np.float64)
            predictions[1] = 2.0
            return predictions

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
    log_rows = []
    policy = RegularAiPolicy(
        turn1_model=BaselineModel(),
        hu_turn1_model=HuTurn1Model(),
        hu_turn1_min_margin=1.0,
        hu_turn1_decision_log=log_rows,
        seed=11,
    )
    positions = capture_decision_log_positions(policy)
    policy.choose_action(
        board,
        ["Jd", "3d", "Js"],
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

    assert len(log_rows) == 1
    record = log_rows[0]
    assert record["replay_ready"] is True
    assert record["visibility_model"] == "actor_observation_v1"
    assert record["dead_cards"] == ["Ks", "Jh", "As", "9c", "6d", "2c"]
    assert record["true_dead_cards"] == ["2c", "3d"]
    assert record["visible_dead_cards"] == ["Ks", "Jh", "As", "9c", "6d", "2c"]
    assert record["hero_private_discards"] == ["2c"]
    assert "opponent_private_discards" not in record
    assert record["true_hero_private_discards"] == ["2c"]
    assert record["true_opponent_private_discards"] == ["3d"]
