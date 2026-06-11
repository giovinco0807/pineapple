import numpy as np

from ofc_regular.play_ai import play_hand
from ofc_regular.policy import RegularAiPolicy, policy_sample
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
        policy_p1=RegularAiPolicy(seed=2),
    )
    assert result.board_p0.is_complete()
    assert result.board_p1.is_complete()
    assert set(result.board_p0.all_cards()).isdisjoint(result.board_p1.all_cards())
    reverse_score, _ = terminal_score(result.board_p1, result.board_p0)
    assert result.score_p0 == -reverse_score
