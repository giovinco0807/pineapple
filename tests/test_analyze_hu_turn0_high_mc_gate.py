from ofc_regular.action_space import generate_actions
from ofc_regular.analyze_hu_turn0_high_mc_gate import (
    choose_recommended_gate,
    decision_rows_for_model,
    sweep_gate_thresholds,
    sweep_seat_thresholds,
)
from ofc_regular.policy import action_to_json, board_to_json
from ofc_regular.state import Board


class FixedModel:
    def predict_sample(self, sample):
        return [0.0, 2.0, 0.5]


def _row(index: int, *, gain: float = 1.0):
    board = Board.from_rows()
    opponent = Board.from_rows()
    dealt = ["As", "Kh", "Qd", "Jc", "Ts"]
    actions = generate_actions(board, dealt)[:3]
    payloads = []
    for action_index, action in enumerate(actions):
        payload = action_to_json(board, action)
        payload.update(
            {
                "action_index": action_index,
                "original_index": action_index,
                "delta_vs_baseline": gain if action_index == 1 else 0.0,
                "delta_se_vs_baseline": 0.1,
            }
        )
        payloads.append(payload)
    return {
        "dataset_state_id": f"state-{index}",
        "sample_id": index,
        "hand_seed": 1000 + index,
        "seat": "first" if index % 2 == 0 else "second",
        "to_act_order": "first" if index % 2 == 0 else "second",
        "board": board_to_json(board),
        "opponent_board": board_to_json(opponent),
        "visible_dead_cards": [],
        "dealt": dealt,
        "actions": payloads,
        "baseline_action_index": 0,
        "best_action_index": 1,
        "delta_best_vs_baseline": gain,
    }


def test_high_mc_gate_analyzer_uses_teacher_delta_only_for_evaluation():
    rows = [_row(index, gain=1.0) for index in range(20)]
    decisions = decision_rows_for_model(rows, FixedModel(), model_name="fixed")

    assert len(decisions) == 20
    assert all(row["candidate_original_index"] == 1 for row in decisions)
    assert all(row["predicted_delta"] == 2.0 for row in decisions)
    assert all(row["teacher_delta"] == 1.0 for row in decisions)

    sweep = sweep_gate_thresholds(decisions, [0.0, 2.0, 2.1])
    assert sweep[0]["fires"] == 20
    assert sweep[1]["fires"] == 20
    assert sweep[2]["fires"] == 0
    assert sweep[1]["avg_gain"] == 1.0
    assert sweep[1]["false_positive_rate"] == 0.0
    assert sweep[1]["first_fires"] == 10
    assert sweep[1]["second_fires"] == 10
    assert choose_recommended_gate(sweep)["threshold"] in {0.0, 2.0}


def test_high_mc_gate_analyzer_rejects_false_positive_candidate():
    rows = [_row(index, gain=(-1.0 if index < 10 else 1.0)) for index in range(20)]
    decisions = decision_rows_for_model(rows, FixedModel(), model_name="fixed")
    sweep = sweep_gate_thresholds(decisions, [0.0])

    assert sweep[0]["false_positive_rate"] == 0.5
    assert choose_recommended_gate(sweep) is None


def test_seat_gate_sweep_can_disable_a_bad_seat():
    rows = [
        _row(index, gain=(1.0 if index % 2 == 0 else -1.0))
        for index in range(20)
    ]
    decisions = decision_rows_for_model(rows, FixedModel(), model_name="fixed")

    sweep = sweep_seat_thresholds(decisions, [2.0])
    recommendation = choose_recommended_gate(sweep)

    assert recommendation is not None
    assert recommendation["allowed_seats"] == "first"
    assert recommendation["first_fires"] == 10
    assert recommendation["second_fires"] == 0
