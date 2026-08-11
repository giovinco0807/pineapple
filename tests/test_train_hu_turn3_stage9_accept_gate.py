import json

from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.state import Board
from ofc_regular.train_hu_turn3_stage9_accept_gate import build_arrays, record_delta


def _record(sample_id: int, *, baseline_score: float, hu_score: float) -> dict:
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
    dealt = ("Qs", "Ah", "7d")
    observation = ActorObservation(
        hero_board=board,
        opponent_public_board=opponent,
        dealt_cards=dealt,
        hero_private_discards=("2c", "5c"),
        seat="first",
        street="T3",
        to_act_order="first",
    )
    actions = [
        {
            "original_index": 0,
            "score": baseline_score,
            "placements": [["Qs", "top"], ["Ah", "middle"]],
            "discards": ["7d"],
        },
        {
            "original_index": 1,
            "score": hu_score,
            "placements": [["Qs", "top"], ["7d", "middle"]],
            "discards": ["Ah"],
        },
    ]
    return {
        "sample_id": sample_id,
        "turn": "T3",
        "board": {"top": list(board.top), "middle": list(board.middle), "bottom": list(board.bottom)},
        "opponent_board": {"top": list(opponent.top), "middle": list(opponent.middle), "bottom": list(opponent.bottom)},
        "dealt": list(dealt),
        "dead_cards": list(observation.legacy_dead_cards()),
        "visible_dead_cards": list(observation.legacy_dead_cards()),
        "hero_private_discards": ["2c", "5c"],
        "policy_observation": observation.to_dict(),
        "seat": "first",
        "to_act_order": "first",
        "selection": {
            "baseline_index": 0,
            "hu_index": 1,
        },
        "runtime_decision": {
            "stage3_action": actions[0],
            "stage7_action": actions[1],
        },
        "actions": actions,
    }


def test_stage9_accept_gate_builds_arrays_from_joint_exact_jsonl(tmp_path):
    path = tmp_path / "joint.jsonl"
    records = [
        _record(1, baseline_score=1.0, hu_score=3.0),
        _record(2, baseline_score=3.0, hu_score=2.0),
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    X, y, deltas, weights = build_arrays([path], self_model=None)

    assert X.shape[0] == 2
    assert y.tolist() == [1, 0]
    assert deltas.tolist() == [2.0, -1.0]
    assert weights[1] > weights[0]


def test_stage9_accept_gate_can_use_runtime_hu_predictions(tmp_path):
    class DummyHuModel:
        def predict_sample(self, sample):
            values = [0.0] * len(sample["actions"])
            for index, action in enumerate(sample["actions"]):
                placements = {tuple(item) for item in action["placements"]}
                if ("Ah", "middle") in placements and ("Qs", "top") in placements:
                    values[index] = 10.0
                if ("7d", "middle") in placements and ("Qs", "top") in placements:
                    values[index] = 14.5
            return values

    path = tmp_path / "joint.jsonl"
    record = _record(1, baseline_score=1.0, hu_score=3.0)
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    X_joint, _y_joint, _deltas, _weights = build_arrays([path], self_model=None)
    X_runtime, y_runtime, deltas_runtime, _weights = build_arrays(
        [path],
        self_model=None,
        hu_model=DummyHuModel(),
    )

    assert X_joint[0, 0] == 2.0
    assert X_runtime[0, 0] == 4.5
    assert y_runtime.tolist() == [1]
    assert deltas_runtime.tolist() == [2.0]


def test_stage9_accept_gate_delta_uses_hu_minus_baseline():
    row = _record(1, baseline_score=4.0, hu_score=1.5)

    assert record_delta(row) == -2.5
