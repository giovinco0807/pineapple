import numpy as np

from ofc_regular.evaluate_hu_turn3_model import sample_rows, source_breakdown


class FixedModel:
    def predict_matrix(self, features):
        # Prefer the first action for the first sample and keep deterministic shape.
        return np.linspace(1.0, 0.0, features.shape[0])


def _sample(source="src"):
    return {
        "source": source,
        "source_input_path": f"outputs/{source}.jsonl",
        "sample_id": 1,
        "phase": "hu_turn3_9card",
        "seat": "first",
        "to_act_order": "first",
        "board": {
            "top": ["Qh"],
            "middle": ["Kh", "Kd", "6c", "8s"],
            "bottom": ["9c", "9d", "9s", "Kc"],
        },
        "opponent_board": {
            "top": ["2h", "3h", "4h"],
            "middle": ["2d", "3d", "4d", "5d", "6d"],
            "bottom": ["7c", "8c", "Tc", "Jc", "Qc"],
        },
        "dealt": ["Qs", "Ah", "7d"],
        "dead_cards": ["2c"],
        "actions": [
            {"placements": [["Qs", "top"], ["Ah", "middle"]], "discards": ["7d"], "score": 5.0, "original_index": 0},
            {"placements": [["Qs", "top"], ["7d", "middle"]], "discards": ["Ah"], "score": 1.0, "original_index": 1},
        ],
    }


def test_source_breakdown_groups_samples():
    metrics = source_breakdown(FixedModel(), [_sample("a"), _sample("b")], batch_samples=2)

    assert set(metrics) == {"a.jsonl", "b.jsonl"}
    assert metrics["a.jsonl"]["samples"] == 1.0


def test_sample_rows_reports_predicted_regret():
    rows = sample_rows(FixedModel(), [_sample("a")])

    assert rows[0]["source"] == "a"
    assert rows[0]["breakdown_source"] == "a.jsonl"
    assert rows[0]["predicted_original_index"] == 0
    assert rows[0]["regret"] == 0.0
