import json

from ofc_regular.analyze_hu_turn3_stage9_tail_veto import _build_rows


def test_stage9_tail_veto_uses_hu_index_delta(tmp_path):
    path = tmp_path / "joint.jsonl"
    record = {
        "sample_id": 7,
        "seat": "first",
        "board": {"top": ["As"], "middle": ["2c", "3c", "4c"], "bottom": ["9d", "Td", "Jd"]},
        "selection": {
            "baseline_index": 0,
            "hu_index": 2,
            "final_index": 1,
            "predicted_margin_vs_baseline": 3.0,
            "reference_margin": 1.0,
            "model_score": 9.0,
        },
        "actions": [
            {
                "original_index": 0,
                "score": 10.0,
                "placements": [["Ah", "top"], ["Kd", "middle"]],
            },
            {
                "original_index": 1,
                "score": 99.0,
                "placements": [["Ah", "middle"], ["Kd", "middle"]],
            },
            {
                "original_index": 2,
                "score": 12.5,
                "placements": [["Ah", "top"], ["Kd", "bottom"]],
            },
        ],
        "best_score": 12.5,
        "legal_action_count": 3,
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    rows = _build_rows(path)

    assert len(rows) == 1
    assert rows[0]["hu_index"] == 2
    assert rows[0]["delta_vs_baseline"] == 2.5
    assert rows[0]["hu_score"] == 12.5
