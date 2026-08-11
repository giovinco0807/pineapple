import json

from ofc_regular.prepare_hu_turn3_stage9_tail_veto_dataset import build_dataset


def _record(sample_id: int, *, baseline_score: float, hu_score: float) -> dict:
    return {
        "sample_id": sample_id,
        "seat": "first",
        "board": {"top": ["As"], "middle": ["2c", "3c", "4c"], "bottom": ["9d", "Td", "Jd"]},
        "source_state": {"state_id": f"s{sample_id}", "seed": 1000 + sample_id},
        "selection": {
            "baseline_index": 0,
            "hu_index": 1,
            "predicted_margin_vs_baseline": 2.0,
            "reference_margin": 0.5,
            "model_score": 10.0,
        },
        "actions": [
            {
                "original_index": 0,
                "score": baseline_score,
                "placements": [["Ah", "top"], ["Kd", "middle"]],
                "discards": ["7c"],
            },
            {
                "original_index": 1,
                "score": hu_score,
                "placements": [["Ah", "top"], ["Kd", "bottom"]],
                "discards": ["Qs"],
            },
        ],
        "best_score": max(baseline_score, hu_score),
        "legal_action_count": 2,
    }


def test_tail_veto_dataset_labels_safe_gray_and_hard_negative(tmp_path):
    path = tmp_path / "override_joint_exact_mc128_a.jsonl"
    records = [
        _record(1, baseline_score=10.0, hu_score=10.5),
        _record(2, baseline_score=10.0, hu_score=9.9),
        _record(3, baseline_score=10.0, hu_score=9.0),
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    rows = build_dataset([path], positive_threshold=0.25, negative_threshold=-0.25)

    assert [row["tail_label"] for row in rows] == ["safe_positive", "gray", "hard_negative"]
    assert [row["tail_label_id"] for row in rows] == [2, 1, 0]
    assert rows[0]["safe_positive"] is True
    assert rows[1]["gray"] is True
    assert rows[2]["hard_negative"] is True
    assert rows[2]["label_weight"] > rows[0]["label_weight"]
    assert rows[0]["source_name"] == str(tmp_path.name)
    assert rows[0]["dataset_row_id"] == 0
    assert rows[0]["hu_places_top_count"] == 1
    assert rows[0]["hu_places_bottom_count"] == 1
    assert rows[0]["baseline_places_middle_count"] == 1
    assert rows[0]["hu_discard_rank_max"] == 12
    assert rows[0]["baseline_discard_rank_max"] == 7
    assert rows[0]["hu_discards_higher_rank_than_baseline"] is True
    assert rows[0]["hu_placed_rank_sum"] == 27
