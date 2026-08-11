from ofc_regular.extract_hu_turn3_stage9b_hard_negatives import extract_hard_negative_rows


def test_extract_hard_negative_rows_keeps_only_bad_hu_overrides():
    sample = {
        "sample_id": 3,
        "source": "smoke",
        "source_state": {"state_id": "s1", "seed": 11, "hand_seed": 11},
        "seat": "first",
        "to_act_order": "first",
        "board": {"top": [], "middle": [], "bottom": []},
        "opponent_board": {"top": [], "middle": [], "bottom": []},
        "dealt": ["Ah", "Kd", "2c"],
        "dead_cards": ["3c"],
        "input_dead_cards": ["3c"],
        "best_score": 10.0,
        "future_count": 32,
        "future_samples": 32,
        "future_digest": "abc",
        "selection": {
            "baseline_index": 0,
            "hu_index": 1,
            "predicted_margin_vs_baseline": 2.0,
        },
        "actions": [
            {"original_index": 0, "score": 8.0, "placements": [], "discards": []},
            {"original_index": 1, "score": 6.0, "placements": [], "discards": []},
        ],
    }

    rows = extract_hard_negative_rows([sample], delta_threshold=-0.25)

    assert len(rows) == 1
    row = rows[0]
    assert row["schema"] == "hu_turn3_stage9b_hard_negative_v1"
    assert row["label"] == "hard_negative"
    assert row["state_id"] == "s1"
    assert row["delta_hu_vs_baseline"] == -2.0
    assert row["baseline_regret"] == 2.0
    assert row["hu_regret"] == 4.0


def test_extract_hard_negative_rows_skips_near_ties_and_positive_hu():
    base = {
        "selection": {"baseline_index": 0, "hu_index": 1},
        "best_score": 10.0,
        "actions": [
            {"original_index": 0, "score": 8.0},
            {"original_index": 1, "score": 7.9},
        ],
    }
    positive = {
        "selection": {"baseline_index": 0, "hu_index": 1},
        "best_score": 10.0,
        "actions": [
            {"original_index": 0, "score": 8.0},
            {"original_index": 1, "score": 9.0},
        ],
    }

    assert extract_hard_negative_rows([base, positive], delta_threshold=-0.25) == []
