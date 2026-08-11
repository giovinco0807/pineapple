import json

from ofc_regular.analyze_hu_turn1_decision_log import analyze_file


def test_analyze_hu_turn1_decision_log_extracts_hard_negatives(tmp_path):
    input_path = tmp_path / "decisions.jsonl"
    rows = [
        {
            "override_fired": True,
            "realized_delta_valid": True,
            "realized_candidate_seat_delta": -6.0,
            "hu_turn1_predicted_margin": 1.1,
            "seat": "first",
            "no_override_reason": "",
        },
        {
            "override_fired": True,
            "realized_delta_valid": True,
            "realized_candidate_seat_delta": 4.0,
            "hu_turn1_predicted_margin": 2.1,
            "seat": "second",
            "no_override_reason": "",
        },
        {
            "override_fired": False,
            "realized_delta_valid": True,
            "realized_candidate_seat_delta": 0.0,
            "hu_turn1_predicted_margin": 0.4,
            "seat": "first",
            "no_override_reason": "below_hu_turn1_margin",
        },
    ]
    input_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "analysis"
    summary = analyze_file(input_path, output_dir)

    assert summary["rows"] == 3
    assert summary["valid_rows"] == 3
    assert summary["fired_valid_rows"] == 2
    assert summary["fired_delta"]["mean"] == -1.0
    assert summary["non_fired_nonzero_count"] == 0
    assert summary["fired_labeled_rows_written"] == 2
    assert summary["positive_rows_written"] == 1
    assert summary["neutral_rows_written"] == 0
    assert summary["hard_negative_rows_written"] == 1
    assert (output_dir / "hu_turn1_margin_buckets.csv").exists()
    positive_lines = (
        output_dir / "hu_turn1_positive_targets.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(positive_lines) == 1
    assert json.loads(positive_lines[0])["hu_turn1_safe_override_label"] == "positive"
    hard_negative_lines = (
        output_dir / "hu_turn1_hard_negative_targets.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(hard_negative_lines) == 1
    assert json.loads(hard_negative_lines[0])["realized_candidate_seat_delta"] == -6.0
    assert json.loads(hard_negative_lines[0])["hu_turn1_safe_override_label_id"] == 0
