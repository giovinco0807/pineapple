import json

from ofc_regular.analyze_hu_turn1_pilot import (
    read_records,
    refinement_candidate_rows,
    summarize,
)


def action(card: str, row: str, score: float, index: int = 0) -> dict:
    return {
        "placements": [[card, row]],
        "discards": ["2c"],
        "score": score,
        "ev": score,
        "se": 0.25,
        "action_index": index,
        "action_eval_seconds": 0.1,
    }


def test_summarize_hu_turn1_pilot_records() -> None:
    best = action("Ah", "top", 2.0, 1)
    other = action("Kd", "middle", 1.0, 2)
    records = [
        {
            "schema": "hu_turn1_stage1_pilot_v1",
            "sample_id": "a",
            "hand_seed": 1,
            "profile": "stage9f_fast_t2_t1_teacher",
            "opponent_profile": "stage9f_fast_t2_t1_teacher",
            "t2_continuation_profile": "stage9f_fast_t2_t1_teacher",
            "t3_continuation": "stage7_candidate_A_m5_r10",
            "seat": "first",
            "player": 0,
            "future_samples": 1,
            "action_count": 2,
            "actions": [best, other],
            "best_action": best,
            "score_gap": 1.0,
        },
        {
            "schema": "hu_turn1_stage1_pilot_v1",
            "sample_id": "b",
            "hand_seed": 2,
            "profile": "stage9f_fast_t2_t1_teacher",
            "opponent_profile": "stage9f_fast_t2_t1_teacher",
            "t2_continuation_profile": "stage9f_fast_t2_t1_teacher",
            "t3_continuation": "stage7_candidate_A_m5_r10",
            "seat": "second",
            "player": 1,
            "future_samples": 1,
            "action_count": 1,
            "actions": [action("Qs", "bottom", 0.5, 3)],
            "best_action": action("Qs", "bottom", 0.5, 3),
            "score_gap": 0.0,
        },
    ]

    summary = summarize(records)

    assert summary["records"] == 2
    assert summary["total_actions"] == 3
    assert summary["invalid_state_rows"] == 0
    assert summary["invalid_action_rows"] == 0
    assert summary["best_action_not_legal"] == 0
    assert summary["action_count_mismatches"] == 0
    assert summary["seat_counts"] == {"first": 1, "second": 1}
    assert summary["seat_distribution"]["single_seat_only"] is False
    assert summary["seat_distribution"]["balance_ratio"] == 1.0
    assert summary["seat_distribution"]["warning"] == ""
    assert summary["score_gap"]["max"] == 1.0
    assert summary["action_se"]["mean"] == 0.25
    assert summary["best_action_se"]["max"] == 0.25


def test_summarize_warns_for_single_seat_turn1_pilot() -> None:
    records = [
        {
            "sample_id": "first-only-a",
            "hand_seed": 1,
            "seat": "first",
            "player": 0,
            "future_samples": 32,
            "action_count": 1,
            "actions": [action("Ah", "top", 1.0, 1)],
            "best_action": action("Ah", "top", 1.0, 1),
            "score_gap": 0.0,
        },
        {
            "sample_id": "first-only-b",
            "hand_seed": 2,
            "seat": "first",
            "player": 0,
            "future_samples": 32,
            "action_count": 1,
            "actions": [action("Kh", "top", 1.0, 2)],
            "best_action": action("Kh", "top", 1.0, 2),
            "score_gap": 0.0,
        },
    ]

    summary = summarize(records)

    assert summary["seat_counts"] == {"first": 2}
    assert summary["seat_distribution"]["single_seat_only"] is True
    assert summary["seat_distribution"]["missing_expected_seats"] == ["second"]
    assert summary["seat_distribution"]["balance_ratio"] == 0.0
    assert "single_seat_only" in summary["seat_distribution"]["warning"]


def test_read_records_from_directory(tmp_path) -> None:
    path = tmp_path / "records"
    path.mkdir()
    (path / "a.jsonl").write_text(json.dumps({"sample_id": "a"}) + "\n", encoding="utf-8")
    (path / "b.jsonl").write_text(json.dumps({"sample_id": "b"}) + "\n", encoding="utf-8")

    assert [row["sample_id"] for row in read_records(path)] == ["a", "b"]


def test_refinement_candidate_rows_selects_close_or_high_se() -> None:
    close_best = action("Ah", "top", 2.0, 1)
    close_other = action("Kd", "middle", 1.8, 2)
    high_se_best = action("Qs", "bottom", 4.0, 3)
    high_se_best["se"] = 5.0
    records = [
        {
            "sample_id": "close",
            "hand_seed": 10,
            "player": 0,
            "seat": "first",
            "future_samples": 32,
            "action_count": 2,
            "actions": [close_best, close_other],
            "score_gap": 0.2,
        },
        {
            "sample_id": "high_se",
            "hand_seed": 11,
            "player": 1,
            "seat": "second",
            "future_samples": 32,
            "action_count": 2,
            "actions": [high_se_best, action("Js", "bottom", 0.0, 6)],
            "score_gap": 2.0,
        },
        {
            "sample_id": "stable",
            "hand_seed": 12,
            "player": 0,
            "seat": "first",
            "future_samples": 32,
            "action_count": 2,
            "actions": [action("2c", "top", 5.0, 4), action("3c", "middle", 1.0, 5)],
            "score_gap": 4.0,
        },
    ]

    rows = refinement_candidate_rows(
        records,
        close_gap_threshold=0.5,
        high_se_threshold=3.0,
        high_regret_threshold=1.0,
    )

    by_id = {row["sample_id"]: row for row in rows}
    assert set(by_id) == {"close", "high_se"}
    assert by_id["close"]["reasons"] == ["close_gap"]
    assert by_id["high_se"]["reasons"] == ["high_se"]
    assert by_id["close"]["top_action_indices"] == [0, 1]
