import copy

from ofc_regular.audit_hu_turn1_teacher import audit_records


def teacher_row() -> dict:
    return {
        "sample_id": 1,
        "hand_seed": 101,
        "player": 0,
        "seat": "first",
        "profile": "stage9f_p2",
        "opponent_profile": "stage9f_p2",
        "t2_continuation_profile": "stage9f_p2",
        "t3_continuation": "stage7_m5_r10",
        "future_samples": 32,
        "actions_truncated": False,
        "opponent_board": {"top": ["As"], "middle": ["Kh", "Qh"], "bottom": ["2c", "3c"]},
        "dead_cards": ["As", "Kh", "Qh", "2c", "3c"],
        "visible_dead_cards": ["As", "Kh", "Qh", "2c", "3c"],
        "true_dead_cards": [],
        "action_count": 2,
        "evaluated_action_count": 2,
        "total_legal_actions": 2,
        "best_action": 0,
        "actions": [
            {
                "placements": ["4d top", "5d middle"],
                "discards": ["6d"],
                "score": 1.0,
                "ev": 1.0,
                "se": 0.5,
                "rollout_count": 32,
                "action_index": 0,
                "original_index": 0,
            },
            {
                "placements": ["4d middle", "5d top"],
                "discards": ["6d"],
                "score": 0.0,
                "ev": 0.0,
                "se": 0.5,
                "rollout_count": 32,
                "action_index": 1,
                "original_index": 1,
            },
        ],
    }


def run_audit(rows: list[dict]) -> dict:
    return audit_records(
        rows,
        expected_records=len(rows),
        expected_seat="first",
        future_samples=32,
        profile="stage9f_p2",
        opponent_profile="stage9f_p2",
        t3_continuation="stage7_m5_r10",
    )


def test_complete_first_seat_teacher_passes():
    assert run_audit([teacher_row()])["status"] == "pass"


def test_hidden_discard_or_truncation_fails_closed():
    row = copy.deepcopy(teacher_row())
    row["dead_cards"].append("7s")
    row["actions_truncated"] = True

    result = run_audit([row])

    assert result["status"] == "fail"
    assert result["errors"]["dead_cards_visibility_mismatch"] == 1
    assert result["errors"]["actions_truncated"] == 1


def test_missing_action_index_or_non_finite_ev_fails_closed():
    row = copy.deepcopy(teacher_row())
    row["actions"][1]["original_index"] = 0
    row["actions"][1]["ev"] = float("nan")

    result = run_audit([row])

    assert result["status"] == "fail"
    assert result["errors"]["duplicate_action_index"] == 1
    assert result["errors"]["incomplete_action_index_set"] == 1
    assert result["errors"]["non_finite_action_value"] == 1
