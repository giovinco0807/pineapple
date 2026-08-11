import numpy as np

from copy import deepcopy

from ofc_regular.evaluate_matchups import (
    _augment_profile_topk_counterfactuals,
    _final_action_match_status,
    classify_board,
    summarize_scores,
)
from ofc_regular.state import Board


def test_classify_board_marks_fl_chase_bust():
    board = Board.from_rows(
        top=["Qh", "Qs", "2c"],
        middle=["Ah", "Kd", "7c", "5s", "3d"],
        bottom=["9h", "8h", "6h", "4h", "2h"],
    )

    assert classify_board(board) == "FL狙いバースト"


def test_classify_board_marks_fl_success():
    board = Board.from_rows(
        top=["Qh", "Qs", "2c"],
        middle=["Ah", "Ad", "7c", "5s", "3d"],
        bottom=["9h", "9d", "9s", "4h", "4d"],
    )

    assert classify_board(board) == "FL成功"


def test_summarize_scores_ci_is_finite():
    summary = summarize_scores([1.0, -1.0, 3.0, 5.0])

    assert summary["avg_score_per_hand_for_a"] == 2.0
    assert np.isfinite(summary["std_error"])
    assert summary["ci95_low"] < summary["ci95_high"]


def test_nonfire_counterfactual_requires_same_action_and_full_trajectory_identity():
    action = {
        "placements": [["Ah", "top"], ["Ks", "middle"]],
        "discards": ["2c"],
    }
    hand = {
        "seed": 7,
        "profiles": {"p0": "candidate", "p1": "opponent"},
        "score_p0": 3.0,
        "turns": [
            {
                "turn": "T2",
                "player": 0,
                "action_key": "rak1:0000000001000:0000000000800:0000000000000:0000000000001",
                "placements": action["placements"],
                "discards": action["discards"],
                "board": {"top": ["Ah"], "middle": ["Ks"], "bottom": []},
            }
        ],
        "final": {"p0": {}, "p1": {}},
    }
    nonfire = {
        "seat": "first",
        "override_fired": False,
        "baseline_action": action,
        "final_action": deepcopy(action),
    }
    fired = {
        "seat": "first",
        "override_fired": True,
        "baseline_action": action,
        "final_action": deepcopy(action),
    }

    _augment_profile_topk_counterfactuals(
        ab_rows=[nonfire, fired],
        ba_rows=[],
        hand_ab=hand,
        hand_ba=hand,
        shadow_ab=deepcopy(hand),
        shadow_ba=deepcopy(hand),
        paired_index=0,
        hand_seed=7,
    )

    assert nonfire["nonfire_cancellation_valid"] is True
    assert nonfire["realized_candidate_seat_delta"] == 0.0
    assert nonfire["realized_delta_basis"] == "same_snapshot_nonfire_trajectory_identity_v1"
    assert fired["realized_delta_valid"] is False
    assert "realized_candidate_seat_delta" not in fired

    broken_shadow = deepcopy(hand)
    broken_shadow["turns"][0]["discards"] = ["3c"]
    broken = deepcopy(nonfire)
    _augment_profile_topk_counterfactuals(
        ab_rows=[broken],
        ba_rows=[],
        hand_ab=hand,
        hand_ba=hand,
        shadow_ab=broken_shadow,
        shadow_ba=hand,
        paired_index=0,
        hand_seed=7,
    )
    assert broken["nonfire_cancellation_valid"] is False
    assert "realized_candidate_seat_delta" not in broken

    mismatched_key = deepcopy(nonfire)
    mismatched_key["final_action_key"] = (
        "rak1:0000000000001:0000000000000:0000000000000:0000000000000"
    )
    _augment_profile_topk_counterfactuals(
        ab_rows=[mismatched_key],
        ba_rows=[],
        hand_ab=hand,
        hand_ba=hand,
        shadow_ab=deepcopy(hand),
        shadow_ba=deepcopy(hand),
        paired_index=0,
        hand_seed=7,
    )
    assert mismatched_key["nonfire_action_key_identical"] is False
    assert mismatched_key["nonfire_cancellation_valid"] is False


def test_final_action_match_status_prefers_checked_semantic_action_over_index():
    action = {
        "placements": [["Ah", "top"], ["Ks", "middle"]],
        "discards": ["2c"],
    }
    row = {
        "final_action_index": 4,
        "baseline_action_index": 4,
        "final_action": action,
        "baseline_action": deepcopy(action),
        "final_action_key": (
            "rak1:0000000000001:0000000000000:0000000000000:0000000000000"
        ),
    }

    assert _final_action_match_status(row) == "mismatch"

    legacy = {"final_action_index": 2, "baseline_action_index": 2}
    assert _final_action_match_status(legacy) == "unknown"
    legacy["legal_action_order_digest"] = "a" * 64
    assert _final_action_match_status(legacy) == "match"
