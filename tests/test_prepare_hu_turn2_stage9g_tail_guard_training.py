from ofc_regular.prepare_hu_turn2_stage9g_tail_guard_training import (
    build_rows,
    build_training_row,
    recommended_training_use,
)


def target_row(**overrides):
    row = {
        "target_id": "target-1",
        "config_id": "stage9f_cse2_csemax2_firstseat",
        "seed": 1,
        "hand_id": 1,
        "hand_seed": 1,
        "seat": "first",
        "hero_board": {"top": ["Qh"], "middle": ["Kh", "Ks"], "bottom": ["5h", "5d", "6c", "6h"]},
        "opponent_board": {"top": ["Ac"], "middle": ["6s", "8c", "8d"], "bottom": ["Jd", "Th", "Tc"]},
        "cards_to_place": ["Ah", "Jh", "Qd"],
        "dead_cards": ["4s", "3d"],
        "baseline_action_index": 19,
        "candidate_action_index": 11,
        "baseline_action": {"placements": [["Jh", "middle"], ["Qd", "top"]], "discards": ["Ah"]},
        "candidate_action": {"placements": [["Ah", "middle"], ["Qd", "top"]], "discards": ["Jh"]},
        "predicted_delta": 1.5,
        "gate_probability": 0.9,
        "confirm_delta": 2.0,
        "confirm_delta_se": 0.5,
        "candidate_ev_rank": 2,
    }
    row.update(overrides)
    return row


def label_row(**overrides):
    row = {
        "target_id": "label-1",
        "source_target_ids": "target-1",
        "replay_event_key": "event-1",
        "high_mc_tail_guard_label": "hard_negative",
        "high_mc_hard_negative_label": "1",
        "high_mc_safe_positive_label": "0",
        "high_mc_gray_label": "0",
        "high_mc_gain_mean": "-1.25",
        "high_mc_gain_stderr": "0.4",
        "high_mc_gain_lower95": "-2.0",
        "high_mc_training_weight": "5.0",
        "duplicate_source_rows": "2",
        "source_target_groups": "tail_loss,confirm_z_boundary",
        "mc_n": "512",
    }
    row.update(overrides)
    return row


def test_recommended_training_use_maps_labels_to_whole_game_risk_targets():
    assert recommended_training_use("hard_negative") == "whole_game_risk_only"
    assert recommended_training_use("safe_positive") == "whole_game_non_loss_control"
    assert recommended_training_use("gray") == "stage9g_tail_guard_gray"


def test_build_training_row_is_stage8c_risk_head_compatible():
    row = build_training_row(label_row(), target_row())

    assert row["recommended_training_use"] == "whole_game_risk_only"
    assert row["use_for_whole_game_risk_head"] == 1
    assert row["realized_delta"] == -1.25
    assert row["baseline_index"] == 19
    assert row["candidate_index"] == 11
    assert row["local_replay_status"] == "ok"
    assert row["local_replay_action_mapping_status"] == "ok"
    assert row["local_replay_delta"] == -1.25
    assert row["stage9g_duplicate_source_rows"] == 2


def test_build_rows_excludes_gray_by_default_and_keeps_safe_controls():
    rows, missing = build_rows(
        [target_row(), target_row(target_id="target-2")],
        [
            label_row(source_target_ids="target-1", high_mc_tail_guard_label="hard_negative"),
            label_row(source_target_ids="target-2", high_mc_tail_guard_label="safe_positive"),
            label_row(source_target_ids="target-3", high_mc_tail_guard_label="gray"),
        ],
        include_gray=False,
    )

    assert missing == []
    assert [row["recommended_training_use"] for row in rows] == [
        "whole_game_risk_only",
        "whole_game_non_loss_control",
    ]


def test_build_rows_can_include_gray_for_audit():
    rows, missing = build_rows(
        [target_row()],
        [label_row(source_target_ids="target-1", high_mc_tail_guard_label="gray")],
        include_gray=True,
    )

    assert missing == []
    assert rows[0]["recommended_training_use"] == "stage9g_tail_guard_gray"
    assert rows[0]["use_for_whole_game_risk_head"] == 0
