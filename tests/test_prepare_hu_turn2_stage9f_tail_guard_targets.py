from pathlib import Path

from ofc_regular.prepare_hu_turn2_stage9f_tail_guard_targets import (
    build_target,
    missing_replay_fields,
    select_targets,
)


def fired_row(**overrides):
    row = {
        "config_id": "stage9f_cse2_csemax2_firstseat",
        "override_fired": True,
        "realized_delta_valid": True,
        "realized_candidate_seat_delta": 3.0,
        "confirm_delta": 4.0,
        "confirm_delta_se": 2.0,
        "seed": 1,
        "hand_id": 2,
        "seat": "first",
        "candidate_ev_rank": 1,
        "hero_board": {"top": ["Qh"], "middle": [], "bottom": []},
        "opponent_board": {"top": ["Ah"], "middle": [], "bottom": []},
        "dead_cards": ["2c"],
        "visible_dead_cards": ["Ah", "2c"],
        "hero_private_discards": ["2c"],
        "opponent_private_discards": ["3c"],
        "cards_to_place": ["Ks", "Kd", "4h"],
        "baseline_action": {"placements": [["Ks", "top"], ["Kd", "top"]], "discards": ["4h"]},
        "final_action": {"placements": [["Ks", "middle"], ["Kd", "middle"]], "discards": ["4h"]},
    }
    row.update(overrides)
    return row


def test_build_target_labels_tail_loss_and_replay_ready():
    row = fired_row(realized_candidate_seat_delta=-9.0)

    target = build_target(
        row,
        target_group="tail_loss",
        severe_loss_threshold=8.0,
        safe_positive_threshold=2.0,
        source_log=Path("topk_decisions.jsonl"),
    )

    assert target["tail_loss_label"] == 1
    assert target["severe_tail_loss_label"] == 1
    assert target["safe_positive_label"] == 0
    assert target["replay_ready"] is True
    assert target["missing_replay_fields"] == ""
    assert target["confirm_z"] == 2.0


def test_missing_replay_fields_keeps_offline_truth_optional():
    row = fired_row(dead_cards=[], opponent_private_discards=[])

    assert missing_replay_fields(row) == []


def test_missing_replay_fields_requires_hero_visible_discard():
    row = fired_row(hero_private_discards=[], visible_dead_cards=[])

    assert "hero_visible_discard" in missing_replay_fields(row)


def test_select_targets_includes_loss_positive_zero_and_boundary_groups():
    rows = [
        (fired_row(seed=1, hand_id=1, realized_candidate_seat_delta=-10.0, confirm_delta=4.0, confirm_delta_se=2.0), Path("a.jsonl")),
        (fired_row(seed=1, hand_id=2, realized_candidate_seat_delta=5.0, confirm_delta=6.0, confirm_delta_se=2.0), Path("a.jsonl")),
        (fired_row(seed=1, hand_id=3, realized_candidate_seat_delta=0.0, confirm_delta=4.1, confirm_delta_se=2.0), Path("a.jsonl")),
    ]

    targets = select_targets(
        rows,
        loss_limit=10,
        positive_limit=10,
        zero_limit=10,
        boundary_limit=10,
        severe_loss_threshold=8.0,
        safe_positive_threshold=2.0,
        confirm_z_threshold=2.0,
        confirm_z_boundary_width=0.1,
    )

    groups = {target["target_group"] for target in targets}
    assert {"tail_loss", "positive_control", "zero_control", "confirm_z_boundary"}.issubset(groups)
    assert any(target["tail_loss_label"] == 1 for target in targets)
    assert any(target["safe_positive_label"] == 1 for target in targets)


def test_select_targets_can_dedupe_same_event_across_groups():
    rows = [
        (
            fired_row(
                seed=1,
                hand_id=1,
                realized_candidate_seat_delta=5.0,
                confirm_delta=4.0,
                confirm_delta_se=2.0,
            ),
            Path("a.jsonl"),
        )
    ]

    targets = select_targets(
        rows,
        loss_limit=10,
        positive_limit=10,
        zero_limit=10,
        boundary_limit=10,
        severe_loss_threshold=8.0,
        safe_positive_threshold=2.0,
        confirm_z_threshold=2.0,
        confirm_z_boundary_width=0.1,
        dedupe_event_key=True,
    )

    assert len(targets) == 1
    assert targets[0]["target_group"] == "confirm_z_boundary"
    assert targets[0]["source_target_groups"] == "confirm_z_boundary,positive_control"
    assert targets[0]["duplicate_source_rows"] == 2
    assert targets[0]["replay_event_key"]
