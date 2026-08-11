from ofc_regular.prepare_hu_turn2_stage8b_topk_hard_negatives import (
    classify_row,
    extract_rows,
    has_replay_fields,
    replay_blocker,
    summary_rows,
)


def _decision(**overrides):
    base = {
        "override_fired": True,
        "realized_delta_valid": True,
        "realized_candidate_seat_delta": -6.0,
        "rerank_delta": 3.0,
        "rerank_delta_se": 0.5,
        "stage_a_delta": 4.0,
        "predicted_delta": -2.0,
        "gate_probability": 0.9,
        "candidate_ev_rank": 3,
        "local_ev_risk_enabled": True,
        "local_ev_risk_probability": 0.42,
        "local_ev_risk_threshold": 0.35,
        "local_ev_risk_rank_min": 4,
        "local_ev_risk_rank_max": 5,
        "local_ev_risk_rank_guard_passed": False,
        "local_ev_risk_vetoed": False,
        "confirm_delta_count": 256,
        "hand_seed": 1,
        "seat": "first",
        "seat_swap": "ab",
        "config_id": "cfg",
        "hero_board": {"top": ["Kh"], "middle": ["As"], "bottom": ["2d"]},
        "opponent_board": {"top": ["Qh"], "middle": ["Ac"], "bottom": ["3d"]},
        "dead_cards": ["2c"],
        "visible_dead_cards": ["Qh", "Ac", "3d", "2c"],
        "hero_private_discards": ["2c"],
        "opponent_private_discards": ["3c"],
        "cards_to_place": ["5h", "2c", "7c"],
        "baseline_action": {
            "placements": [["2c", "bottom"], ["7c", "bottom"]],
            "discards": ["5h"],
        },
        "final_action": {
            "placements": [["5h", "middle"], ["7c", "top"]],
            "discards": ["2c"],
        },
        "baseline_action_index": 1,
        "final_action_index": 3,
        "rerank_best_index": 3,
    }
    base.update(overrides)
    return base


def test_classify_row_marks_negative_model_delta_false_positive():
    row = _decision()

    assert classify_row(row, false_positive_threshold=0.0, neutral_threshold=1.0) == "false_positive_negative_model_delta"


def test_has_replay_fields_requires_actor_visible_discard_and_actions():
    assert has_replay_fields(_decision())
    assert has_replay_fields(_decision(dead_cards=[]))
    assert has_replay_fields(_decision(opponent_private_discards=[]))
    assert has_replay_fields(_decision(visible_dead_cards=[]))
    assert has_replay_fields(_decision(hero_private_discards=[]))
    assert not has_replay_fields(
        _decision(visible_dead_cards=[], hero_private_discards=[])
    )
    assert replay_blocker(_decision(opponent_private_discards=[])) == ""
    assert replay_blocker(
        _decision(visible_dead_cards=[], hero_private_discards=[])
    ) == "hero_visible_discard"


def test_extract_rows_dedupes_false_positive_by_state_action():
    duplicate_better = _decision(rerank_delta=2.0, realized_candidate_seat_delta=-4.0)
    neutral = _decision(
        hand_seed=2,
        realized_candidate_seat_delta=0.0,
        predicted_delta=1.0,
        rerank_delta=2.5,
    )
    positive = _decision(
        hand_seed=3,
        realized_candidate_seat_delta=8.0,
        predicted_delta=1.0,
        rerank_delta=2.5,
    )

    false_positives, neutral_rows, all_fired = extract_rows(
        [("log.jsonl", [_decision(), duplicate_better, neutral, positive])],
        false_positive_threshold=0.0,
        neutral_overconfirm_threshold=1.0,
    )

    assert len(false_positives) == 1
    assert false_positives[0]["realized_delta"] == -6.0
    assert false_positives[0]["hard_negative_label"] == 1
    assert false_positives[0]["risk_probability"] == 0.42
    assert false_positives[0]["local_ev_risk_probability"] == 0.42
    assert false_positives[0]["local_ev_risk_rank_guard_passed"] is False
    assert false_positives[0]["replay_ready"] is True
    assert false_positives[0]["visible_dead_cards"] == ["Qh", "Ac", "3d", "2c"]
    assert false_positives[0]["hero_private_discards"] == ["2c"]
    assert false_positives[0]["opponent_private_discards"] == ["3c"]
    assert len(neutral_rows) == 1
    assert neutral_rows[0]["diagnosis"] == "neutral_overconfirm"
    assert len(all_fired) == 3


def test_summary_rows_mark_confirm_mean_as_diagnostic_only():
    summary = summary_rows([_decision()], [], [_decision()])
    by_group = {row["group"]: row for row in summary}

    assert by_group["false_positive"]["realized_delta_metric_source"] == "realized_fired_whole_game_delta"
    assert by_group["false_positive"]["confirm_delta_metric_role"] == "gate_diagnostic_only"
    assert by_group["false_positive"]["confirm_delta_performance_claim_allowed"] is False
