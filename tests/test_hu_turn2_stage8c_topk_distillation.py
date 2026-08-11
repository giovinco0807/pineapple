from ofc_regular.prepare_hu_turn2_stage8c_topk_distillation import (
    FIRE_SELECTOR_REJECTED_USE,
    LOSS_USE,
    POSITIVE_USE,
    REJECTED_USE,
    TOPK_EMPTY_USE,
    breakdown_rows,
    build_rows,
    candidate_from_row,
    label_for_row,
    summary_rows,
)


def _action(card="As", row="top"):
    return {
        "placements": [[card, row], ["Kh", "middle"]],
        "discards": ["2c"],
    }


def _decision(**overrides):
    row = {
        "config_id": "cfg",
        "hand_seed": 1,
        "hand_id": "h1",
        "seat": "first",
        "seat_swap": "ab",
        "hero_board": {"top": ["Qh"], "middle": ["2d"], "bottom": ["3c"]},
        "opponent_board": {"top": ["Ah"], "middle": ["4d"], "bottom": ["5c"]},
        "dead_cards": ["9c"],
        "visible_dead_cards": ["Ah", "4d", "5c", "9c"],
        "cards_to_place": ["As", "Kh", "2c"],
        "baseline_action": _action("Qs", "bottom"),
        "baseline_action_index": 7,
        "final_action": _action(),
        "final_action_index": 3,
        "rerank_best_action": _action(),
        "rerank_best_index": 3,
        "stage8b_top1_action": _action("Jd", "top"),
        "stage8b_top1_index": 4,
        "candidate_ev_rank": 2,
        "predicted_delta": 1.25,
        "gate_probability": 0.8,
        "confirm_delta": 2.0,
        "confirm_delta_se": 0.5,
        "confirm_delta_count": 128,
        "override_fired": True,
        "realized_delta_valid": True,
        "realized_candidate_seat_delta": 4.0,
        "no_override_reason": "",
    }
    row.update(overrides)
    return row


def test_label_for_row_separates_realized_positive_loss_and_rejected():
    assert label_for_row(_decision(realized_candidate_seat_delta=4.0), positive_threshold=0.0) == (1, POSITIVE_USE)
    assert label_for_row(_decision(realized_candidate_seat_delta=-2.0), positive_threshold=0.0) == (0, LOSS_USE)
    assert label_for_row(
        _decision(override_fired=False, no_override_reason="below_confirm_delta", realized_candidate_seat_delta=0.0),
        positive_threshold=0.0,
    ) == (0, REJECTED_USE)
    assert label_for_row(
        _decision(
            override_fired=False,
            no_override_reason="below_fire_selector_threshold",
            realized_candidate_seat_delta=0.0,
        ),
        positive_threshold=0.0,
    ) == (0, FIRE_SELECTOR_REJECTED_USE)


def test_candidate_from_row_uses_rerank_action_for_fire_selector_rejected():
    row = _decision(override_fired=False, no_override_reason="below_fire_selector_threshold")

    candidate, index, source = candidate_from_row(row, include_topk_empty=False)

    assert candidate == row["rerank_best_action"]
    assert index == 3
    assert source == "below_fire_selector_threshold"


def test_candidate_from_row_uses_logged_fire_selector_candidate_action():
    row = _decision(
        override_fired=False,
        no_override_reason="below_fire_selector_threshold",
        rerank_best_action=None,
        rerank_best_index=None,
        stage8c_fire_selector_candidates=[
            {
                "action_index": 8,
                "action": _action("9d", "bottom"),
                "probability": 0.31,
                "predicted_delta": 1.3,
                "gate_probability": 0.2,
                "candidate_ev_rank": 2,
            },
            {
                "action_index": 9,
                "action": _action("8d", "middle"),
                "probability": 0.12,
                "predicted_delta": 0.8,
                "gate_probability": 0.1,
                "candidate_ev_rank": 3,
            },
        ],
    )

    candidate, index, source = candidate_from_row(row, include_topk_empty=False)

    assert candidate == _action("9d", "bottom")
    assert index == 8
    assert source == "below_fire_selector_threshold"


def test_candidate_from_row_can_use_topk_empty_when_enabled():
    row = _decision(override_fired=False, no_override_reason="topk_empty", rerank_best_action=None, rerank_best_index=None)

    candidate, index, source = candidate_from_row(row, include_topk_empty=True)

    assert candidate == row["stage8b_top1_action"]
    assert index == 4
    assert source == "topk_empty"


def test_build_rows_balances_and_keeps_replay_ready_payloads():
    rows, source_rows = build_rows(
        [
            (
                "log.jsonl",
                [
                    _decision(hand_seed=1, realized_candidate_seat_delta=4.0),
                    _decision(hand_seed=2, realized_candidate_seat_delta=-2.0),
                    _decision(hand_seed=3, override_fired=False, no_override_reason="below_confirm_se"),
                    _decision(hand_seed=5, override_fired=False, no_override_reason="below_fire_selector_threshold"),
                    _decision(
                        hand_seed=4,
                        override_fired=False,
                        no_override_reason="topk_empty",
                        rerank_best_action=None,
                        rerank_best_index=None,
                    ),
                ],
            )
        ],
        positive_threshold=0.0,
        max_rejected_negatives=10,
        max_fire_selector_negatives=10,
        max_topk_empty_negatives=10,
        sample_seed=1,
        include_topk_empty=True,
    )

    uses = [row["recommended_training_use"] for row in rows]
    assert uses == [FIRE_SELECTOR_REJECTED_USE, LOSS_USE, POSITIVE_USE, REJECTED_USE, TOPK_EMPTY_USE]
    assert all(row["training_ready"] for row in rows)
    assert all(row["candidate_action"] != row["baseline_action"] for row in rows)
    assert source_rows[0]["rows"] == 5


def test_summary_and_breakdown_mark_confirm_delta_as_runtime_feature_only():
    rows, source_rows = build_rows(
        [
            (
                "log.jsonl",
                [
                    _decision(hand_seed=1, realized_candidate_seat_delta=4.0),
                    _decision(hand_seed=2, realized_candidate_seat_delta=-2.0),
                ],
            )
        ],
        positive_threshold=0.0,
        max_rejected_negatives=10,
        max_fire_selector_negatives=10,
        max_topk_empty_negatives=10,
        sample_seed=1,
        include_topk_empty=False,
    )

    summary = {row["metric"]: row["value"] for row in summary_rows(rows, source_rows)}
    breakdown = breakdown_rows(rows)
    overall = next(row for row in breakdown if row["group_field"] == "overall")

    assert summary["primary_label_source"] == "realized_fired_delta_or_confirm_gate_rejection"
    assert summary["confirm_delta_metric_role"] == "runtime_feature_and_gate_diagnostic_only"
    assert summary["confirm_delta_performance_claim_allowed"] is False
    assert overall["primary_label_source"] == "realized_fired_delta_or_confirm_gate_rejection"
    assert overall["confirm_delta_metric_role"] == "runtime_feature_and_gate_diagnostic_only"
    assert overall["confirm_delta_performance_claim_allowed"] is False
