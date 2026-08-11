import json

from ofc_regular.analyze_hu_turn2_stage9f_profile_canary import (
    fired_loss_rows,
    missing_replay_fields,
    summarize_decisions,
)


def _decision(**overrides):
    row = {
        "runtime_profile": "stage9f_cse1p5_firstseat",
        "t3_continuation_policy": "Stage7_candidate_A_m5_r10",
        "seat": "first",
        "override_fired": False,
        "no_override_reason": "topk_empty",
        "dead_cards": ["2c"],
        "visible_dead_cards": ["2c"],
        "hero_private_discards": ["2c"],
        "opponent_private_discards": ["3d"],
        "baseline_action": {"placements": []},
        "final_action": {"placements": []},
        "runtime_latency_ms": 10.0,
        "mc_rerank_latency_ms": 0.0,
        "confirm_mc_latency_ms": 0.0,
        "confirm_delta": "",
        "stage_a_delta": "",
        "realized_delta_valid": True,
        "realized_candidate_seat_delta": 0.0,
    }
    row.update(overrides)
    return row


def test_profile_canary_missing_replay_fields():
    assert missing_replay_fields(_decision()) == []
    assert "visible_dead_cards" in missing_replay_fields(_decision(visible_dead_cards=[]))


def test_profile_canary_summary_counts_reasons_and_seats(tmp_path):
    rows = [
        _decision(seat="first", no_override_reason="topk_empty", runtime_latency_ms=10.0),
        _decision(
            seat="first",
            override_fired=True,
            no_override_reason="",
            confirm_delta=3.0,
            stage_a_delta=2.0,
            mc_rerank_latency_ms=15.0,
            confirm_mc_latency_ms=20.0,
            realized_candidate_seat_delta=6.0,
            runtime_latency_ms=40.0,
        ),
        _decision(
            seat="first",
            override_fired=True,
            no_override_reason="",
            confirm_delta=2.0,
            stage_a_delta=1.0,
            mc_rerank_latency_ms=12.0,
            confirm_mc_latency_ms=8.0,
            realized_candidate_seat_delta=-2.0,
            runtime_latency_ms=30.0,
        ),
        _decision(seat="second", no_override_reason="seat_not_allowed", runtime_latency_ms=20.0),
    ]
    matchup = {"profile_a": "stage9f_cse1p5_firstseat", "profile_b": "stage7_m5_r10", "hands": 6}

    summary, reasons, seats, configs = summarize_decisions(
        rows,
        decisions_path=tmp_path / "decisions.jsonl",
        matchup_summary=matchup,
    )

    assert summary[0]["decision_count"] == 4
    assert summary[0]["override_count"] == 2
    assert summary[0]["confirm_evaluated_count"] == 2
    assert summary[0]["stage_a_evaluated_count"] == 2
    assert summary[0]["realized_delta_count"] == 4
    assert summary[0]["realized_override_count"] == 2
    assert summary[0]["realized_per_fire_delta_mean"] == 2.0
    assert summary[0]["realized_per_fire_delta_min"] == -2.0
    assert summary[0]["realized_per_fire_delta_max"] == 6.0
    assert summary[0]["realized_per_fire_positive_count"] == 1
    assert summary[0]["realized_per_fire_negative_count"] == 1
    assert summary[0]["realized_per_fire_loss_max"] == 2.0
    assert summary[0]["realized_estimated_ev_per_decision"] == 1.0
    assert summary[0]["non_fired_nonzero_count"] == 0
    assert summary[0]["replay_ready_count"] == 4
    assert json.loads(summary[0]["missing_replay_field_counts"]) == {}
    assert summary[0]["latency_component_stage_a_p95_ms"] == 15.0
    assert summary[0]["latency_component_confirm_p95_ms"] == 20.0
    assert summary[0]["latency_component_overhead_p95_ms"] == 20.0
    reason_counts = {row["bucket"]: row["decision_count"] for row in reasons}
    assert reason_counts == {"override_fired": 2, "seat_not_allowed": 1, "topk_empty": 1}
    seat_counts = {row["seat"]: row["decision_count"] for row in seats}
    assert seat_counts == {"first": 3, "second": 1}
    config_counts = {(row["bucket_type"], row["bucket"]): row["decision_count"] for row in configs}
    assert config_counts[("runtime_profile", "stage9f_cse1p5_firstseat")] == 4
    assert config_counts[("t3_continuation_policy", "Stage7_candidate_A_m5_r10")] == 4


def test_profile_canary_fired_loss_rows_keep_replay_fields():
    rows = [
        _decision(
            override_fired=True,
            realized_candidate_seat_delta=3.0,
            hand_seed=1,
            baseline_action_index=2,
            final_action_index=3,
        ),
        _decision(
            override_fired=True,
            realized_candidate_seat_delta=-7.0,
            hand_seed=2,
            baseline_action_index=4,
            final_action_index=5,
            confirm_delta=1.5,
            confirm_delta_se=0.5,
        ),
    ]

    losses = fired_loss_rows(rows)

    assert losses[0]["hand_seed"] == 2
    assert losses[0]["loss"] == 7.0
    assert losses[0]["dead_cards"] == ["2c"]
    assert losses[0]["baseline_action_index"] == 4
    assert losses[0]["final_action_index"] == 5
    assert losses[0]["confirm_delta"] == 1.5
