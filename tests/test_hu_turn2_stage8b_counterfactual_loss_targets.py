from pathlib import Path

from ofc_regular.prepare_hu_turn2_stage8b_counterfactual_loss_targets import (
    build_replay_index,
    downstream_coverage_rows,
    local_replay_bucket,
    replay_summary_key,
    summary_rows,
    runtime_decision_key,
    target_row,
    target_rows,
    target_use_for,
)


def _decision(**overrides):
    row = {
        "_source_log": "outputs/evals/run/runtime_decisions.jsonl",
        "config_id": "cfg",
        "hand_seed": 123,
        "hand_id": 123,
        "game_id": 123,
        "seat": "first",
        "seat_swap": "ab",
        "override_fired": True,
        "realized_delta_valid": True,
        "realized_candidate_seat_delta": -6.0,
        "realized_delta_basis": "opposite_seat_swap_counterfactual",
        "final_action_index": 11,
        "rerank_best_index": 11,
        "baseline_action_index": 1,
        "predicted_delta": -1.5,
        "gate_probability": 0.9,
        "candidate_ev_rank": 6,
        "rerank_delta": 2.0,
        "rerank_delta_se": 0.5,
        "confirm_delta_count": 256,
        "hero_board": {"top": ["Ad"], "middle": [], "bottom": []},
        "opponent_board": {"top": ["Qd"], "middle": [], "bottom": []},
        "dead_cards": ["2c"],
        "cards_to_place": ["3h", "4s", "6d"],
        "baseline_action": {"placements": [["3h", "top"], ["4s", "middle"]], "discards": ["6d"]},
        "final_action": {"placements": [["3h", "middle"], ["6d", "top"]], "discards": ["4s"]},
        "post_t2_baseline_board": {"top": ["Ad", "3h"], "middle": ["4s"], "bottom": []},
        "post_t2_candidate_board": {"top": ["Ad", "6d"], "middle": ["3h"], "bottom": []},
        "t3_decision_summary": {
            "candidate": {"seat": "first", "stage7_override_fired": True},
            "baseline": {"seat": "first", "stage7_override_fired": False},
        },
        "downstream_override_fired": True,
        "paired_future_delta_summary": {
            "candidate_index": 11,
            "baseline_index": 1,
            "count": 256,
            "mean": -1.25,
            "standard_error": 0.25,
            "p05": -8.0,
            "p95": 4.0,
        },
        "paired_future_delta_source": "confirm",
        "candidate_final_board_hero": {"top": ["Ad", "6d", "6h"], "middle": ["3h", "3d", "4d", "5d", "7d"], "bottom": ["8c", "8d", "8h", "9s", "9c"]},
        "candidate_final_board_opponent": {"top": ["Qd", "Qs", "2c"], "middle": ["Ah", "Kh", "Th", "9h", "3h"], "bottom": ["Ac", "Kc", "Tc", "9c", "3c"]},
        "candidate_hero_foul": False,
        "candidate_opponent_foul": True,
        "candidate_hero_royalty": 6,
        "candidate_opponent_royalty": 0,
        "candidate_royalty_delta": 6,
        "candidate_hero_fl_entry": False,
        "candidate_hero_fl_stay": False,
        "candidate_opponent_fl_entry": False,
        "candidate_line_score_delta": 0,
        "candidate_scoop_delta": 0,
        "candidate_foul_delta": 6,
        "candidate_terminal_score": 12.0,
        "baseline_final_board_hero": {"top": ["Ad", "3h", "2d"], "middle": ["4s", "4d", "5d", "7d", "8d"], "bottom": ["8c", "8h", "9s", "9c", "Ts"]},
        "baseline_final_board_opponent": {"top": ["Qd", "Qs", "2c"], "middle": ["Ah", "Kh", "Th", "9h", "3h"], "bottom": ["Ac", "Kc", "Tc", "9c", "3c"]},
        "baseline_hero_foul": False,
        "baseline_opponent_foul": True,
        "baseline_hero_royalty": 0,
        "baseline_opponent_royalty": 0,
        "baseline_royalty_delta": 0,
        "baseline_hero_fl_entry": False,
        "baseline_hero_fl_stay": False,
        "baseline_opponent_fl_entry": False,
        "baseline_line_score_delta": 0,
        "baseline_scoop_delta": 0,
        "baseline_foul_delta": 6,
        "baseline_terminal_score": 6.0,
        "terminal_score_vs_baseline": 6.0,
        "royalty_delta_vs_baseline": 6.0,
    }
    row.update(overrides)
    return row


def _pack(**overrides):
    row = {
        "schema": "hu_turn2_stage8b_topk_hard_negative_v1",
        "_source_log": "outputs/evals/replay_pack/topk_all_fired_deduped.jsonl",
        "source_log": "outputs\\evals\\run\\runtime_decisions.jsonl",
        "config_id": "cfg",
        "hand_seed": 123,
        "hand_id": 123,
        "game_id": 123,
        "seat": "first",
        "seat_swap": "ab",
        "realized_delta": -6.0,
        "realized_delta_basis": "opposite_seat_swap_counterfactual",
        "candidate_action_index": 11,
        "rerank_best_index": 11,
        "baseline_action_index": 1,
        "predicted_delta": -1.5,
        "gate_probability": 0.9,
        "candidate_ev_rank": 6,
        "confirm_delta": 2.0,
        "confirm_delta_se": 0.5,
        "confirm_count": 128,
        "hero_board": {"top": ["Ad"], "middle": [], "bottom": []},
        "opponent_board": {"top": ["Qd"], "middle": [], "bottom": []},
        "dead_cards": ["2c"],
        "cards_to_place": ["3h", "4s", "6d"],
        "baseline_action": {
            "placements": [["3h", "top"], ["4s", "middle"]],
            "discards": ["6d"],
            "next_board": {"top": ["Ad", "3h"], "middle": ["4s"], "bottom": []},
        },
        "candidate_action": {
            "placements": [["3h", "middle"], ["6d", "top"]],
            "discards": ["4s"],
            "next_board": {"top": ["Ad", "6d"], "middle": ["3h"], "bottom": []},
        },
    }
    row.update(overrides)
    return row


def _replay(**overrides):
    row = {
        "source_log": "outputs\\evals\\run\\runtime_decisions.jsonl",
        "source_config_id": "cfg",
        "hand_seed": "123",
        "seat": "first",
        "candidate_index": "11",
        "logged_baseline_index": "1",
        "status": "ok",
        "future_samples": "2048",
        "delta_for_label": "1.2",
        "delta_standard_error_for_label": "0.2",
        "replay_delta_lcb196": "0.8",
        "safe_lcb196_label": "positive",
        "hard_negative_label": "0",
        "action_mapping_status": "ok",
    }
    row.update(overrides)
    return row


def test_runtime_and_replay_keys_match_across_path_separators():
    decision = _decision()
    replay = _replay()

    assert runtime_decision_key(decision) == replay_summary_key(replay)


def test_topk_pack_and_replay_keys_match_original_source_log():
    pack = _pack()
    replay = _replay()

    assert runtime_decision_key(pack) == replay_summary_key(replay)


def test_local_replay_bucket_distinguishes_positive_gray_and_negative():
    assert local_replay_bucket(_replay(replay_delta_lcb196="0.1", delta_for_label="0.2")) == "local_positive_lcb"
    assert local_replay_bucket(_replay(replay_delta_lcb196="-0.1", delta_for_label="0.2")) == "local_positive_gray"
    assert local_replay_bucket(_replay(hard_negative_label="1", delta_for_label="-0.2")) == "local_negative"
    assert local_replay_bucket(None) == "missing"


def test_target_use_keeps_local_positive_realized_loss_out_of_local_ev_training():
    assert target_use_for(-6.0, "local_positive_lcb") == "whole_game_risk_only"
    assert target_use_for(-6.0, "local_positive_gray") == "whole_game_risk_only"
    assert target_use_for(-6.0, "local_negative") == "local_ev_hard_negative"
    assert target_use_for(-6.0, "missing") == "requires_local_replay"
    assert target_use_for(1.0, "local_negative") == "not_realized_loss"
    assert (
        target_use_for(1.0, "local_negative", include_non_loss_controls=True)
        == "whole_game_non_loss_control"
    )


def test_target_row_marks_realized_loss_with_local_positive_as_risk_only():
    replay_index = build_replay_index([_replay()])

    row = target_row(
        _decision(),
        replay_index=replay_index,
        loss_threshold=0.0,
        local_positive_lcb_threshold=0.0,
    )

    assert row is not None
    assert row["realized_loss_label"] == 1
    assert row["local_replay_bucket"] == "local_positive_lcb"
    assert row["recommended_training_use"] == "whole_game_risk_only"
    assert row["use_for_local_ev_hard_negative"] == 0
    assert row["use_for_whole_game_risk_head"] == 1
    assert row["local_replay_delta"] == 1.2
    assert row["post_t2_candidate_board"]["top"] == ["Ad", "6d"]
    assert row["post_t2_baseline_board"]["middle"] == ["4s"]
    assert row["t3_decision_summary"]["candidate"]["stage7_override_fired"] is True
    assert row["downstream_override_fired"] is True
    assert row["paired_future_delta_summary"]["mean"] == -1.25
    assert row["paired_future_delta_source"] == "confirm"
    assert row["final_board_hero"] == row["candidate_final_board_hero"]
    assert row["baseline_final_board_hero"]["top"] == ["Ad", "3h", "2d"]
    assert row["hero_foul"] is False
    assert row["opponent_foul"] is True
    assert row["hero_royalty"] == 6
    assert row["line_score_delta"] == 0
    assert row["scoop_delta"] == 0
    assert row["terminal_score"] == 12.0
    assert row["terminal_score_vs_baseline"] == 6.0
    assert row["royalty_delta_vs_baseline"] == 6.0
    assert row["downstream_trajectory_complete"] == 1
    assert row["downstream_trajectory_present_fields"] == row["downstream_trajectory_total_fields"]


def test_target_row_reuses_replay_across_config_when_state_action_matches():
    replay_index = build_replay_index([_replay(source_config_id="cfg_a")])

    row = target_row(
        _decision(config_id="cfg_b"),
        replay_index=replay_index,
        loss_threshold=0.0,
        local_positive_lcb_threshold=0.0,
    )

    assert row is not None
    assert row["local_replay_bucket"] == "local_positive_lcb"
    assert row["recommended_training_use"] == "whole_game_risk_only"


def test_target_row_accepts_replay_pack_without_original_runtime_log():
    replay_index = build_replay_index([_replay(hard_negative_label="1", delta_for_label="-0.4", replay_delta_lcb196="-1.0")])

    row = target_row(
        _pack(),
        replay_index=replay_index,
        loss_threshold=0.0,
        local_positive_lcb_threshold=0.0,
    )

    assert row is not None
    assert row["source_log"] == "outputs\\evals\\run\\runtime_decisions.jsonl"
    assert row["target_source_path"] == "outputs/evals/replay_pack/topk_all_fired_deduped.jsonl"
    assert row["candidate_index"] == 11
    assert row["baseline_index"] == 1
    assert row["confirm_delta"] == 2.0
    assert row["confirm_delta_se"] == 0.5
    assert row["confirm_delta_count"] == 128
    assert row["local_replay_bucket"] == "local_negative"
    assert row["recommended_training_use"] == "local_ev_hard_negative"
    assert row["use_for_local_ev_hard_negative"] == 1
    assert row["candidate_action"]["placements"] == [["3h", "middle"], ["6d", "top"]]
    assert row["post_t2_candidate_board"]["top"] == ["Ad", "6d"]
    assert row["post_t2_baseline_board"]["middle"] == ["4s"]
    assert row["downstream_trajectory_complete"] == 0
    assert row["downstream_trajectory_present_fields"] < row["downstream_trajectory_total_fields"]


def test_target_rows_filters_non_fired_and_non_loss_rows():
    rows = target_rows(
        [
            (
                Path("outputs/evals/run/runtime_decisions.jsonl"),
                [
                    _decision(_source_log=""),
                    _decision(_source_log="", override_fired=False),
                    _decision(_source_log="", realized_candidate_seat_delta=2.0),
                ],
            )
        ],
        replay_rows=[_replay()],
    )

    assert len(rows) == 1
    assert rows[0]["recommended_training_use"] == "whole_game_risk_only"


def test_target_rows_can_emit_non_loss_controls_for_risk_head():
    rows = target_rows(
        [
            (
                Path("outputs/evals/run/runtime_decisions.jsonl"),
                [
                    _decision(_source_log="", realized_candidate_seat_delta=-2.0),
                    _decision(_source_log="", hand_seed=124, realized_candidate_seat_delta=3.0),
                ],
            )
        ],
        replay_rows=[_replay(), _replay(hand_seed="124", delta_for_label="2.0", replay_delta_lcb196="1.0")],
        include_non_loss_controls=True,
    )

    assert [row["recommended_training_use"] for row in rows] == [
        "whole_game_risk_only",
        "whole_game_non_loss_control",
    ]
    assert [row["realized_loss_label"] for row in rows] == [1, 0]
    assert [row["whole_game_risk_label"] for row in rows] == [1, 0]
    assert all(row["use_for_whole_game_risk_head"] == 1 for row in rows)


def test_summary_and_coverage_report_downstream_trajectory_completeness():
    replay_index = build_replay_index([_replay()])
    complete = target_row(
        _decision(),
        replay_index=replay_index,
        loss_threshold=0.0,
        local_positive_lcb_threshold=0.0,
    )
    incomplete = target_row(
        _pack(realized_delta=2.0, realized_delta_basis="topk_replay_pack"),
        replay_index=replay_index,
        loss_threshold=0.0,
        local_positive_lcb_threshold=0.0,
        include_non_loss_controls=True,
    )

    rows = [complete, incomplete]
    summary = {row["metric"]: row["value"] for row in summary_rows(rows)}
    coverage = {
        (row["group_field"], row["group_value"], row["field"]): row
        for row in downstream_coverage_rows(rows)
    }

    assert summary["downstream_trajectory_complete_rows"] == 1
    assert summary["downstream_trajectory_incomplete_rows"] == 1
    assert coverage[("overall", "all", "all_downstream_trajectory_fields")]["present_rows"] == 1
    assert (
        coverage[
            ("realized_delta_basis", "opposite_seat_swap_counterfactual", "all_downstream_trajectory_fields")
        ]["present_rate"]
        == 1.0
    )
    assert coverage[("realized_delta_basis", "topk_replay_pack", "all_downstream_trajectory_fields")][
        "present_rate"
    ] == 0.0
