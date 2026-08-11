from ofc_regular.extract_hu_turn2_stage8c_ranker_replay_targets import (
    extract_ranker_targets,
    load_distillation_index,
    selected_top_rows,
    selected_top_rows_with_audit,
)
from ofc_regular.extract_hu_turn2_stage8c_topk_replay_targets import row_key


def _action(card="As", row="top"):
    return {"placements": [[card, row], ["Kh", "middle"]], "discards": ["2c"]}


def _distillation_row(**overrides):
    row = {
        "schema": "hu_turn2_stage8c_topk_confirm_distillation_v1",
        "source_log": "outputs/run/runtime_decisions.jsonl",
        "config_id": "cfg",
        "hand_seed": "2026061501",
        "seat": "first",
        "seat_swap": "ab",
        "state_signature": "state-a",
        "action_signature": "action-a",
        "baseline_action_signature": "baseline-a",
        "candidate_index": 3,
        "baseline_index": 1,
        "recommended_training_use": "topk_confirm_rejected",
        "candidate_source": "below_confirm_delta",
        "hero_board": {"top": ["Qh"], "middle": ["2c"], "bottom": ["3d", "4h", "5s", "6c"]},
        "opponent_board": {"top": ["Ah"], "middle": ["7c"], "bottom": ["8d", "9h", "Ts"]},
        "dead_cards": ["Jc", "2d"],
        "cards_to_place": ["As", "Kh", "2c"],
        "baseline_action": _action("Qs", "bottom"),
        "candidate_action": _action(),
        "realized_delta_observed": False,
        "realized_delta": 0.0,
    }
    row.update(overrides)
    return row


def _top_row(source, **overrides):
    row = {
        "source_log": source["source_log"].replace("\\", "/"),
        "config_id": source["config_id"],
        "hand_seed": str(source["hand_seed"]),
        "seat": source["seat"],
        "seat_swap": source["seat_swap"],
        "state_signature": source["state_signature"],
        "action_signature": source["action_signature"],
        "baseline_action_signature": source["baseline_action_signature"],
        "candidate_index": str(source["candidate_index"]),
        "baseline_index": str(source["baseline_index"]),
        "recommended_training_use": source["recommended_training_use"],
        "ranker": "fire_probability+policy",
        "rank": "1",
        "split": "test",
        "ranker_score": "2.5",
        "risk_probability": "0.8",
        "policy_delta_prediction": "1.25",
        "observed_delta_prediction": "",
        "predicted_delta": "1.0",
        "confirm_delta": "2.0",
        "confirm_delta_se": "0.4",
        "candidate_ev_rank": "2",
        "realized_delta_observed": "0",
    }
    row.update(overrides)
    return row


def test_ranker_target_extraction_outputs_replay_ready_unknown_rows(tmp_path):
    source = _distillation_row()
    source_path = tmp_path / "distillation.jsonl"
    source_path.write_text(__import__("json").dumps(source) + "\n", encoding="utf-8")
    index, manifest = load_distillation_index([source_path])

    targets, audit, result = extract_ranker_targets(
        index,
        [_top_row(source)],
        include_observed=False,
        max_targets_per_ranker=10,
    )

    assert manifest["loaded_distillation_rows"] == 1
    assert len(targets) == 1
    assert targets[0]["schema"] == "hu_turn2_stage8c_prediction_ranker_replay_target_v1"
    assert targets[0]["replay_ready"] is True
    assert targets[0]["observed_performance_claim"] == "No"
    assert targets[0]["ranker"] == "fire_probability+policy"
    assert targets[0]["ranker_role"] == "diagnostic_combo"
    assert targets[0]["runtime_eligible_ranker"] is False
    assert audit[0]["status"] == "target"
    assert audit[0]["ranker_role"] == "diagnostic_combo"
    assert result["counts"]["targets"] == 1


def test_ranker_target_extraction_skips_observed_and_counts_missing():
    unknown = _distillation_row(hand_seed="1", state_signature="unknown", action_signature="unknown-a")
    observed = _distillation_row(
        hand_seed="2",
        state_signature="observed",
        action_signature="observed-a",
        realized_delta_observed=True,
        realized_delta=5.0,
    )
    missing = _distillation_row(hand_seed="3", state_signature="missing", action_signature="missing-a")

    index = {row_key(row): row for row in [unknown, observed]}

    targets, _audit, result = extract_ranker_targets(
        index,
        [
            _top_row(unknown, rank="1"),
            _top_row(observed, rank="2", realized_delta_observed="1"),
            _top_row(missing, rank="3"),
        ],
        include_observed=False,
        max_targets_per_ranker=10,
    )

    assert [row["hand_seed"] for row in targets] == ["1"]
    assert result["counts"]["skipped_observed_delta"] == 1
    assert result["counts"]["missing_distillation_row"] == 1


def test_selected_top_rows_filters_split_ranker_and_rank():
    source = _distillation_row()
    rows = [
        _top_row(source, ranker="a", rank="1", split="test"),
        _top_row(source, ranker="b", rank="1", split="test"),
        _top_row(source, ranker="a", rank="3", split="test"),
        _top_row(source, ranker="a", rank="1", split="val"),
    ]

    selected = selected_top_rows(rows, rankers={"a"}, splits={"test"}, max_rank=2)

    assert len(selected) == 1
    assert selected[0]["ranker"] == "a"
    assert selected[0]["rank"] == "1"
    assert selected[0]["runtime_eligible"] is True


def test_selected_top_rows_filters_seat_and_reports_audit():
    first = _distillation_row(seat="first", hand_seed="1", state_signature="s1", action_signature="a1")
    second = _distillation_row(seat="second", hand_seed="2", state_signature="s2", action_signature="a2")
    rows = [
        _top_row(first, ranker="predicted_delta", rank="1", split="test"),
        _top_row(second, ranker="predicted_delta", rank="2", split="test"),
    ]

    selected, audit = selected_top_rows_with_audit(
        rows,
        rankers=None,
        splits={"test"},
        seats={"first"},
        max_rank=10,
    )

    assert len(selected) == 1
    assert selected[0]["seat"] == "first"
    assert selected[0]["hand_seed"] == "1"
    assert audit["selection_counts"]["selected_top_rows"] == 1
    assert audit["selection_counts"]["skipped_seat_filter"] == 1


def test_selected_top_rows_skips_non_runtime_rankers_by_default():
    source = _distillation_row()
    rows = [
        _top_row(source, ranker="confirm_delta", rank="1", split="test"),
        _top_row(source, ranker="fire_probability", rank="2", split="test"),
    ]

    selected = selected_top_rows(rows, rankers=None, splits={"test"}, max_rank=10)

    assert [row["ranker"] for row in selected] == ["fire_probability"]
    assert selected[0]["ranker_role"] == "deployable_ranker_input"
    assert selected[0]["runtime_eligible"] is True


def test_selected_top_rows_audit_reports_non_runtime_exclusions():
    source = _distillation_row()
    rows = [
        _top_row(source, ranker="confirm_delta", rank="1", split="test"),
        _top_row(source, ranker="confirm_delta+policy", rank="2", split="test"),
        _top_row(source, ranker="fire_probability", rank="3", split="test"),
        _top_row(source, ranker="policy", rank="4", split="test"),
        _top_row(source, ranker="predicted_delta", rank="20", split="test"),
        _top_row(source, ranker="fire_probability", rank="1", split="train"),
    ]

    selected, audit = selected_top_rows_with_audit(
        rows,
        rankers=None,
        splits={"test"},
        max_rank=10,
    )

    assert [row["ranker"] for row in selected] == ["fire_probability", "policy"]
    assert audit["selection_counts"]["input_rows"] == 6
    assert audit["selection_counts"]["selected_top_rows"] == 2
    assert audit["selection_counts"]["skipped_non_runtime_ranker"] == 2
    assert audit["selection_counts"]["skipped_rank_filter"] == 1
    assert audit["selection_counts"]["skipped_split_filter"] == 1
    assert audit["skipped_non_runtime_rankers"] == {
        "confirm_delta": 1,
        "confirm_delta+policy": 1,
    }
    assert audit["selected_top_rows_by_ranker"] == {"fire_probability": 1, "policy": 1}


def test_selected_top_rows_requires_explicit_opt_in_for_confirm_delta_ranker():
    source = _distillation_row()
    rows = [
        _top_row(source, ranker="confirm_delta", rank="1", split="test"),
        _top_row(source, ranker="fire_probability", rank="2", split="test"),
    ]

    selected = selected_top_rows(
        rows,
        rankers=None,
        splits={"test"},
        max_rank=10,
        allow_non_runtime_rankers=True,
    )

    assert [row["ranker"] for row in selected] == ["confirm_delta", "fire_probability"]
    assert selected[0]["ranker_role"] == "replay_triage_only"
    assert selected[0]["runtime_eligible"] is False


def test_selected_top_rows_does_not_trust_malformed_runtime_eligible_flag():
    source = _distillation_row()
    rows = [
        _top_row(
            source,
            ranker="confirm_delta",
            rank="1",
            split="test",
            ranker_role="deployable_ranker_input",
            runtime_eligible="True",
        ),
        _top_row(source, ranker="fire_probability", rank="2", split="test"),
    ]

    selected = selected_top_rows(rows, rankers=None, splits={"test"}, max_rank=10)

    assert [row["ranker"] for row in selected] == ["fire_probability"]


def test_ranker_target_extraction_caps_per_ranker():
    first = _distillation_row(hand_seed="1", state_signature="s1", action_signature="a1")
    second = _distillation_row(hand_seed="2", state_signature="s2", action_signature="a2")
    index = {row_key(row): row for row in [first, second]}

    targets, _audit, result = extract_ranker_targets(
        index,
        [_top_row(first, rank="1"), _top_row(second, rank="2")],
        include_observed=False,
        max_targets_per_ranker=1,
    )

    assert [row["hand_seed"] for row in targets] == ["1"]
    assert result["counts"]["skipped_ranker_cap"] == 1
