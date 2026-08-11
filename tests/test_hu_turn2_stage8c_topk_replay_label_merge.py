from ofc_regular.merge_hu_turn2_stage8c_topk_replay_labels import (
    build_replay_label_index,
    merge_rows,
    read_jsonl_many,
    replay_label,
    summary_rows,
)
from ofc_regular.train_hu_turn2_stage8c_risk_head import target_label_for_row


def _row(**overrides):
    row = {
        "source_log": "outputs/run/runtime_decisions.jsonl",
        "config_id": "cfg",
        "hand_seed": "1",
        "seat": "first",
        "state_signature": "state-a",
        "action_signature": "action-a",
        "candidate_index": 3,
        "baseline_index": 1,
        "recommended_training_use": "topk_confirm_rejected",
        "topk_distill_label_id": 0,
        "hero_board": {"top": ["Qh"], "middle": ["2c"], "bottom": ["3d"]},
        "opponent_board": {"top": ["Ah"], "middle": ["4c"], "bottom": ["5d"]},
        "dead_cards": ["9c"],
        "cards_to_place": ["As", "Kh", "2c"],
        "baseline_action": {"placements": [["As", "top"], ["Kh", "middle"]], "discards": ["2c"]},
        "candidate_action": {"placements": [["As", "middle"], ["Kh", "top"]], "discards": ["2c"]},
    }
    row.update(overrides)
    return row


def _summary(row_index: int, *, delta: float, lcb: float, label: str):
    return {
        "row_index": str(row_index),
        "status": "ok",
        "action_mapping_status": "ok",
        "future_samples": "512",
        "delta_for_label": str(delta),
        "delta_standard_error_for_label": "0.5",
        "replay_delta_lcb196": str(lcb),
        "replay_delta_lcb164": str(lcb + 0.1),
        "safe_lcb196_label": label,
        "hard_negative_label": "1" if delta < 0 else "0",
    }


def test_replay_label_maps_positive_negative_and_gray():
    assert replay_label(_summary(0, delta=2.0, lcb=1.0, label="positive")) == "topk_confirm_replay_positive"
    assert replay_label(_summary(0, delta=-1.0, lcb=-2.0, label="negative")) == "topk_confirm_replay_negative"
    assert replay_label(_summary(0, delta=0.5, lcb=-0.1, label="gray")) == "topk_confirm_replay_gray"


def test_merge_rows_replaces_unknown_labels_and_excludes_gray_from_training():
    distillation = [
        _row(hand_seed="1", state_signature="state-pos", action_signature="action-pos"),
        _row(hand_seed="2", state_signature="state-neg", action_signature="action-neg"),
        _row(hand_seed="3", state_signature="state-gray", action_signature="action-gray"),
    ]
    replay_index = build_replay_label_index(
        distillation,
        [
            _summary(0, delta=2.0, lcb=1.0, label="positive"),
            _summary(1, delta=-1.0, lcb=-2.0, label="negative"),
            _summary(2, delta=0.5, lcb=-0.1, label="gray"),
        ],
    )

    merged, replacements = merge_rows(distillation, replay_index)

    assert [row["recommended_training_use"] for row in merged] == [
        "topk_confirm_replay_positive",
        "topk_confirm_replay_negative",
        "topk_confirm_replay_gray",
    ]
    assert target_label_for_row(merged[0], "topk_confirm_fire") == (1, "topk_confirm_replay_positive")
    assert target_label_for_row(merged[1], "topk_confirm_fire") == (0, "topk_confirm_replay_negative")
    assert target_label_for_row(merged[2], "topk_confirm_fire") is None
    assert merged[0]["realized_delta_basis"] == "mc512_independent_replay"
    assert len(replacements) == 3


def test_replay_label_basis_uses_future_sample_count():
    distillation = [_row(hand_seed="1", state_signature="state-pos", action_signature="action-pos")]
    replay_index = build_replay_label_index(
        distillation,
        [_summary(0, delta=2.0, lcb=1.0, label="positive") | {"future_samples": "2048"}],
    )

    merged, _replacements = merge_rows(distillation, replay_index)

    assert merged[0]["local_replay_future_samples"] == 2048
    assert merged[0]["replay_label_basis"] == "mc2048_independent_replay"
    assert merged[0]["realized_delta_basis"] == "mc2048_independent_replay"
    assert _replacements[0]["local_replay_future_samples"] == 2048
    assert _replacements[0]["replay_label_basis"] == "mc2048_independent_replay"


def test_summary_rows_counts_replay_replacements_and_trainable_rows():
    merged = [
        _row(recommended_training_use="topk_confirm_replay_positive"),
        _row(hand_seed="2", recommended_training_use="topk_confirm_replay_negative"),
        _row(hand_seed="3", recommended_training_use="topk_confirm_replay_gray"),
        _row(hand_seed="4", recommended_training_use="topk_confirm_realized_positive"),
    ]
    replacements = merged[:3]

    summary = {row["metric"]: row["value"] for row in summary_rows(merged, replacements)}

    assert summary["replay_label_replacements"] == 3
    assert summary["replay_positive_replacements"] == 1
    assert summary["replay_negative_replacements"] == 1
    assert summary["replay_gray_replacements"] == 1
    assert summary["trainable_topk_confirm_fire_rows"] == 3


def test_summary_rows_counts_replay_basis_and_future_samples():
    merged = [
        _row(recommended_training_use="topk_confirm_replay_positive"),
        _row(hand_seed="2", recommended_training_use="topk_confirm_replay_negative"),
    ]
    replacements = [
        merged[0] | {"replay_label_basis": "mc512_independent_replay", "local_replay_future_samples": 512},
        merged[1] | {"replay_label_basis": "mc2048_independent_replay", "local_replay_future_samples": 2048},
    ]

    summary = {row["metric"]: row["value"] for row in summary_rows(merged, replacements)}

    assert summary["replay_label_basis.mc512_independent_replay"] == 1
    assert summary["replay_label_basis.mc2048_independent_replay"] == 1
    assert summary["local_replay_future_samples.512"] == 1
    assert summary["local_replay_future_samples.2048"] == 1


def test_read_jsonl_many_preserves_source_counts(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text('{"id":1}\n{"id":2}\n', encoding="utf-8")
    second.write_text('{"id":3}\n', encoding="utf-8")

    rows, sources = read_jsonl_many([first, second])

    assert [row["id"] for row in rows] == [1, 2, 3]
    assert sources == [
        {"path": str(first), "rows": 2},
        {"path": str(second), "rows": 1},
    ]
