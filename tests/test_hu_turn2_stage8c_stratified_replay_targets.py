from ofc_regular.extract_hu_turn2_stage8c_stratified_replay_targets import (
    _selected_rows,
    extract_stratified_targets,
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


def _ranker_row(source, **overrides):
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
        "ranker": "policy",
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


def test_selected_rows_keeps_runtime_rankers_and_skips_confirm_by_default():
    source = _distillation_row()
    rows = [
        _ranker_row(source, ranker="confirm_delta", rank="1"),
        _ranker_row(source, ranker="policy", rank="2"),
        _ranker_row(source, ranker="fire_probability", rank="3"),
    ]

    selected, audit = _selected_rows(
        rows,
        rankers=None,
        splits={"test"},
        seats=None,
        max_rank=0,
        allow_non_runtime_rankers=False,
    )

    assert [row["ranker"] for row in selected] == ["policy", "fire_probability"]
    assert audit["selection_counts"]["skipped_non_runtime_ranker"] == 1
    assert audit["skipped_non_runtime_by_ranker"] == {"confirm_delta": 1}


def test_stratified_targets_sample_across_score_buckets():
    rows = []
    ranker_rows = []
    for index in range(6):
        source = _distillation_row(
            hand_seed=str(index),
            state_signature=f"state-{index}",
            action_signature=f"action-{index}",
            candidate_index=index + 1,
        )
        rows.append(source)
        ranker_rows.append(_ranker_row(source, rank=str(index + 1), ranker_score=str(100 - index)))
    index = {row_key(row): row for row in rows}

    targets, audit, manifest = extract_stratified_targets(
        index,
        ranker_rows,
        buckets=3,
        targets_per_bucket=1,
        include_observed=False,
    )

    assert len(targets) == 3
    assert [target["ranker_score_bucket"] for target in targets] == [0, 1, 2]
    assert all(target["schema"] == "hu_turn2_stage8c_stratified_replay_target_v1" for target in targets)
    assert all(target["observed_performance_claim"] == "No" for target in targets)
    assert manifest["counts"]["targets"] == 3
    assert sum(1 for row in audit if row["status"] == "target") == 3


def test_stratified_targets_skip_observed_and_duplicates():
    unknown = _distillation_row(hand_seed="1", state_signature="unknown", action_signature="a1", candidate_index=1)
    observed = _distillation_row(
        hand_seed="2",
        state_signature="observed",
        action_signature="a2",
        candidate_index=2,
        realized_delta_observed=True,
        realized_delta=5.0,
    )
    distillation_index = {row_key(row): row for row in [unknown, observed]}
    ranker_rows = [
        _ranker_row(unknown, rank="1", ranker_score="2"),
        _ranker_row(unknown, rank="2", ranker_score="1.5"),
        _ranker_row(observed, rank="3", ranker_score="1"),
    ]

    targets, _audit, manifest = extract_stratified_targets(
        distillation_index,
        ranker_rows,
        buckets=1,
        targets_per_bucket=10,
        include_observed=False,
    )

    assert [target["hand_seed"] for target in targets] == ["1"]
    assert manifest["counts"]["skipped_duplicate_target"] == 1
    assert manifest["counts"]["skipped_observed_delta"] == 1


def test_stratified_targets_report_replay_blockers():
    source = _distillation_row(dead_cards=[], hand_seed="1")
    distillation_index = {row_key(source): source}

    targets, _audit, manifest = extract_stratified_targets(
        distillation_index,
        [_ranker_row(source)],
        buckets=1,
        targets_per_bucket=1,
        include_observed=False,
    )

    assert len(targets) == 1
    assert targets[0]["replay_ready"] is False
    assert targets[0]["replay_blocker"] == "dead_cards"
    assert manifest["counts"]["replay_blocked"] == 1
