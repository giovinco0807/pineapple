import json

import pytest

from ofc_regular.action_key import action_key
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.policy import action_to_json, board_to_json
from ofc_regular.state import Board
from ofc_regular.train_hu_turn1_safe_override_selector import (
    build_training_data,
    deterministic_split,
    label_for_row,
    parse_args,
    row_to_feature_vector,
    stage10_mc32_candidate_delta,
    train_selector,
)


def test_hu_turn1_safe_override_cli_accepts_multiple_inputs(tmp_path):
    args = parse_args(
        [
            "--input",
            str(tmp_path / "first.jsonl"),
            "--input",
            str(tmp_path / "second.jsonl"),
            "--output-dir",
            str(tmp_path / "out"),
            "--model-output",
            str(tmp_path / "model.pkl"),
        ]
    )

    assert args.input == [tmp_path / "first.jsonl", tmp_path / "second.jsonl"]


def _row(delta: float, *, seed: int, candidate_index: int = 1, fallback_index: int = 0):
    board = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd"],
        bottom=["9c", "9d"],
    )
    opponent = Board.from_rows(
        top=["2h"],
        middle=["3h", "4h"],
        bottom=["7h", "8h", *(["5c", "6c"] if seed % 2 == 0 else [])],
    )
    dealt = ["Qs", "Ah", "7d"]
    actions = generate_turn_actions(board, dealt)
    seat = "first" if seed % 2 else "second"
    observation = ActorObservation(
        hero_board=board,
        opponent_public_board=opponent,
        dealt_cards=tuple(dealt),
        hero_private_discards=(),
        seat=seat,
        street="T1",
        to_act_order=seat,
    )
    return {
        "hand_seed": seed,
        "paired_index": seed,
        "seat_swap": "ab" if seed % 2 else "ba",
        "seat": seat,
        "override_fired": True,
        "realized_delta_valid": True,
        "realized_candidate_seat_delta": delta,
        "hu_turn1_realized_delta": delta,
        "hu_turn1_predicted_margin": 1.0 + seed * 0.01,
        "candidate_score": 2.0,
        "fallback_score": 0.5,
        "action_count": len(actions),
        "hero_board": board_to_json(board),
        "opponent_board": board_to_json(opponent),
        "dead_cards": list(observation.legacy_dead_cards()),
        "visible_dead_cards": list(observation.legacy_dead_cards()),
        "hero_private_discards": [],
        "policy_observation": observation.to_dict(),
        "cards_to_place": dealt,
        "candidate_action_index": candidate_index,
        "fallback_action_index": fallback_index,
        "hu_turn1_action": action_to_json(board, actions[candidate_index]),
        "baseline_action": action_to_json(board, actions[fallback_index]),
        "hu_turn1_safe_override_label": "positive" if delta > 0 else "hard_negative",
        "hu_turn1_safe_override_label_id": 1 if delta > 0 else 0,
    }


def _relabel_row(
    regret: float,
    *,
    seed: int,
    delta: float = 0.0,
    candidate_index: int = 1,
    fallback_index: int = 0,
):
    row = _row(delta, seed=seed, candidate_index=candidate_index, fallback_index=fallback_index)
    row["schema"] = "hu_turn1_stage1_refinement_relabel_v1"
    row["board"] = row.pop("hero_board")
    row["dealt"] = row.pop("cards_to_place")
    row["runtime_candidate_action"] = row.pop("hu_turn1_action")
    row["runtime_baseline_action"] = row.pop("baseline_action")
    row["runtime_candidate_score"] = row.pop("candidate_score")
    row["runtime_baseline_score"] = row.pop("fallback_score")
    row["runtime_predicted_margin"] = row.pop("hu_turn1_predicted_margin")
    row["runtime_candidate_action_index"] = row.pop("candidate_action_index")
    row["runtime_baseline_action_index"] = row.pop("fallback_action_index")
    row["relabel_compare"] = {
        "best_action_changed": regret > 0.0,
        "source_best_new_regret": regret,
        "source_best_new_rank": 1 if regret <= 0.25 else 5,
    }
    return row


def test_hu_turn1_safe_override_feature_vector_has_expected_meta_size():
    row = _row(6.0, seed=1)

    vector = row_to_feature_vector(row, feature_mode="delta_plus_meta")

    assert vector.ndim == 1
    assert vector.shape[0] > 100
    assert vector[-6] == row["hu_turn1_predicted_margin"]


def test_hu_turn1_safe_override_runtime_meta_feature_mode_appends_confirm_stats():
    row = _row(6.0, seed=1)
    row["stage_a_delta"] = 1.5
    row["stage_a_delta_se"] = 2.0
    row["confirm_delta"] = 3.0
    row["confirm_delta_se"] = 1.5
    row["confirm_delta_count"] = 8

    base = row_to_feature_vector(row, feature_mode="candidate_delta_plus_meta")
    vector = row_to_feature_vector(row, feature_mode="candidate_delta_plus_runtime_meta")

    assert vector.shape[0] == base.shape[0] + 7
    assert vector[-7:].tolist() == [1.5, 2.0, 3.0, 1.5, 8.0, 1.5, 2.0]


def test_hu_turn1_safe_override_selector_training_smoke(tmp_path):
    rows = [_row(6.0, seed=i) for i in range(1, 9)]
    rows += [_row(-6.0, seed=i) for i in range(9, 17)]
    input_path = tmp_path / "rows.jsonl"
    input_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    data = build_training_data(rows, feature_mode="meta_only")

    summary = train_selector(
        data,
        output_dir=tmp_path / "training",
        model_output=tmp_path / "model.pkl",
    )

    assert summary["rows"] == 16
    assert summary["positives"] == 8
    assert summary["negatives"] == 8
    assert (tmp_path / "model.pkl").exists()
    assert (tmp_path / "training" / "safe_selector_metrics.csv").exists()
    assert (tmp_path / "training" / "safe_selector_thresholds.csv").exists()


def test_hu_turn1_safe_override_training_can_include_non_fired_candidates():
    fired = _row(6.0, seed=1)
    gated = _row(-4.0, seed=2, candidate_index=2, fallback_index=0)
    gated["override_fired"] = False
    gated["no_override_reason"] = "below_confirm_delta"
    gated["final_action"] = gated["baseline_action"]
    gated["final_action_index"] = gated["fallback_action_index"]

    default_data = build_training_data([fired, gated], feature_mode="meta_only")
    expanded_data = build_training_data(
        [fired, gated],
        feature_mode="meta_only",
        include_non_fired_candidates=True,
    )

    assert len(default_data.rows) == 1
    assert len(expanded_data.rows) == 2
    assert expanded_data.labels.tolist() == [1, 0]


def test_hu_turn1_safe_override_split_is_label_stratified():
    rows = [_row(6.0, seed=i) for i in range(1, 38)]
    rows += [_row(-6.0, seed=i) for i in range(38, 150)]
    data = build_training_data(rows, feature_mode="meta_only")

    split = deterministic_split(data.rows, data.labels)

    for split_id in (0, 1, 2):
        mask = split == split_id
        assert data.labels[mask].sum() > 0
        assert (data.labels[mask] == 0).sum() > 0


def test_hu_turn1_safe_override_can_train_from_teacher_regret_relabel_rows():
    rows = [_relabel_row(0.0, seed=i, delta=-6.0) for i in range(1, 8)]
    rows += [_relabel_row(6.0, seed=i, delta=6.0) for i in range(8, 18)]

    data = build_training_data(
        rows,
        feature_mode="meta_only",
        label_mode="teacher_regret",
        teacher_accept_regret=0.25,
        teacher_gray_regret=1.0,
    )

    assert data.label_mode == "teacher_regret"
    assert data.labels.sum() == 7
    assert len(data.rows) == 17
    assert data.regrets.max() == 6.0
    vector = row_to_feature_vector(rows[0], feature_mode="delta_plus_meta")
    assert vector[-6] == rows[0]["runtime_predicted_margin"]


def test_hu_turn1_safe_override_accept_gray_delta_label_mode():
    rows = [
        _row(3.0, seed=1),
        _row(0.5, seed=2),
        _row(-6.0, seed=3),
    ]

    data = build_training_data(
        rows,
        feature_mode="meta_only",
        label_mode="accept_gray_delta",
        accept_delta=2.0,
        hard_negative_delta=-5.0,
        gray_weight=0.2,
    )

    assert data.labels.tolist() == [1, 0, 0]
    assert data.weights.tolist() == [1.0, 0.2, 2.0]
    assert data.deltas.tolist() == [3.0, 0.5, -6.0]


def test_hu_turn1_safe_override_stage10_mc32_delta_label_mode_prefers_relabel_delta():
    row = _row(-10.0, seed=1)
    row["stage10_mc32_candidate_delta"] = 3.5

    label, weight, delta, _regret, is_gray = label_for_row(
        row,
        label_mode="stage10_mc32_delta",
        teacher_accept_regret=0.25,
        teacher_gray_regret=1.0,
        accept_delta=2.0,
        hard_negative_delta=-5.0,
        gray_weight=0.1,
    )

    assert label == 1
    assert weight == 1.0
    assert delta == 3.5
    assert is_gray is False


def test_hu_turn1_safe_override_stage10_mc32_delta_recovers_from_relabel_actions():
    row = _relabel_row(-10.0, seed=1, candidate_index=1, fallback_index=0)
    row["actions"] = [
        {
            **row["runtime_baseline_action"],
            "action_index": row["runtime_baseline_action_index"],
            "score": -3.0,
        },
        {
            **row["runtime_candidate_action"],
            "action_index": row["runtime_candidate_action_index"],
            "score": 2.5,
        },
    ]

    label, weight, delta, _regret, is_gray = label_for_row(
        row,
        label_mode="stage10_mc32_delta",
        teacher_accept_regret=0.25,
        teacher_gray_regret=1.0,
        accept_delta=2.0,
        hard_negative_delta=-5.0,
        gray_weight=0.1,
    )

    assert stage10_mc32_candidate_delta(row) == 5.5
    assert label == 1
    assert weight == 1.0
    assert delta == 5.5
    assert is_gray is False


def _stage10_semantic_row():
    row = _relabel_row(-10.0, seed=1, candidate_index=1, fallback_index=0)
    board = Board.from_rows(**row["board"])
    actions = generate_turn_actions(board, row["dealt"])
    row["runtime_candidate_action_key"] = action_key(actions[1]).to_token()
    row["runtime_baseline_action_key"] = action_key(actions[0]).to_token()
    row["actions"] = [
        {
            **row["runtime_baseline_action"],
            "action_index": 0,
            "original_index": 0,
            "canonical_action_key": action_key(actions[0]).to_token(),
            "score": -3.0,
            "ev": -3.0,
        },
        {
            **row["runtime_candidate_action"],
            "action_index": 1,
            "original_index": 1,
            "canonical_action_key": action_key(actions[1]).to_token(),
            "score": 2.5,
            "ev": 2.5,
        },
    ]
    return row, actions


def test_hu_turn1_safe_override_stage10_mc32_rejects_corrupt_saved_index():
    row, _actions = _stage10_semantic_row()
    row["runtime_candidate_action_index"] = 2

    with pytest.raises(ValueError, match="candidate saved action index disagrees"):
        stage10_mc32_candidate_delta(row)


def test_hu_turn1_safe_override_stage10_mc32_rejects_payload_key_disagreement():
    row, actions = _stage10_semantic_row()
    row["runtime_candidate_action"] = action_to_json(
        Board.from_rows(**row["board"]), actions[2]
    )

    with pytest.raises(ValueError, match="candidate semantic action evidence disagrees"):
        stage10_mc32_candidate_delta(row)


def test_hu_turn1_safe_override_stage10_mc32_rejects_corrupt_score_row_key():
    row, actions = _stage10_semantic_row()
    row["actions"][1]["canonical_action_key"] = action_key(actions[2]).to_token()

    with pytest.raises(ValueError, match=r"actions\[1\] semantic action evidence disagrees"):
        stage10_mc32_candidate_delta(row)


def test_hu_turn1_safe_override_stage10_mc32_delta_training_labels():
    positive = _row(-10.0, seed=1)
    positive["stage10_mc32_candidate_delta"] = 1.25
    negative = _row(10.0, seed=2)
    negative["stage10_mc32_candidate_delta"] = -2.5
    gray = _row(10.0, seed=3)
    gray["stage10_mc32_candidate_delta"] = 0.0

    data = build_training_data(
        [positive, negative, gray],
        feature_mode="meta_only",
        label_mode="stage10_mc32_delta",
        gray_weight=0.1,
    )

    assert data.labels.tolist() == [1, 0, 0]
    assert data.weights.tolist() == [1.0, 2.0, 0.1]
    assert data.deltas.tolist() == [1.25, -2.5, 0.0]


def test_hu_turn1_safe_override_high_mc_delta_thresholded_labels():
    positive = _row(-10.0, seed=1)
    positive["stage10_mc32_candidate_delta"] = 1.5
    negative = _row(10.0, seed=2)
    negative["stage10_mc32_candidate_delta"] = -0.25
    gray = _row(10.0, seed=3)
    gray["stage10_mc32_candidate_delta"] = 0.5

    data = build_training_data(
        [positive, negative, gray],
        feature_mode="meta_only",
        label_mode="high_mc_delta_thresholded",
        accept_delta=1.0,
        hard_negative_delta=0.0,
        gray_weight=0.2,
    )

    assert data.labels.tolist() == [1, 0, 0]
    assert data.weights.tolist() == [1.0, 2.0, 0.2]
    assert data.deltas.tolist() == [1.5, -0.25, 0.5]


def test_hu_turn1_safe_override_hybrid_label_mode_uses_row_schema():
    teacher_good = _relabel_row(1.0, seed=1, delta=-10.0)
    teacher_bad = _relabel_row(8.0, seed=2, delta=10.0)
    runtime_good = _row(3.0, seed=3)
    runtime_gray = _row(0.5, seed=4)

    data = build_training_data(
        [teacher_good, teacher_bad, runtime_good, runtime_gray],
        feature_mode="meta_only",
        label_mode="teacher_regret_or_accept_gray_delta",
        teacher_accept_regret=2.0,
        teacher_gray_regret=5.0,
        accept_delta=2.0,
        hard_negative_delta=-5.0,
        gray_weight=0.25,
    )

    assert data.labels.tolist() == [1, 0, 1, 0]
    assert data.weights.tolist() == [1.0, 2.0, 1.0, 0.25]
