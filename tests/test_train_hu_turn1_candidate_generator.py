import json
import os
import subprocess
import sys
from pathlib import Path

from ofc_regular.action_space import generate_turn_actions
from ofc_regular.hu_turn3_model import HuSklearnActionValueModel, load_hu_action_value_model
from ofc_regular.policy import action_to_json, board_to_json
from ofc_regular.state import Board
from ofc_regular.train_hu_turn1_candidate_generator import (
    baseline_delta_targets,
    build_training_matrix,
    evaluate_candidate_generator,
    infer_artifact_stage,
    main,
)


def _sample(sample_id: int, *, best_index: int = 1):
    board = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd"],
        bottom=["9c", "9d"],
    )
    opponent = Board.from_rows(
        top=["2h"],
        middle=["3h", "4h"],
        bottom=["7h", "8h"],
    )
    dealt = ["Qs", "Ah", "7d"]
    actions = generate_turn_actions(board, dealt)
    scored_actions = []
    for index, action in enumerate(actions[:6]):
        payload = action_to_json(board, action)
        payload["score"] = 10.0 if index == best_index else float(-index)
        scored_actions.append(payload)
    return {
        "schema": "hu_turn1_stage1_merged_teacher_v1",
        "phase": "hu_turn1_5card",
        "sample_id": sample_id,
        "hand_seed": 2026062400 + sample_id,
        "player": sample_id % 2,
        "seat": "first" if sample_id % 2 else "second",
        "label_source": "stage9f_p2_refinement" if sample_id % 3 else "stage9f_p2_runtime_relabel",
        "board": board_to_json(board),
        "opponent_board": board_to_json(opponent),
        "dead_cards": list(opponent.all_cards()),
        "dealt": dealt,
        "best_action": best_index,
        "score_gap": 10.0,
        "actions": scored_actions,
    }


def test_candidate_generator_infers_turn0_artifact_label():
    turn0 = _sample(1)
    turn0["schema"] = "hu_turn0_stage1_teacher_v1"
    turn0["phase"] = "hu_turn0_0card"

    assert infer_artifact_stage([turn0]) == "hu_turn0"
    assert infer_artifact_stage([_sample(2)]) == "hu_turn1"


def test_candidate_generator_builds_binary_action_labels():
    rows = [_sample(1, best_index=1), _sample(2, best_index=2)]

    features, labels, weights, offsets, regrets = build_training_matrix(
        rows,
        source_weights={"stage9f_p2_runtime_relabel": 2.0},
        args=type(
            "Args",
            (),
            {
                "accept_regret": 0.25,
                "gray_regret": 2.0,
                "positive_weight": 4.0,
                "gray_weight": 0.75,
                "negative_weight": 1.0,
                "hard_negative_regret": 10.0,
                "hard_negative_weight": 3.0,
            },
        )(),
    )

    assert features.shape[0] == labels.shape[0] == weights.shape[0] == regrets.shape[0]
    assert labels.sum() == 2
    assert len(offsets) == 2
    assert weights.max() > weights.min()


def test_candidate_generator_can_downweight_noisy_delta_rows():
    row = _sample(1, best_index=1)
    for index, action in enumerate(row["actions"]):
        action["delta_se_vs_baseline"] = 0.0 if index == 0 else 2.0
    args = type(
        "Args",
        (),
        {
            "accept_regret": 0.25,
            "gray_regret": 2.0,
            "positive_weight": 4.0,
            "gray_weight": 0.75,
            "negative_weight": 1.0,
            "hard_negative_regret": 10.0,
            "hard_negative_weight": 3.0,
            "delta_se_weight_floor": 1.0,
        },
    )()

    _features, _labels, weights, _offsets, _regrets = build_training_matrix(
        [row], source_weights={}, args=args
    )

    assert weights[0] > weights[1]


def test_candidate_generator_safe_lcb_target_uses_delta_uncertainty():
    row = _sample(1, best_index=1)
    for index, action in enumerate(row["actions"]):
        action["delta_vs_baseline"] = 2.0 if index == 1 else -1.0
        action["delta_se_vs_baseline"] = 0.5
    args = type(
        "Args",
        (),
        {
            "accept_regret": 0.25,
            "gray_regret": 2.0,
            "positive_weight": 10.0,
            "gray_weight": 0.1,
            "negative_weight": 1.0,
            "hard_negative_regret": 10.0,
            "hard_negative_weight": 3.0,
            "delta_se_weight_floor": 0.0,
            "classification_target": "safe_lcb196",
        },
    )()

    _features, labels, weights, _offsets, _regrets = build_training_matrix(
        [row], source_weights={}, args=args
    )

    assert labels.sum() == 1
    assert labels[1] == 1
    assert weights[1] == 10.0
    assert weights[0] == 3.0


def test_candidate_generator_training_cli_saves_runtime_loadable_model(tmp_path, monkeypatch):
    rows = [_sample(index, best_index=(index % 4) + 1) for index in range(20)]
    input_path = tmp_path / "teacher.jsonl"
    input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    model_path = tmp_path / "candidate.pkl"
    metrics_path = tmp_path / "metrics.json"
    holdout_path = tmp_path / "holdout.jsonl"

    monkeypatch.setattr(
        "sys.argv",
        [
            "train_hu_turn1_candidate_generator",
            "--input",
            str(input_path),
            "--model-output",
            str(model_path),
            "--metrics-output",
            str(metrics_path),
            "--holdout-output",
            str(holdout_path),
            "--model-type",
            "logistic",
            "--accept-regret",
            "0.25",
            "--gray-regret",
            "2.0",
        ],
    )
    main()

    model = load_hu_action_value_model(model_path)
    assert isinstance(model, HuSklearnActionValueModel)
    scores = model.predict_sample(rows[0])
    assert scores.shape[0] == len(rows[0]["actions"])
    assert ((0.0 <= scores) & (scores <= 1.0)).all()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["schema"] == "hu_turn1_candidate_generator_training_metrics_v1"
    assert metrics["holdout_output"] == str(holdout_path)
    assert len(holdout_path.read_text(encoding="utf-8").splitlines()) == metrics["holdout_samples"]
    assert metrics["holdout"]["top3_accepted_action_recall"] >= 0.0


def test_candidate_generator_accepts_explicit_validation_input(tmp_path, monkeypatch):
    train_rows = [_sample(index, best_index=(index % 4) + 1) for index in range(20)]
    validation_rows = [_sample(100 + index, best_index=(index % 4) + 1) for index in range(6)]
    input_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    input_path.write_text(
        "\n".join(json.dumps(row) for row in train_rows) + "\n",
        encoding="utf-8",
    )
    validation_path.write_text(
        "\n".join(json.dumps(row) for row in validation_rows) + "\n",
        encoding="utf-8",
    )
    model_path = tmp_path / "candidate.pkl"
    metrics_path = tmp_path / "metrics.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "train_hu_turn1_candidate_generator",
            "--input",
            str(input_path),
            "--validation-input",
            str(validation_path),
            "--model-output",
            str(model_path),
            "--metrics-output",
            str(metrics_path),
            "--model-type",
            "logistic",
        ],
    )
    main()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["split_strategy"] == "explicit_validation_input"
    assert metrics["validation_input"] == str(validation_path)
    assert metrics["raw_train_samples"] == 20
    assert metrics["holdout_samples"] == 6


def test_candidate_generator_regressor_cli_saves_runtime_loadable_model(tmp_path, monkeypatch):
    rows = [_sample(index, best_index=(index % 4) + 1) for index in range(20)]
    input_path = tmp_path / "teacher.jsonl"
    input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    model_path = tmp_path / "candidate_regressor.pkl"
    metrics_path = tmp_path / "metrics_regressor.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "train_hu_turn1_candidate_generator",
            "--input",
            str(input_path),
            "--model-output",
            str(model_path),
            "--metrics-output",
            str(metrics_path),
            "--model-type",
            "hgb_regressor",
            "--max-leaf-nodes",
            "3",
            "--max-iter",
            "20",
        ],
    )
    main()

    model = load_hu_action_value_model(model_path)
    assert isinstance(model, HuSklearnActionValueModel)
    scores = model.predict_sample(rows[0])
    assert scores.shape[0] == len(rows[0]["actions"])

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["training_objective"] == "score_regression"
    assert metrics["holdout"]["top3_accepted_action_recall"] >= 0.0


def test_candidate_generator_pairwise_cli_saves_runtime_loadable_model(tmp_path, monkeypatch):
    rows = [_sample(index, best_index=(index % 4) + 1) for index in range(20)]
    input_path = tmp_path / "teacher.jsonl"
    input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    model_path = tmp_path / "candidate_pairwise.pkl"
    metrics_path = tmp_path / "metrics_pairwise.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "train_hu_turn1_candidate_generator",
            "--input",
            str(input_path),
            "--model-output",
            str(model_path),
            "--metrics-output",
            str(metrics_path),
            "--model-type",
            "pairwise_logistic",
            "--pairwise-min-gap",
            "0.25",
            "--pairwise-max-pairs-per-sample",
            "16",
        ],
    )
    main()

    model = load_hu_action_value_model(model_path)
    assert isinstance(model, HuSklearnActionValueModel)
    scores = model.predict_sample(rows[0])
    assert scores.shape[0] == len(rows[0]["actions"])

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["training_objective"] == "pairwise_ranking"
    assert metrics["pairwise_train_pairs"] > 0
    assert metrics["holdout"]["top3_accepted_action_recall"] >= 0.0


def test_candidate_generator_pairwise_model_loads_in_fresh_process(tmp_path, monkeypatch):
    rows = [_sample(index, best_index=(index % 4) + 1) for index in range(20)]
    input_path = tmp_path / "teacher.jsonl"
    input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    model_path = tmp_path / "candidate_pairwise.pkl"
    metrics_path = tmp_path / "metrics_pairwise.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "train_hu_turn1_candidate_generator",
            "--input",
            str(input_path),
            "--model-output",
            str(model_path),
            "--metrics-output",
            str(metrics_path),
            "--model-type",
            "pairwise_logistic",
            "--pairwise-min-gap",
            "0.25",
            "--pairwise-max-pairs-per-sample",
            "16",
        ],
    )
    main()

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from ofc_regular.hu_turn3_model import load_hu_action_value_model; "
                f"model = load_hu_action_value_model(r'{model_path}'); "
                "print(type(model).__name__)"
            ),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "HuSklearnActionValueModel" in result.stdout


def test_candidate_generator_eval_reports_topk_regret_with_constant_estimator():
    class ConstantEstimator:
        def predict_proba(self, features):
            import numpy as np

            return np.tile(np.array([[0.5, 0.5]]), (features.shape[0], 1))

        classes_ = [0, 1]

    rows = [_sample(1, best_index=1), _sample(2, best_index=2)]
    result = evaluate_candidate_generator(ConstantEstimator(), rows, accept_regret=0.25)

    assert result["samples"] == 2
    assert result["actions"] > 0
    assert "top3_best_avg_regret" in result


def test_baseline_delta_targets_remove_each_states_baseline_score():
    rows = [_sample(1, best_index=1), _sample(2, best_index=2)]
    for row in rows:
        row["baseline_action_row_index"] = 3
    args = type(
        "Args",
        (),
        {
            "accept_regret": 0.25,
            "gray_regret": 2.0,
            "positive_weight": 4.0,
            "gray_weight": 0.75,
            "negative_weight": 1.0,
            "hard_negative_regret": 10.0,
            "hard_negative_weight": 3.0,
        },
    )()
    _features, _labels, _weights, offsets, _regrets, scores = build_training_matrix(
        rows,
        source_weights={},
        args=args,
        include_scores=True,
    )

    targets = baseline_delta_targets(rows, scores, offsets)

    for start, end in offsets:
        assert targets[start + 3] == 0.0
        assert list(targets[start:end]) == list(scores[start:end] - scores[start + 3])
