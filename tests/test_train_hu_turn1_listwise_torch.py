import json

import pytest

from ofc_regular.action_space import generate_turn_actions
from ofc_regular.hu_turn3_model import HuTorchActionValueModel, load_hu_action_value_model
from ofc_regular.policy import action_to_json, board_to_json
from ofc_regular.state import Board
from ofc_regular.train_hu_turn1_listwise_torch import main


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
        "hand_seed": 2026062600 + sample_id,
        "seat": "first" if sample_id % 2 else "second",
        "label_source": "test",
        "board": board_to_json(board),
        "opponent_board": board_to_json(opponent),
        "dead_cards": list(opponent.all_cards()),
        "dealt": dealt,
        "best_action": best_index,
        "score_gap": 10.0,
        "actions": scored_actions,
    }


def test_hu_turn1_listwise_torch_training_smoke(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    rows = [_sample(index, best_index=(index % 4) + 1) for index in range(20)]
    input_path = tmp_path / "teacher.jsonl"
    input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    model_path = tmp_path / "listwise.pt"
    metrics_path = tmp_path / "metrics.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "train_hu_turn1_listwise_torch",
            "--input",
            str(input_path),
            "--model-output",
            str(model_path),
            "--metrics-output",
            str(metrics_path),
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--hidden-sizes",
            "16",
        ],
    )
    main()

    model = load_hu_action_value_model(model_path)
    assert isinstance(model, HuTorchActionValueModel)
    scores = model.predict_sample(rows[0])
    assert scores.shape[0] == len(rows[0]["actions"])

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["schema"] == "hu_turn1_listwise_torch_training_metrics_v1"
    assert metrics["training_objective"] == "state_listwise_softmax"
    assert metrics["holdout"]["top3_accepted_action_recall"] >= 0.0


def test_hu_turn1_listwise_torch_early_stopping_restores_best(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    rows = [_sample(index, best_index=(index % 4) + 1) for index in range(20)]
    input_path = tmp_path / "teacher.jsonl"
    input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    model_path = tmp_path / "listwise_early.pt"
    metrics_path = tmp_path / "metrics_early.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "train_hu_turn1_listwise_torch",
            "--input",
            str(input_path),
            "--model-output",
            str(model_path),
            "--metrics-output",
            str(metrics_path),
            "--epochs",
            "5",
            "--batch-size",
            "4",
            "--hidden-sizes",
            "16",
            "--early-stopping-patience",
            "1",
            "--early-stopping-min-delta",
            "1000000000",
        ],
    )
    main()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["epochs_trained"] == 2
    assert metrics["early_stopping"]["best_epoch"] == 1
    assert metrics["early_stopping"]["stopped_early"] is True
    assert metrics["early_stopping"]["restored_best_state"] is True
    assert len(metrics["losses"]) == 2
    assert "validation_loss" in metrics["losses"][0]


def test_hu_turn0_listwise_torch_accepts_explicit_validation(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    train_rows = [_sample(index, best_index=(index % 4) + 1) for index in range(12)]
    validation_rows = [_sample(100 + index, best_index=(index % 4) + 1) for index in range(4)]
    for row in train_rows + validation_rows:
        row["schema"] = "hu_turn0_stage1_teacher_v1"
        row["phase"] = "hu_turn0_0card"
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
    model_path = tmp_path / "listwise_t0.pt"
    metrics_path = tmp_path / "metrics_t0.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "train_hu_turn1_listwise_torch",
            "--input",
            str(input_path),
            "--validation-input",
            str(validation_path),
            "--model-output",
            str(model_path),
            "--metrics-output",
            str(metrics_path),
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--hidden-sizes",
            "16",
            "--early-stopping-patience",
            "1",
        ],
    )
    main()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["schema"] == "hu_turn0_listwise_torch_training_metrics_v1"
    assert metrics["artifact_stage"] == "hu_turn0"
    assert metrics["split_strategy"] == "explicit_validation_input"
    assert metrics["raw_train_samples"] == 12
    assert metrics["holdout_samples"] == 4
