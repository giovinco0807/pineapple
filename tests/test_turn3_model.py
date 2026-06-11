import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor

from ofc_regular.turn3_model import (
    FEATURE_DIM,
    SklearnActionValueModel,
    TorchActionValueModel,
    Turn3RidgeModel,
    _build_torch_mlp,
    evaluate_model,
    load_action_value_model,
    read_teacher_samples,
    sample_to_matrix,
    split_samples,
    train_ridge_model,
)


def _sample(score_a=3.0, score_b=1.0):
    return {
        "sample_id": 0,
        "rule_set": "regular",
        "phase": "turn3_9card",
        "fl_ev": 12.196164,
        "board": {
            "top": ["Qh"],
            "middle": ["Kh", "Kd", "6c", "8s"],
            "bottom": ["9c", "9d", "9s", "Kc"],
        },
        "dealt": ["Qs", "Ah", "7d"],
        "best_action": 0,
        "score_gap": score_a - score_b,
        "actions": [
            {
                "placements": [["Qs", "top"], ["Ah", "top"]],
                "discards": ["7d"],
                "score": score_a,
                "future_count": 4,
                "next_board": {
                    "top": ["Qh", "Qs", "Ah"],
                    "middle": ["Kh", "Kd", "6c", "8s"],
                    "bottom": ["9c", "9d", "9s", "Kc"],
                },
            },
            {
                "placements": [["Qs", "bottom"], ["Ah", "top"]],
                "discards": ["7d"],
                "score": score_b,
                "future_count": 4,
                "next_board": {
                    "top": ["Qh", "Ah"],
                    "middle": ["Kh", "Kd", "6c", "8s"],
                    "bottom": ["9c", "9d", "9s", "Kc", "Qs"],
                },
            },
        ],
    }


def test_sample_to_matrix_encodes_actions():
    features, targets = sample_to_matrix(_sample())
    assert features.shape == (2, FEATURE_DIM)
    assert targets.tolist() == [3.0, 1.0]
    assert np.count_nonzero(features[0]) > 0


def test_train_ridge_model_predicts_and_round_trips():
    samples = [_sample(), _sample(4.0, 0.5), _sample(2.0, -1.0)]
    model = train_ridge_model(samples, l2=1.0)
    metrics = evaluate_model(model, samples)
    assert metrics["top1_accuracy"] == 1.0
    assert metrics["top3_accuracy"] == 1.0
    assert model.choose_action_index(samples[0]) == 0

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "turn3.npz"
        model.save(path)
        loaded = Turn3RidgeModel.load(path)
        assert loaded.choose_action_index(samples[0]) == 0
        generic = load_action_value_model(path)
        assert generic.choose_action_index(samples[0]) == 0


def test_sklearn_model_round_trips_and_loads_generically():
    samples = [_sample(), _sample(4.0, 0.5), _sample(2.0, -1.0)]
    features, targets = sample_to_matrix(samples[0])
    for sample in samples[1:]:
        x, y = sample_to_matrix(sample)
        features = np.vstack([features, x])
        targets = np.concatenate([targets, y])
    estimator = DecisionTreeRegressor(max_depth=2, random_state=1).fit(features, targets)
    model = SklearnActionValueModel(estimator=estimator)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "turn3.pkl"
        model.save(path)
        loaded = load_action_value_model(path)

    assert loaded.choose_action_index(samples[0]) == 0


def test_torch_model_round_trips_and_loads_generically():
    torch = pytest.importorskip("torch")
    net = _build_torch_mlp(torch, FEATURE_DIM, (8,), 0.0)
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.zero_()
    model = TorchActionValueModel(
        state_dict={key: value.detach().cpu().clone() for key, value in net.state_dict().items()},
        hidden_layer_sizes=(8,),
        feature_mean=np.zeros(FEATURE_DIM, dtype=np.float32),
        feature_scale=np.ones(FEATURE_DIM, dtype=np.float32),
        target_mean=0.0,
        target_scale=1.0,
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "turn3.pt"
        model.save(path)
        loaded = load_action_value_model(path)

    assert loaded.choose_action_index(_sample()) == 0


def test_split_samples_keeps_holdout_by_sample():
    samples = [_sample() for _ in range(10)]
    train, holdout = split_samples(samples, holdout_fraction=0.2, seed=1)
    assert len(train) == 8
    assert len(holdout) == 2


def test_read_teacher_samples_accepts_utf8_bom():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "teacher.jsonl"
        path.write_text(json.dumps(_sample()) + "\n", encoding="utf-8-sig")
        samples = read_teacher_samples(path)

    assert len(samples) == 1
    assert samples[0]["phase"] == "turn3_9card"
