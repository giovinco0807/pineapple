from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from sklearn.dummy import DummyClassifier

from ofc_regular.action_space import generate_turn_actions
from ofc_regular.hu_turn3_gate_model import (
    GATE_FEATURE_NAMES,
    HuTurn3GateModel,
    decision_gate_features,
    load_hu_turn3_gate_model,
)
from ofc_regular.hu_turn3_model import hu_policy_sample
from ofc_regular.state import Board


def _sample():
    hero = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd", "6c", "8s"],
        bottom=["9c", "9d", "9s", "Kc"],
    )
    opponent = Board.from_rows(
        top=["2h"],
        middle=["3h", "4h", "5h", "6h"],
        bottom=["7h", "8h", "Th", "Jh"],
    )
    dealt = ["Qs", "Ah", "7d"]
    actions = generate_turn_actions(hero, dealt)
    return hu_policy_sample(
        hero,
        dealt,
        actions,
        opponent_board=opponent,
        dead_cards=opponent.all_cards(),
        seat="second",
        to_act_order="second",
    )


def test_hu_turn3_gate_features_include_margin_and_self_regret():
    sample = _sample()
    hu_predictions = np.zeros(len(sample["actions"]), dtype=np.float64)
    self_predictions = np.zeros(len(sample["actions"]), dtype=np.float64)
    hu_predictions[-1] = 10.0
    hu_predictions[0] = 2.0
    self_predictions[-1] = 1.0
    self_predictions[0] = 4.0

    features = decision_gate_features(
        sample,
        chosen_index=len(sample["actions"]) - 1,
        baseline_index=0,
        hu_predictions=hu_predictions,
        self_predictions=self_predictions,
    )

    assert features.shape == (len(GATE_FEATURE_NAMES),)
    assert features[0] == 8.0
    assert features[1] == 3.0
    assert features[8] == 1.0
    assert features[9] == 1.0


def test_hu_turn3_gate_model_round_trips():
    estimator = DummyClassifier(strategy="constant", constant=1)
    estimator.fit(np.zeros((2, len(GATE_FEATURE_NAMES))), np.ones(2, dtype=np.int64))
    model = HuTurn3GateModel(estimator=estimator)
    sample = _sample()
    predictions = np.zeros(len(sample["actions"]), dtype=np.float64)

    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "gate.pkl"
        model.save(path)
        loaded = load_hu_turn3_gate_model(path)

    probability = loaded.predict_accept_probability(
        sample,
        chosen_index=0,
        baseline_index=1,
        hu_predictions=predictions,
        self_predictions=predictions,
    )
    assert probability == 1.0
