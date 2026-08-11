from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from ofc_regular.action_key import action_key
from ofc_regular.action_space import Action, generate_turn_actions
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m4_t1_policy import (
    HU_M4_T1_DECISION_SCHEMA,
    HuM4T1SafetyModel,
    HuM4T1SelectiveOverridePolicy,
    build_hu_m4_t1_safety_features,
)
from ofc_regular.state import Board


class _SemanticBaseline:
    def __init__(self, choose: str = "max") -> None:
        self.choose = choose
        self.observations: list[ActorObservation] = []
        self.kwargs: list[dict] = []

    def choose_action_observation(self, observation, **kwargs):
        self.observations.append(observation)
        self.kwargs.append(kwargs)
        actions = generate_turn_actions(observation.hero_board, observation.dealt_cards)
        selector = min if self.choose == "min" else max
        return selector(actions, key=lambda action: action_key(action).sort_key())


class _TieActionValueModel:
    def __init__(self) -> None:
        self.samples: list[dict] = []

    def predict_sample(self, sample):
        self.samples.append(sample)
        return np.zeros(len(sample["actions"]), dtype=np.float64)


class _BadActionValueModel:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def predict_sample(self, sample):
        if self.mode == "raise":
            raise RuntimeError("candidate exploded")
        if self.mode == "shape":
            return np.zeros(len(sample["actions"]) + 1)
        values = np.zeros(len(sample["actions"]))
        values[0] = np.nan
        return values


class _BaselineAwareActionValueModel:
    def __init__(self) -> None:
        self.calls: list[tuple[dict, int]] = []

    def predict_sample(self, sample):
        raise AssertionError("baseline-aware API must be preferred")

    def predict_sample_with_baseline(self, sample, *, baseline_index):
        self.calls.append((sample, baseline_index))
        values = np.zeros(len(sample["actions"]), dtype=np.float64)
        values[baseline_index] = 0.0
        return values


class _SafetyProbability:
    def __init__(self, probability: float = 1.0, *, raises: bool = False) -> None:
        self.probability = probability
        self.raises = raises
        self.rows: list[np.ndarray] = []

    def predict_probability(self, features):
        self.rows.append(np.asarray(features).copy())
        if self.raises:
            raise RuntimeError("selector exploded")
        return self.probability


class _ConstantEstimator:
    classes_ = np.asarray([0, 1])

    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, matrix):
        count = np.asarray(matrix).shape[0]
        return np.tile([1.0 - self.probability, self.probability], (count, 1))


def _t1_second_observation(*, dealt=("8c", "9c", "Tc")) -> ActorObservation:
    return ActorObservation(
        hero_board=Board.from_rows(
            top=["Ah"],
            middle=["Kd"],
            bottom=["2s", "3s", "9s"],
        ),
        opponent_public_board=Board.from_rows(
            top=["Qh"],
            middle=["Jd", "Td"],
            bottom=["4s", "5s", "6s", "7s"],
        ),
        dealt_cards=dealt,
        hero_private_discards=(),
        seat="second",
        street="T1",
        to_act_order="second",
    )


def _t1_first_observation() -> ActorObservation:
    return ActorObservation(
        hero_board=Board.from_rows(
            top=["Ah"],
            middle=["Kd"],
            bottom=["2s", "3s", "9s"],
        ),
        opponent_public_board=Board.from_rows(
            top=["Qh"],
            middle=["Jd"],
            bottom=["4s", "5s", "6s"],
        ),
        dealt_cards=("8c", "9c", "Tc"),
        hero_private_discards=(),
        seat="first",
        street="T1",
        to_act_order="first",
    )


def _policy(
    baseline,
    *,
    action_model=None,
    safety_model=None,
    threshold=0.5,
    log=None,
    **kwargs,
):
    return HuM4T1SelectiveOverridePolicy(
        baseline,
        action_value_model=action_model or _TieActionValueModel(),
        safety_model=safety_model or _SafetyProbability(),
        safety_probability_threshold=threshold,
        decision_log=log,
        **kwargs,
    )


def test_second_seat_forced_fire_uses_actionkey_tie_and_logs_safe_observation():
    observation = _t1_second_observation()
    baseline = _SemanticBaseline("max")
    action_model = _TieActionValueModel()
    safety = _SafetyProbability(0.9)
    log: list[dict] = []
    policy = _policy(
        baseline,
        action_model=action_model,
        safety_model=safety,
        threshold=0.8,
        log=log,
    )

    selected = policy.choose_action_observation(
        observation, hand_id=7, game_id="g", decision_seed=11
    )

    actions = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    expected = min(actions, key=lambda action: action_key(action).sort_key())
    assert action_key(selected) == action_key(expected)
    assert action_key(selected) != action_key(baseline.choose_action_observation(observation))
    assert len(action_model.samples) == 1
    assert safety.rows and safety.rows[0].ndim == 1
    assert len(log) == 1
    row = log[0]
    assert row["schema"] == HU_M4_T1_DECISION_SCHEMA
    assert row["override_fired"] is True
    assert row["nonfire_reason"] == ""
    assert row["policy_observation"] == observation.to_dict()
    assert row["observation_fingerprint"] == observation.fingerprint()
    assert row["candidate_action_key"] == action_key(expected).to_token()
    assert row["final_action_key"] == row["candidate_action_key"]
    assert row["baseline_action_key"] != row["candidate_action_key"]
    assert row["hand_id"] == 7
    assert row["game_id"] == "g"
    assert row["decision_seed"] == 11


def test_second_seat_low_safety_probability_falls_back_to_exact_baseline_action():
    observation = _t1_second_observation()
    baseline = _SemanticBaseline("max")
    log: list[dict] = []
    policy = _policy(
        baseline,
        safety_model=_SafetyProbability(0.49),
        threshold=0.5,
        log=log,
    )

    selected = policy.choose_action_observation(observation)

    assert selected is not None
    assert action_key(selected) == action_key(
        max(
            generate_turn_actions(observation.hero_board, observation.dealt_cards),
            key=lambda action: action_key(action).sort_key(),
        )
    )
    assert log[0]["override_fired"] is False
    assert log[0]["nonfire_reason"] == "below_safety_probability_threshold"
    assert log[0]["final_action_key"] == log[0]["baseline_action_key"]


def test_first_seat_is_complete_delegate_and_never_calls_models():
    class ExplodingModel:
        def predict_sample(self, _sample):
            raise AssertionError("first seat must not call candidate")

        def predict_probability(self, _features):
            raise AssertionError("first seat must not call safety")

    observation = _t1_first_observation()
    baseline = _SemanticBaseline("max")
    log: list[dict] = []
    policy = _policy(
        baseline,
        action_model=ExplodingModel(),
        safety_model=ExplodingModel(),
        log=log,
    )

    selected = policy.choose_action_observation(observation, decision_seed=19)

    assert len(baseline.observations) == 1
    expected = max(
        generate_turn_actions(observation.hero_board, observation.dealt_cards),
        key=lambda action: action_key(action).sort_key(),
    )
    assert action_key(selected) == action_key(expected)
    assert log[0]["nonfire_reason"] == "first_seat_delegated"
    assert log[0]["override_fired"] is False


def test_policy_input_and_log_have_no_hidden_truth_or_opponent_private_discards(tmp_path):
    observation = _t1_second_observation()
    baseline = _SemanticBaseline("max")
    action_model = _TieActionValueModel()
    log: list[dict] = []
    log_path = tmp_path / "decision.jsonl"
    policy = _policy(
        baseline,
        action_model=action_model,
        log=log,
        decision_log_path=log_path,
    )

    policy.choose_action_observation(observation)

    assert baseline.observations == [observation]
    sample = action_model.samples[0]
    assert set(sample["dead_cards"]) == set(observation.opponent_public_board.all_cards())
    serialized = json.dumps(log[0], sort_keys=True)
    for forbidden in (
        "opponent_private_discards",
        "true_dead_cards",
        "replay_truth",
        "world_state",
        "remaining_deck",
    ):
        assert forbidden not in serialized
    persisted = json.loads(log_path.read_text(encoding="utf-8"))
    assert persisted["policy_observation"] == observation.to_dict()


def test_dealt_permutation_preserves_semantic_candidate_and_firing_decision():
    first = _t1_second_observation(dealt=("8c", "9c", "Tc"))
    permuted = replace(first, dealt_cards=("Tc", "8c", "9c"))
    first_log: list[dict] = []
    second_log: list[dict] = []

    first_action = _policy(_SemanticBaseline("max"), log=first_log).choose_action_observation(first)
    second_action = _policy(_SemanticBaseline("max"), log=second_log).choose_action_observation(
        permuted
    )

    assert action_key(first_action) == action_key(second_action)
    assert first_log[0]["candidate_action_key"] == second_log[0]["candidate_action_key"]
    assert first_log[0]["baseline_action_key"] == second_log[0]["baseline_action_key"]
    assert first_log[0]["override_fired"] is second_log[0]["override_fired"] is True
    assert first.fingerprint() == permuted.fingerprint()


@pytest.mark.parametrize("mode", ["raise", "shape", "nan"])
def test_action_value_failure_is_fail_closed(mode):
    observation = _t1_second_observation()
    baseline = _SemanticBaseline("max")
    log: list[dict] = []
    policy = _policy(baseline, action_model=_BadActionValueModel(mode), log=log)

    selected = policy.choose_action_observation(observation)

    expected = max(
        generate_turn_actions(observation.hero_board, observation.dealt_cards),
        key=lambda action: action_key(action).sort_key(),
    )
    assert action_key(selected) == action_key(expected)
    assert log[0]["override_fired"] is False
    assert log[0]["nonfire_reason"]


@pytest.mark.parametrize(
    "safety", [_SafetyProbability(raises=True), _SafetyProbability(float("nan"))]
)
def test_safety_failure_is_fail_closed(safety):
    observation = _t1_second_observation()
    baseline = _SemanticBaseline("max")
    log: list[dict] = []
    selected = _policy(baseline, safety_model=safety, log=log).choose_action_observation(
        observation
    )

    expected = max(
        generate_turn_actions(observation.hero_board, observation.dealt_cards),
        key=lambda action: action_key(action).sort_key(),
    )
    assert action_key(selected) == action_key(expected)
    assert log[0]["override_fired"] is False
    assert log[0]["nonfire_reason"]


def test_model_path_load_failure_constructs_usable_fail_closed_policy(tmp_path):
    observation = _t1_second_observation()
    baseline = _SemanticBaseline("max")
    log: list[dict] = []
    policy = HuM4T1SelectiveOverridePolicy.from_model_paths(
        baseline,
        action_value_model_path=tmp_path / "missing.pkl",
        safety_model_path=tmp_path / "missing-safety.pkl",
        safety_probability_threshold=0.5,
        decision_log=log,
    )

    selected = policy.choose_action_observation(observation)

    assert action_key(selected) == action_key(
        max(
            generate_turn_actions(observation.hero_board, observation.dealt_cards),
            key=lambda action: action_key(action).sort_key(),
        )
    )
    assert log[0]["nonfire_reason"] == "model_load_failed"
    assert len(log[0]["model_load_failures"]) == 2


def test_safety_feature_layout_and_versioned_model_pickle_roundtrip(tmp_path):
    observation = _t1_second_observation()
    actions = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    model = _TieActionValueModel()
    from ofc_regular.hu_turn3_model import hu_policy_sample

    sample = hu_policy_sample(
        observation.hero_board,
        observation.dealt_cards,
        actions,
        opponent_board=observation.opponent_public_board,
        dead_cards=observation.legacy_dead_cards(),
        seat=observation.seat,
        to_act_order=observation.to_act_order,
    )
    values = model.predict_sample(sample)
    candidate_index = min(
        range(len(actions)), key=lambda index: action_key(actions[index]).sort_key()
    )
    baseline_index = max(
        range(len(actions)), key=lambda index: action_key(actions[index]).sort_key()
    )
    features = build_hu_m4_t1_safety_features(
        sample,
        candidate_index=candidate_index,
        baseline_index=baseline_index,
        predicted_values=values,
        seat="second",
    )
    assert features.ndim == 1
    assert np.isfinite(features).all()
    assert tuple(features[-2:]) == (0.0, 1.0)

    artifact = HuM4T1SafetyModel(
        estimator=_ConstantEstimator(0.73),
        feature_dim=features.shape[0],
        model_id="fixture",
    )
    path = tmp_path / "safety.pkl"
    artifact.save(path)
    loaded = HuM4T1SafetyModel.load(path)
    assert loaded.schema == artifact.schema
    assert loaded.model_id == "fixture"
    assert loaded.feature_dim == features.shape[0]
    assert loaded.predict_probability(features) == pytest.approx(0.73)


def test_joint_artifact_loads_as_both_action_and_safety_model(tmp_path):
    from sklearn.dummy import DummyClassifier, DummyRegressor

    from ofc_regular.hu_m4_joint_model import HuM4JointActionModel
    from ofc_regular.hu_turn3_model import HU_FEATURE_DIM

    x = np.zeros((2, HU_FEATURE_DIM), dtype=np.float32)
    policy_head = DummyClassifier(strategy="constant", constant=1).fit(
        x, np.ones(2, dtype=np.int8)
    )
    value_head = DummyRegressor(strategy="constant", constant=0.0).fit(x, [0.0, 0.0])
    delta_head = DummyRegressor(strategy="constant", constant=0.0).fit(x, [0.0, 0.0])
    uncertainty_head = DummyRegressor(strategy="constant", constant=1.0).fit(
        x, [1.0, 1.0]
    )
    safety_x = np.zeros((2, 18), dtype=np.float32)
    safety_head = DummyClassifier(strategy="constant", constant=1).fit(
        safety_x, np.ones(2, dtype=np.int8)
    )
    artifact = HuM4JointActionModel(
        policy_estimator=policy_head,
        value_estimator=value_head,
        delta_estimator=delta_head,
        uncertainty_estimator=uncertainty_head,
        safety_estimator=safety_head,
        safety_threshold=0.5,
        safety_enabled=True,
    )
    path = tmp_path / "joint.pkl"
    artifact.save(path)

    policy = HuM4T1SelectiveOverridePolicy.from_model_paths(
        _SemanticBaseline("max"),
        action_value_model_path=path,
        safety_model_path=path,
        safety_probability_threshold=0.5,
    )

    assert isinstance(policy.action_value_model, HuM4JointActionModel)
    assert isinstance(policy.safety_model, HuM4JointActionModel)
    assert policy.model_load_failures == ()

    mismatched = HuM4T1SelectiveOverridePolicy.from_model_paths(
        _SemanticBaseline("max"),
        action_value_model_path=path,
        safety_model_path=path,
        safety_probability_threshold=0.6,
    )
    assert mismatched.safety_model is None
    assert "safety_threshold_mismatch" in mismatched.model_load_failures

    second_path = tmp_path / "joint-copy.pkl"
    artifact.save(second_path)
    mixed = HuM4T1SelectiveOverridePolicy.from_model_paths(
        _SemanticBaseline("max"),
        action_value_model_path=path,
        safety_model_path=second_path,
        safety_probability_threshold=0.5,
    )
    assert mixed.safety_model is None
    assert "joint_action_safety_artifact_mismatch" in mixed.model_load_failures


def test_frozen_joint_artifact_binding_verifies_sha_manifest_and_threshold(tmp_path):
    from sklearn.dummy import DummyClassifier, DummyRegressor

    from ofc_regular.hu_m4_joint_model import HuM4JointActionModel
    from ofc_regular.hu_turn3_model import HU_FEATURE_DIM

    x = np.zeros((2, HU_FEATURE_DIM), dtype=np.float32)
    artifact = HuM4JointActionModel(
        policy_estimator=DummyClassifier(strategy="constant", constant=1).fit(
            x, np.ones(2, dtype=np.int8)
        ),
        value_estimator=DummyRegressor(strategy="constant", constant=0.0).fit(
            x, [0.0, 0.0]
        ),
        delta_estimator=DummyRegressor(strategy="constant", constant=0.0).fit(
            x, [0.0, 0.0]
        ),
        uncertainty_estimator=DummyRegressor(
            strategy="constant", constant=1.0
        ).fit(x, [1.0, 1.0]),
        safety_estimator=DummyClassifier(strategy="constant", constant=1).fit(
            np.zeros((2, 18), dtype=np.float32), np.ones(2, dtype=np.int8)
        ),
        safety_threshold=0.5,
        safety_enabled=True,
        model_id="frozen-fixture",
    )
    model_path = tmp_path / "joint-frozen.pkl"
    artifact.save(model_path)
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()

    training_manifest_path = tmp_path / "training.json"
    training_manifest_path.write_text(
        json.dumps({"schema": "fixture", "model_id": artifact.model_id}),
        encoding="utf-8",
    )
    training_sha256 = hashlib.sha256(training_manifest_path.read_bytes()).hexdigest()
    freeze_path = tmp_path / "freeze.json"
    freeze_payload = {
        "schema": "hu_m43_model_threshold_freeze_v1",
        "status": "model_and_threshold_frozen_locked_unopened",
        "model_id": artifact.model_id,
        "model_sha256": model_sha256,
        "training_manifest_sha256": training_sha256,
        "frozen_threshold": 0.5,
    }
    freeze_path.write_text(
        json.dumps(freeze_payload),
        encoding="utf-8",
    )

    policy = HuM4T1SelectiveOverridePolicy.from_model_paths(
        _SemanticBaseline("max"),
        action_value_model_path=model_path,
        safety_model_path=model_path,
        safety_probability_threshold=0.5,
        expected_joint_artifact_sha256=model_sha256,
        freeze_manifest_path=freeze_path,
        training_manifest_path=training_manifest_path,
    )
    assert policy.action_value_model is policy.safety_model
    assert policy.runtime_binding_verified is True
    assert policy.model_load_failures == ()

    wrong_sha = HuM4T1SelectiveOverridePolicy.from_model_paths(
        _SemanticBaseline("max"),
        action_value_model_path=model_path,
        safety_model_path=model_path,
        safety_probability_threshold=0.5,
        expected_joint_artifact_sha256="0" * 64,
        freeze_manifest_path=freeze_path,
        training_manifest_path=training_manifest_path,
    )
    assert wrong_sha.action_value_model is None
    assert wrong_sha.safety_model is None
    assert wrong_sha.runtime_binding_verified is False
    assert wrong_sha.model_load_failures == (
        "frozen_joint_artifact_binding_failed:ValueError",
    )

    changed_training_path = tmp_path / "changed-training.json"
    changed_training_path.write_text('{"changed":true}', encoding="utf-8")
    changed_training = HuM4T1SelectiveOverridePolicy.from_model_paths(
        _SemanticBaseline("max"),
        action_value_model_path=model_path,
        safety_model_path=model_path,
        safety_probability_threshold=0.5,
        expected_joint_artifact_sha256=model_sha256,
        freeze_manifest_path=freeze_path,
        training_manifest_path=changed_training_path,
    )
    assert changed_training.action_value_model is None
    assert changed_training.safety_model is None
    assert changed_training.runtime_binding_verified is False
    assert changed_training.model_load_failures == (
        "frozen_joint_artifact_binding_failed:ValueError",
    )

    for name, invalid_freeze in (
        ("empty", {}),
        (
            "missing-model-sha",
            {
                key: value
                for key, value in freeze_payload.items()
                if key != "model_sha256"
            },
        ),
    ):
        invalid_freeze_path = tmp_path / f"{name}-freeze.json"
        invalid_freeze_path.write_text(json.dumps(invalid_freeze), encoding="utf-8")
        rejected = HuM4T1SelectiveOverridePolicy.from_model_paths(
            _SemanticBaseline("max"),
            action_value_model_path=model_path,
            safety_model_path=model_path,
            safety_probability_threshold=0.5,
            freeze_manifest_path=invalid_freeze_path,
        )
        assert rejected.action_value_model is None
        assert rejected.safety_model is None
        assert rejected.runtime_binding_verified is False
        assert rejected.model_load_failures == (
            "frozen_joint_artifact_binding_failed:ValueError",
        )


def test_joint_artifact_calibration_no_go_is_fail_closed():
    class DisabledJointSafety(_SafetyProbability):
        safety_enabled = False

    observation = _t1_second_observation()
    baseline = _SemanticBaseline("max")
    log: list[dict] = []
    selected = _policy(
        baseline,
        safety_model=DisabledJointSafety(1.0),
        log=log,
    ).choose_action_observation(observation)

    expected = max(
        generate_turn_actions(observation.hero_board, observation.dealt_cards),
        key=lambda action: action_key(action).sort_key(),
    )
    assert action_key(selected) == action_key(expected)
    assert log[0]["override_fired"] is False
    assert log[0]["nonfire_reason"] == "artifact_safety_disabled"


def test_one_class_negative_safety_probability_is_zero_not_one():
    class NegativeOnlyEstimator:
        classes_ = np.asarray([0])

        def predict_proba(self, matrix):
            return np.ones((np.asarray(matrix).shape[0], 1), dtype=np.float64)

    features = np.zeros(3, dtype=np.float32)
    model = HuM4T1SafetyModel(
        estimator=NegativeOnlyEstimator(), feature_dim=features.size
    )
    assert model.predict_probability(features) == 0.0


def test_baseline_action_is_resolved_by_actionkey_not_positional_index():
    observation = _t1_second_observation()
    actions = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    baseline = _SemanticBaseline("max")
    log: list[dict] = []
    _policy(baseline, log=log).choose_action_observation(observation)

    # Digests and semantic keys make the index mapping replay-verifiable.  The
    # baseline's key is the maximum key even though generator order is legacy.
    assert log[0]["baseline_action_key"] == max(
        (action_key(action) for action in actions), key=lambda key: key.sort_key()
    ).to_token()
    assert log[0]["legal_action_set_digest"]
    assert log[0]["legal_action_order_digest"]


def test_policy_passes_semantically_resolved_baseline_to_opt_in_model_api():
    observation = _t1_second_observation()
    baseline = _SemanticBaseline("max")
    action_model = _BaselineAwareActionValueModel()
    log: list[dict] = []
    _policy(baseline, action_model=action_model, log=log).choose_action_observation(
        observation
    )

    assert len(action_model.calls) == 1
    sample, baseline_index = action_model.calls[0]
    assert sample["actions"][baseline_index]
    assert baseline_index == log[0]["baseline_action_index"]
    assert log[0]["baseline_score"] == 0.0
