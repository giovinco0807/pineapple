from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import pickle
import sys

import numpy as np
import pytest

from ofc_regular.action_key import action_key_from_payload
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.cards import ALL_CARDS
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m43_joint_model_v5 import (
    HU_M43_V5_ACTION_SCORE_MODE,
    HU_M43_V5_FIT_ROW_SCOPE,
    HU_M43_V5_FEATURE_NAMES,
    HU_M43_V5_META_FEATURE_DIM,
    HU_M43_V5_MIN_POSITIVE_VOTES,
    HuM43JointModelV5,
    V5MetaRankerConfig,
    build_v5_stacked_features,
    build_v5_state_balanced_weights,
    fit_v5_meta_ranker,
)
from ofc_regular.hu_m4_joint_model import (
    ConstantProbabilityEstimator,
    PairedDeltaRiskFoldEstimator,
)
from ofc_regular.hu_m4_t1_policy import HuM4T1SelectiveOverridePolicy
from ofc_regular.hu_turn3_model import (
    HU_FEATURE_DIM,
    hu_policy_sample,
    load_hu_action_value_model,
)
from ofc_regular.state import Board


class _ConstantRegression:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, matrix):
        return np.full(np.asarray(matrix).shape[0], self.value, dtype=np.float64)


class _WeightedDifferenceRegression:
    def __init__(self, scale: float) -> None:
        self.scale = float(scale)

    def predict(self, matrix):
        values = np.asarray(matrix, dtype=np.float64)
        difference = values[:, 2 * HU_FEATURE_DIM : 3 * HU_FEATURE_DIM]
        weights = np.linspace(0.25, 1.25, HU_FEATURE_DIM, dtype=np.float64)
        return self.scale * (difference @ weights)


class _FeatureColumnRegression:
    def __init__(self, column: int, scale: float = 1.0) -> None:
        self.column = int(column)
        self.scale = float(scale)

    def predict(self, matrix):
        return self.scale * np.asarray(matrix, dtype=np.float64)[:, self.column]


class _SemanticStage18Scorer:
    def predict_sample(self, sample):
        keys = [action_key_from_payload(action) for action in sample["actions"]]
        ordered = {
            key: rank
            for rank, key in enumerate(sorted(keys, key=lambda key: key.sort_key()))
        }
        return np.asarray([ordered[key] / max(len(keys) - 1, 1) for key in keys])


class _RecordingStage18Scorer(_SemanticStage18Scorer):
    def __init__(self) -> None:
        self.last_sample = None

    def predict_sample(self, sample):
        self.last_sample = sample
        return super().predict_sample(sample)


class _ExplodingSafetyEstimator:
    classes_ = (0, 1)

    def predict_proba(self, _matrix):
        raise AssertionError("safety must not run without an eligible proposal")


class _BaselinePolicy:
    def choose_action_observation(self, observation, **_kwargs):
        return generate_turn_actions(
            observation.hero_board, observation.dealt_cards
        )[0]


def _fold(index: int, *, scale: float) -> PairedDeltaRiskFoldEstimator:
    return PairedDeltaRiskFoldEstimator(
        delta_estimator=_WeightedDifferenceRegression(scale),
        positive_gain_estimator=_ConstantRegression(0.55 + 0.01 * index),
        downside_p95_estimator=_ConstantRegression(4.0 + index),
        downside_p99_estimator=_ConstantRegression(8.0 + index),
        downside_max_estimator=_ConstantRegression(16.0 + index),
        paired_feature_dim=4 * HU_FEATURE_DIM,
        fold_index=index,
    )


def _constant_fold(index: int) -> PairedDeltaRiskFoldEstimator:
    return PairedDeltaRiskFoldEstimator(
        delta_estimator=_ConstantRegression(float(index)),
        positive_gain_estimator=_ConstantRegression(0.5),
        downside_p95_estimator=_ConstantRegression(5.0),
        downside_p99_estimator=_ConstantRegression(10.0),
        downside_max_estimator=_ConstantRegression(20.0),
        paired_feature_dim=4 * HU_FEATURE_DIM,
        fold_index=index,
    )


def _folds() -> tuple[PairedDeltaRiskFoldEstimator, ...]:
    # A positive semantic delta gets exactly three positive votes; a negative
    # delta gets two.  This makes the hard 3/5 boundary observable in tests.
    scales = (1.0, 1.0, 1.0, -1.0, -1.0)
    return tuple(_fold(index, scale=scale) for index, scale in enumerate(scales))


def _sample(offset: int = 0) -> dict:
    hero = Board.from_rows(
        top=["Ah"], middle=["Kd"], bottom=["2s", "3s", "4s"]
    )
    opponent = Board.from_rows(
        top=["Qh"], middle=["Jd", "Td"], bottom=["5s", "6s", "7s", "8s"]
    )
    used = {*hero.all_cards(), *opponent.all_cards()}
    remaining = [card for card in ALL_CARDS if card not in used]
    dealt = tuple(remaining[offset : offset + 3])
    observation = ActorObservation(
        hero_board=hero,
        opponent_public_board=opponent,
        dealt_cards=dealt,
        hero_private_discards=(),
        seat="second",
        street="T1",
        to_act_order="second",
    )
    legal = generate_turn_actions(hero, dealt)
    sample = hu_policy_sample(
        hero,
        dealt,
        legal,
        opponent_board=opponent,
        dead_cards=observation.legacy_dead_cards(),
        seat="second",
        to_act_order="second",
    )
    sample["baseline_action_row_index"] = len(legal) // 2
    return sample


def _model(
    *,
    safety_probability: float | None = None,
    safety_enabled: bool | None = None,
) -> HuM43JointModelV5:
    safety = (
        ConstantProbabilityEstimator(safety_probability)
        if safety_probability is not None
        else None
    )
    return HuM43JointModelV5(
        paired_fold_estimators=_folds(),
        stage18_scorer=_SemanticStage18Scorer(),
        # Positive base delta is feature column 0 and passes the 3/5 vote in
        # this deterministic fixture.
        meta_ranker=_FeatureColumnRegression(0),
        safety_estimator=safety,
        safety_threshold=0.7,
        safety_enabled=(safety is not None if safety_enabled is None else safety_enabled),
        model_id="v5-test",
        manifest={"current_profile_mutated": False, "runtime_enabled": False},
    )


def test_v5_exact_22_feature_schema_and_hard_three_of_five_eligibility():
    sample = _sample()
    baseline = sample["baseline_action_row_index"]
    stacked = build_v5_stacked_features(
        sample,
        paired_fold_estimators=_folds(),
        stage18_scorer=_SemanticStage18Scorer(),
        baseline_index=baseline,
    )

    assert len(HU_M43_V5_FEATURE_NAMES) == HU_M43_V5_META_FEATURE_DIM == 22
    assert HU_M43_V5_FEATURE_NAMES == (
        "base_delta",
        "base_positive",
        "base_downside_p95",
        "base_downside_p99",
        "base_downside_max",
        "base_delta_disagreement",
        "stage18_action_score",
        "stage18_delta_vs_baseline",
        "stage18_nonbaseline_rank_fraction",
        "stage18_next_margin",
        "stage18_nonbaseline_mean",
        "stage18_nonbaseline_std",
        "stage18_nonbaseline_max",
        "stage18_nonbaseline_min",
        "base_delta_nonbaseline_rank_fraction",
        "base_delta_next_margin",
        "base_delta_nonbaseline_mean",
        "base_delta_nonbaseline_std",
        "base_delta_nonbaseline_max",
        "base_delta_nonbaseline_min",
        "nonbaseline_action_count",
        "stage18_baseline_score",
    )
    assert stacked.features.shape == (len(sample["actions"]), 22)
    expected_votes = np.sum(stacked.base.fold_centered_delta > 0.0, axis=0)
    np.testing.assert_array_equal(stacked.base.delta_positive_votes, expected_votes)
    np.testing.assert_array_equal(
        stacked.base.eligible_mask,
        expected_votes >= HU_M43_V5_MIN_POSITIVE_VOTES,
    )
    assert not stacked.base.eligible_mask[baseline]
    assert stacked.base.delta_positive_votes[baseline] == 0
    assert np.any(stacked.base.eligible_mask)
    assert np.any(
        (~stacked.base.eligible_mask)
        & (np.arange(len(sample["actions"])) != baseline)
    )

    row = next(index for index in range(len(sample["actions"])) if index != baseline)
    nonbaseline = np.delete(stacked.stage18_score, baseline)
    assert stacked.features[row, 6] == pytest.approx(stacked.stage18_score[row])
    assert stacked.features[row, 7] == pytest.approx(
        stacked.stage18_score[row] - stacked.stage18_score[baseline]
    )
    assert stacked.features[row, 10] == pytest.approx(np.mean(nonbaseline))
    assert stacked.features[row, 11] == pytest.approx(np.std(nonbaseline))
    base_nonbaseline = np.delete(stacked.base.delta, baseline)
    assert stacked.features[row, 16] == pytest.approx(np.mean(base_nonbaseline))
    assert stacked.features[row, 17] == pytest.approx(np.std(base_nonbaseline))
    assert stacked.features[row, 18] == pytest.approx(np.max(base_nonbaseline))
    assert stacked.features[row, 19] == pytest.approx(np.min(base_nonbaseline))
    assert stacked.features[row, 20] == len(sample["actions"]) - 1
    assert stacked.features[row, 21] == pytest.approx(
        stacked.stage18_score[baseline]
    )


def test_v5_runtime_projects_out_teacher_values_and_lcb_before_stage18():
    sample = _sample()
    baseline = sample["baseline_action_row_index"]
    scorer = _RecordingStage18Scorer()
    model = HuM43JointModelV5(
        paired_fold_estimators=_folds(),
        stage18_scorer=scorer,
        meta_ranker=_FeatureColumnRegression(7),
    )
    original = model.predict_heads_sample(sample, baseline_index=baseline)

    changed = deepcopy(sample)
    changed["best_action"] = 123456
    changed["score_gap"] = 1.0e9
    changed["teacher_ev"] = 1.0e9
    changed["teacher_lcb"] = -1.0e9
    for index, action in enumerate(changed["actions"]):
        action["score"] = float(index * 100_000)
        action["score_se"] = 0.001
        action["teacher_lcb"] = float(index)
        action["paired_delta_vs_baseline"] = {"mean": float(-index)}
        action["delta_se_vs_baseline"] = 0.01
    altered = model.predict_heads_sample(changed, baseline_index=baseline)

    np.testing.assert_array_equal(original.meta_features, altered.meta_features)
    np.testing.assert_array_equal(original.meta_score, altered.meta_score)
    np.testing.assert_array_equal(original.action_score, altered.action_score)
    assert scorer.last_sample is not None
    assert not {"best_action", "score_gap", "teacher_ev", "teacher_lcb"} & set(
        scorer.last_sample
    )
    assert all(
        set(action) <= {"placements", "discards", "next_board"}
        for action in scorer.last_sample["actions"]
    )


def test_v5_runtime_projection_smoke_with_local_fixed_stage18_model():
    model_path = (
        Path(__file__).resolve().parents[1]
        / "models"
        / "hu_turn1_stage18_first1801_mc32_hgb_abs_aug3.pkl"
    )
    if not model_path.is_file():
        pytest.skip("local fixed Stage18 artifact is not present")
    stage18 = load_hu_action_value_model(model_path)
    sample = _sample()
    stacked = build_v5_stacked_features(
        sample,
        paired_fold_estimators=_folds(),
        stage18_scorer=stage18,
        baseline_index=sample["baseline_action_row_index"],
    )
    assert stacked.stage18_score.shape == (len(sample["actions"]),)
    assert np.isfinite(stacked.stage18_score).all()
    assert np.ptp(stacked.stage18_score) > 0.0


def test_v5_canonical_actionkey_tie_break_and_permutation_invariance():
    sample = _sample()
    baseline = sample["baseline_action_row_index"]
    model = HuM43JointModelV5(
        paired_fold_estimators=_folds(),
        stage18_scorer=_SemanticStage18Scorer(),
        meta_ranker=_ConstantRegression(0.0),
        safety_estimator=ConstantProbabilityEstimator(0.9),
        safety_threshold=0.7,
        safety_enabled=True,
    )
    original = model.select_action_index(sample, baseline_index=baseline)
    original_key = action_key_from_payload(sample["actions"][original.proposal_index])
    expected_key = min(
        (
            action_key_from_payload(sample["actions"][index])
            for index in range(len(sample["actions"]))
            if index != baseline
        ),
        key=lambda key: key.sort_key(),
    )
    assert original_key == expected_key

    reversed_sample = deepcopy(sample)
    reversed_sample["actions"] = list(reversed(reversed_sample["actions"]))
    reversed_sample["baseline_action_row_index"] = (
        len(reversed_sample["actions"]) - 1 - baseline
    )
    reversed_decision = model.select_action_index(reversed_sample)
    reversed_key = action_key_from_payload(
        reversed_sample["actions"][reversed_decision.proposal_index]
    )
    assert reversed_key == original_key
    assert reversed_decision.eligibility_passed == original.eligibility_passed
    assert reversed_decision.override_fired == original.override_fired


def test_v5_safety_runs_only_after_eligibility_and_disabled_is_baseline_fallback():
    sample = _sample()
    baseline = sample["baseline_action_row_index"]

    enabled = _model(safety_probability=0.8)
    decision = enabled.select_action_index(sample, baseline_index=baseline)
    assert decision.eligibility_passed
    assert decision.positive_vote_count >= 3
    assert decision.selected_index == decision.proposal_index
    assert decision.override_fired

    disabled = _model(safety_probability=0.8, safety_enabled=False)
    disabled_decision = disabled.select_action_index(sample, baseline_index=baseline)
    assert disabled_decision.eligibility_passed
    assert disabled_decision.selected_index == baseline
    assert not disabled_decision.override_fired

    # The all-nonbaseline meta argmax deliberately chooses an ineligible
    # negative-base-delta action even though eligible alternatives exist.  v5
    # must reject that proposal, not rerank into the eligible subset.
    no_rerank = HuM43JointModelV5(
        paired_fold_estimators=_folds(),
        stage18_scorer=_SemanticStage18Scorer(),
        meta_ranker=_FeatureColumnRegression(0, scale=-1.0),
        safety_estimator=_ExplodingSafetyEstimator(),
        safety_threshold=0.7,
        safety_enabled=True,
    )
    heads = no_rerank.predict_heads_sample(sample, baseline_index=baseline)
    assert np.any(heads.eligible_mask)
    assert not heads.proposal_eligible
    closed = no_rerank.select_action_index(sample, baseline_index=baseline)
    assert not closed.eligibility_passed
    assert closed.proposal_index != baseline
    assert closed.selected_index == baseline
    assert not closed.override_fired
    scores = no_rerank.predict_sample_with_baseline(sample, baseline_index=baseline)
    assert int(np.argmax(scores)) == baseline
    assert np.all(np.delete(scores, baseline) < 0.0)

    no_eligible = HuM43JointModelV5(
        paired_fold_estimators=tuple(_constant_fold(index) for index in range(5)),
        stage18_scorer=_SemanticStage18Scorer(),
        meta_ranker=_FeatureColumnRegression(7),
        safety_estimator=_ExplodingSafetyEstimator(),
        safety_threshold=0.7,
        safety_enabled=True,
    )
    completely_closed = no_eligible.select_action_index(
        sample, baseline_index=baseline
    )
    assert not completely_closed.eligibility_passed
    assert completely_closed.selected_index == baseline


def test_v5_lightgbm_huber_config_state_balancing_and_determinism():
    rng = np.random.default_rng(20260714)
    features = rng.normal(size=(180, HU_M43_V5_META_FEATURE_DIM)).astype(np.float32)
    targets = features[:, 0] - 0.4 * features[:, 1] + 0.1 * features[:, 7]
    states = np.asarray([0] * 20 + [1] * 60 + [2] * 100, dtype=np.int32)
    weights = build_v5_state_balanced_weights(states)
    assert np.sum(weights[states == 0]) == pytest.approx(1.0)
    assert np.sum(weights[states == 1]) == pytest.approx(1.0)
    assert np.sum(weights[states == 2]) == pytest.approx(1.0)

    config = V5MetaRankerConfig()
    first = fit_v5_meta_ranker(features, targets, states, config=config)
    second = fit_v5_meta_ranker(features, targets, states, config=config)
    params = first.estimator.get_params()
    assert params["objective"] == "huber"
    assert params["n_estimators"] == 180
    assert params["learning_rate"] == 0.035
    assert params["num_leaves"] == 7
    assert params["min_child_samples"] == 50
    assert params["reg_lambda"] == 8.0
    assert params["reg_alpha"] == 0.5
    assert params["n_jobs"] == 1
    np.testing.assert_array_equal(first.sample_weights, weights)
    np.testing.assert_array_equal(
        first.estimator.booster_.predict(features),
        second.estimator.booster_.predict(features),
    )
    assert first.manifest["runtime_teacher_inputs"] is False
    assert first.manifest["current_profile_mutated"] is False
    assert first.manifest["runtime_policy_activated"] is False
    assert first.manifest["fit_row_scope"] == HU_M43_V5_FIT_ROW_SCOPE
    assert first.manifest["eligibility_after_canonical_argmax"] is True
    assert first.manifest["eligible_rerank_allowed"] is False
    assert first.manifest["eligibility_application"] == (
        "post_all_nonbaseline_argmax_no_eligible_rerank"
    )
    assert first.manifest["state_weight_sum_min"] == pytest.approx(1.0)
    assert first.manifest["state_weight_sum_max"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="every nonbaseline action"):
        fit_v5_meta_ranker(
            features,
            targets,
            states,
            row_scope="eligibility_passing_actions_only",
        )


def test_v5_artifact_roundtrip_and_exact_fold_contract(tmp_path):
    sample = _sample()
    baseline = sample["baseline_action_row_index"]
    model = _model(safety_probability=0.8)
    assert model.action_score_mode == HU_M43_V5_ACTION_SCORE_MODE
    artifact = tmp_path / "v5.pkl"
    digest = model.save(artifact)
    assert digest == hashlib.sha256(artifact.read_bytes()).hexdigest()
    payload = pickle.loads(artifact.read_bytes())
    assert payload["fit_row_scope"] == HU_M43_V5_FIT_ROW_SCOPE
    assert payload["eligibility_after_canonical_argmax"] is True
    assert payload["eligible_rerank_allowed"] is False
    assert model.runtime_contract["eligibility_after_canonical_argmax"] is True
    assert model.runtime_contract["eligible_rerank_allowed"] is False
    loaded = HuM43JointModelV5.load(artifact, expected_sha256=digest)
    np.testing.assert_array_equal(
        model.predict_sample_with_baseline(sample, baseline_index=baseline),
        loaded.predict_sample_with_baseline(sample, baseline_index=baseline),
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        HuM43JointModelV5.load(artifact, expected_sha256="f" * 64)
    with pytest.raises(ValueError, match="exactly five"):
        HuM43JointModelV5(
            paired_fold_estimators=_folds()[:4],
            stage18_scorer=_SemanticStage18Scorer(),
            meta_ranker=_ConstantRegression(0.0),
        )


def test_v5_duck_type_works_with_existing_opt_in_policy():
    sample = _sample()
    hero = Board.from_rows(
        top=["Ah"], middle=["Kd"], bottom=["2s", "3s", "4s"]
    )
    opponent = Board.from_rows(
        top=["Qh"], middle=["Jd", "Td"], bottom=["5s", "6s", "7s", "8s"]
    )
    observation = ActorObservation(
        hero_board=hero,
        opponent_public_board=opponent,
        dealt_cards=tuple(sample["dealt"]),
        hero_private_discards=(),
        seat="second",
        street="T1",
        to_act_order="second",
    )
    model = _model(safety_probability=0.9)
    decisions = []
    policy = HuM4T1SelectiveOverridePolicy(
        _BaselinePolicy(),
        action_value_model=model,
        safety_model=model,
        safety_probability_threshold=0.7,
        decision_log=decisions,
        runtime_binding_verified=True,
    )
    baseline = _BaselinePolicy().choose_action_observation(observation)
    chosen = policy.choose_action_observation(observation)
    assert decisions[-1]["override_fired"] is True
    assert decisions[-1]["candidate_action_index"] != decisions[-1]["baseline_action_index"]
    assert chosen != baseline


def test_v5_frozen_hyperparameters_reject_silent_algorithm_drift():
    with pytest.raises(ValueError, match="freezes n_estimators"):
        V5MetaRankerConfig(n_estimators=179)
    with pytest.raises(ValueError, match="freezes objective"):
        V5MetaRankerConfig(objective="regression")


def test_v5_fit_fails_closed_when_lightgbm_is_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "lightgbm", None)
    features = np.zeros((1, HU_M43_V5_META_FEATURE_DIM), dtype=np.float32)
    with pytest.raises(RuntimeError, match="requires lightgbm"):
        fit_v5_meta_ranker(features, [0.0], [0])
