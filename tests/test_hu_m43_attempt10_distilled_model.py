from __future__ import annotations

import dataclasses
from copy import deepcopy

import numpy as np
import pytest

from ofc_regular.action_key import action_key_from_payload
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m43_attempt05_model import (
    Attempt05FoldOutput,
    HuM43Attempt05Model,
    LambdaRankFoldPredictor,
)
from ofc_regular.hu_m43_attempt10_distilled_model import (
    BoundHuM43Attempt10DistilledModel,
    HU_M43_ATTEMPT10_DISTILLED_FEATURE_DIM,
    Attempt10DistilledFoldPredictor,
    HuM43Attempt10DistilledModel,
    is_bound_attempt10_distilled_model,
)
from ofc_regular.hu_m43_attempt10_distilled_runtime import (
    ATTEMPT10_BOUND_EXECUTION_MODULES,
    FrozenExecutionModulesAttestation,
)
from ofc_regular.hu_m43_attempt10_teacher import (
    ATTEMPT10_FROZEN_MODEL_ID,
    ATTEMPT10_FROZEN_MODEL_SHA256,
    FrozenAttempt10LambdaRanker,
)
from ofc_regular.hu_turn3_model import HU_FEATURE_DIM, hu_policy_sample
from ofc_regular.state import Board


class _CandidateFold:
    family = "lambda_rank"

    def __init__(self, fold_index: int) -> None:
        self.fold_index = fold_index

    def predict(self, runtime_sample, *, baseline_index):
        count = len(runtime_sample["actions"])
        rank = np.arange(count, dtype=np.float64) + self.fold_index * 0.01
        return Attempt05FoldOutput(
            rank_score=rank,
            gain_probability=np.full(count, 0.5),
            downside_p95=np.full(count, 3.0),
            downside_p99=np.full(count, 5.0),
            downside_max=np.full(count, 7.0),
        )


class _FeatureRegression:
    def __init__(self, column: int | None = None, value: float = 0.0) -> None:
        self.column = column
        self.value = value

    def predict(self, features):
        matrix = np.asarray(features)
        if self.column is None:
            return np.full(matrix.shape[0], self.value, dtype=np.float64)
        return matrix[:, self.column].astype(np.float64)


class _Probability:
    classes_ = np.asarray([0, 1])

    def __init__(self, value: float) -> None:
        self.value = value

    def predict_proba(self, features):
        positive = np.full(np.asarray(features).shape[0], self.value)
        return np.column_stack((1.0 - positive, positive))


def _observation(*, seat: str = "second") -> ActorObservation:
    opponent = (
        Board.from_rows(
            top=("2h",),
            middle=("3h", "4h", "5h"),
            bottom=("6h", "7h", "8h"),
        )
        if seat == "second"
        else Board.from_rows(
            top=("2h",), middle=("3h", "4h"), bottom=("6h", "7h")
        )
    )
    return ActorObservation(
        hero_board=Board.from_rows(
            top=("9h",), middle=("Th", "Jh"), bottom=("Qh", "Kh")
        ),
        opponent_public_board=opponent,
        dealt_cards=("Ah", "2d", "3d"),
        hero_private_discards=(),
        seat=seat,
        street="T1",
        to_act_order=seat,
    )


def _sample(*, seat: str = "second") -> dict:
    observation = _observation(seat=seat)
    actions = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    sample = hu_policy_sample(
        observation.hero_board,
        observation.dealt_cards,
        actions,
        opponent_board=observation.opponent_public_board,
        dead_cards=observation.legacy_dead_cards(),
        seat=seat,
        to_act_order=seat,
    )
    sample["policy_observation"] = observation.to_dict()
    baseline = len(actions) // 2
    sample["baseline_action_row_index"] = baseline
    sample["baseline_action_key"] = action_key_from_payload(
        sample["actions"][baseline]
    ).to_token()
    return sample


def _candidate() -> HuM43Attempt05Model:
    return HuM43Attempt05Model(
        family="lambda_rank",
        fold_predictors=tuple(_CandidateFold(index) for index in range(5)),
        runtime_enabled=False,
        winner_frozen=False,
        model_id=ATTEMPT10_FROZEN_MODEL_ID,
    )


def _fast_candidate() -> HuM43Attempt05Model:
    return HuM43Attempt05Model(
        family="lambda_rank",
        fold_predictors=tuple(
            LambdaRankFoldPredictor(
                ranker=_FeatureRegression(column=0),
                gain_head=_Probability(0.5),
                tail_p95_head=_FeatureRegression(value=3.0 + index),
                tail_p99_head=_FeatureRegression(value=5.0 + index),
                tail_max_head=_FeatureRegression(value=7.0 + index),
                paired_feature_dim=4 * HU_FEATURE_DIM,
                fold_index=index,
            )
            for index in range(5)
        ),
        runtime_enabled=False,
        winner_frozen=False,
        model_id=ATTEMPT10_FROZEN_MODEL_ID,
    )


def _distilled_fold(index: int, *, safe: float = 0.9):
    return Attempt10DistilledFoldPredictor(
        ranker=_FeatureRegression(column=-10),
        delta_head=_FeatureRegression(value=1.0),
        safe_head=_Probability(safe),
        tail_p95_head=_FeatureRegression(value=2.0),
        tail_p99_head=_FeatureRegression(value=4.0),
        tail_max_head=_FeatureRegression(value=6.0),
        fold_index=index,
    )


def _test_attestation() -> FrozenExecutionModulesAttestation:
    """Construct an isolated unit-test receipt without exposing a runtime issuer."""

    attestation = object.__new__(FrozenExecutionModulesAttestation)
    object.__setattr__(attestation, "extracted_root", "unit-test-only")
    object.__setattr__(
        attestation,
        "module_paths",
        tuple((name, f"unit-test/{name}.py") for name in ATTEMPT10_BOUND_EXECUTION_MODULES),
    )
    return attestation


def _test_bound(
    model: HuM43Attempt10DistilledModel,
) -> BoundHuM43Attempt10DistilledModel:
    """Initialize wrapper slots directly for isolated model-head unit tests."""

    bound = object.__new__(BoundHuM43Attempt10DistilledModel)
    object.__setattr__(
        bound, "_BoundHuM43Attempt10DistilledModel__model", model
    )
    object.__setattr__(
        bound,
        "_BoundHuM43Attempt10DistilledModel__attestation",
        _test_attestation(),
    )
    return bound


def _model(*, unsafe_folds: int = 0):
    raw = HuM43Attempt10DistilledModel(
        candidate_generator=_candidate(),
        fold_predictors=tuple(
            _distilled_fold(index, safe=0.1 if index < unsafe_folds else 0.9)
            for index in range(5)
        ),
        safety_enabled=True,
        winner_frozen=True,
        model_id="attempt10-distilled-test",
    )
    return _test_bound(raw)


def test_distilled_model_rebuilds_full_legal_set_and_selects_only_top12() -> None:
    sample = _sample()
    model = _model()
    heads = model.predict_heads_sample(sample)
    assert heads.action_score.shape == (len(sample["actions"]),)
    assert model.build_features(sample).features.shape[1] == (
        HU_M43_ATTEMPT10_DISTILLED_FEATURE_DIM
    )
    assert heads.selected_index != heads.baseline_index
    assert heads.candidate_mask[heads.selected_index]
    assert heads.fold_vote_count[heads.selected_index] == 5
    assert heads.action_score[heads.baseline_index] == 0.0
    assert heads.action_score[heads.selected_index] == 1.0
    assert np.sum(heads.action_score == 1.0) == 1
    assert model.runtime_contract["runtime_teacher_ev"] is False
    assert model.runtime_contract["opponent_private_discard_input"] is False


def test_distilled_model_requires_four_identical_fold_votes_and_falls_back() -> None:
    sample = _sample()
    heads = _model(unsafe_folds=2).predict_heads_sample(sample)
    assert heads.selected_index == heads.baseline_index
    assert not np.any(heads.gate_eligible_mask)
    assert np.max(heads.fold_vote_count) == 3
    np.testing.assert_array_equal(
        _model(unsafe_folds=2).predict_sample(sample), heads.action_score
    )


def test_distilled_model_rejects_hidden_or_incomplete_runtime_input() -> None:
    sample = _sample()
    polluted = deepcopy(sample)
    polluted["policy_observation"]["opponent_private_discards"] = ["As"]
    with pytest.raises((TypeError, ValueError)):
        _model().predict_heads_sample(polluted)
    incomplete = deepcopy(sample)
    incomplete["actions"].pop()
    with pytest.raises(ValueError, match="complete legal action set"):
        _model().predict_heads_sample(incomplete)
    with pytest.raises(ValueError, match="T1-second"):
        _model().predict_heads_sample(_sample(seat="first"))


def test_distilled_model_artifact_round_trip_and_sha_binding(tmp_path) -> None:
    path = tmp_path / "model.pkl"
    model = _model()
    sample = _sample()
    heads = model.predict_heads_sample(sample)
    model.predict_sample_with_baseline(
        sample, baseline_index=sample["baseline_action_row_index"]
    )
    assert model._last_prediction_cache is not None
    assert model.predict_safety_probability(
        sample,
        candidate_index=heads.selected_index,
        baseline_index=heads.baseline_index,
    ) == pytest.approx(heads.safe_probability[heads.selected_index])
    digest = model.save(path)
    loaded = HuM43Attempt10DistilledModel.load(path, expected_sha256=digest)
    assert loaded._last_prediction_cache is None
    assert loaded.runtime_binding_verified is False
    raw_heads = loaded.predict_heads_sample(_sample())
    assert raw_heads.selected_index == raw_heads.baseline_index
    assert not np.any(raw_heads.action_score == 1.0)
    with pytest.raises(TypeError, match="unexpected keyword"):
        dataclasses.replace(loaded, runtime_binding_verified=True)
    with pytest.raises(TypeError, match="loader-issued"):
        BoundHuM43Attempt10DistilledModel(loaded, object())
    with pytest.raises(TypeError, match="loader-issued"):
        BoundHuM43Attempt10DistilledModel(loaded, _test_attestation())
    forged = object.__new__(BoundHuM43Attempt10DistilledModel)
    assert is_bound_attempt10_distilled_model(forged) is False
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        HuM43Attempt10DistilledModel.load(path, expected_sha256="0" * 64)


def test_shared_matrix_fast_candidate_path_matches_legacy_fold_contract() -> None:
    sample = _sample()
    baseline = sample["baseline_action_row_index"]
    candidate = _fast_candidate()
    built = HuM43Attempt10DistilledModel(
        candidate_generator=candidate,
        fold_predictors=tuple(_distilled_fold(index) for index in range(5)),
    ).build_features(sample)
    observation = ActorObservation.from_dict(sample["policy_observation"])
    actions = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    legacy = FrozenAttempt10LambdaRanker(
        model=candidate, artifact_sha256=ATTEMPT10_FROZEN_MODEL_SHA256
    ).score_actions(observation, actions, baseline_index=baseline)
    np.testing.assert_allclose(built.features[:, -10], legacy.rank_mean)
    np.testing.assert_allclose(built.features[:, -9], legacy.rank_disagreement)
    np.testing.assert_allclose(
        built.features[:, -6], np.asarray(legacy.raw_downside_p95) / 22.0
    )
    np.testing.assert_allclose(
        built.features[:, -5], np.asarray(legacy.raw_downside_p99) / 36.0
    )
    np.testing.assert_allclose(
        built.features[:, -4], np.asarray(legacy.raw_downside_max) / 45.0
    )


def test_distilled_action_mapping_is_invariant_to_legal_enumeration_order() -> None:
    sample = _sample()
    model = _test_bound(
        HuM43Attempt10DistilledModel(
            candidate_generator=_fast_candidate(),
            fold_predictors=tuple(_distilled_fold(index) for index in range(5)),
            safety_enabled=True,
            winner_frozen=True,
        )
    )
    original_heads = model.predict_heads_sample(sample)
    original_key = action_key_from_payload(
        sample["actions"][original_heads.selected_index]
    )
    permuted = deepcopy(sample)
    permuted["actions"] = list(reversed(permuted["actions"]))
    permuted["baseline_action_row_index"] = next(
        index
        for index, action in enumerate(permuted["actions"])
        if action_key_from_payload(action).to_token() == sample["baseline_action_key"]
    )
    permuted_heads = model.predict_heads_sample(permuted)
    assert action_key_from_payload(
        permuted["actions"][permuted_heads.selected_index]
    ) == original_key


