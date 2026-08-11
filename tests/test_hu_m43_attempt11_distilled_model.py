from __future__ import annotations

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
from ofc_regular.hu_m43_attempt11_distilled_model import (
    HU_M43_ATTEMPT11_DISTILLED_FEATURE_DIM,
    Attempt11DistilledFeatures,
    Attempt11DistilledFoldPredictor,
    HuM43Attempt11DistilledModel,
)
from ofc_regular.hu_m43_attempt11_teacher import ATTEMPT11_FROZEN_MODEL_ID
from ofc_regular.hu_turn3_model import HU_FEATURE_DIM, hu_policy_sample
from ofc_regular.state import Board


class _CandidateFold:
    family = "lambda_rank"

    def __init__(self, fold_index: int) -> None:
        self.fold_index = fold_index

    def predict(self, runtime_sample, *, baseline_index):
        del baseline_index
        count = len(runtime_sample["actions"])
        return Attempt05FoldOutput(
            rank_score=np.arange(count, dtype=np.float64),
            gain_probability=np.full(count, 0.5),
            downside_p95=np.full(count, 2.0),
            downside_p99=np.full(count, 4.0),
            downside_max=np.full(count, 6.0),
        )


class _Regression:
    def __init__(self, value: float, *, column: int | None = None) -> None:
        self.value = value
        self.column = column

    def predict(self, features):
        matrix = np.asarray(features)
        if self.column is not None:
            return matrix[:, self.column].astype(np.float64)
        return np.full(matrix.shape[0], self.value, dtype=np.float64)


class _Probability:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, features):
        positive = np.full(np.asarray(features).shape[0], 0.9, dtype=np.float64)
        return np.column_stack((1.0 - positive, positive))


def _observation() -> ActorObservation:
    return ActorObservation(
        hero_board=Board.from_rows(
            top=("9h",), middle=("Th", "Jh"), bottom=("Qh", "Kh")
        ),
        opponent_public_board=Board.from_rows(
            top=("2h",),
            middle=("3h", "4h", "5h"),
            bottom=("6h", "7h", "8h"),
        ),
        dealt_cards=("Ah", "2d", "3d"),
        hero_private_discards=(),
        seat="second",
        street="T1",
        to_act_order="second",
    )


def _sample() -> dict:
    observation = _observation()
    actions = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    sample = hu_policy_sample(
        observation.hero_board,
        observation.dealt_cards,
        actions,
        opponent_board=observation.opponent_public_board,
        dead_cards=observation.legacy_dead_cards(),
        seat="second",
        to_act_order="second",
    )
    sample["policy_observation"] = observation.to_dict()
    sample["baseline_action_row_index"] = len(actions) // 2
    sample["baseline_action_key"] = action_key_from_payload(
        sample["actions"][sample["baseline_action_row_index"]]
    ).to_token()
    return sample


def _model() -> HuM43Attempt11DistilledModel:
    candidate = HuM43Attempt05Model(
        family="lambda_rank",
        fold_predictors=tuple(_CandidateFold(index) for index in range(5)),
        runtime_enabled=False,
        winner_frozen=False,
        model_id=ATTEMPT11_FROZEN_MODEL_ID,
    )
    folds = tuple(
        Attempt11DistilledFoldPredictor(
            ranker=_Regression(float(index)),
            delta_head=_Regression(1.0),
            safe_head=_Probability(),
            tail_p95_head=_Regression(2.0),
            tail_p99_head=_Regression(4.0),
            tail_max_head=_Regression(6.0),
            fold_index=index,
        )
        for index in range(5)
    )
    return HuM43Attempt11DistilledModel(
        candidate_generator=candidate,
        fold_predictors=folds,
        safety_enabled=True,
        winner_frozen=True,
    )


def _order_invariant_candidate() -> HuM43Attempt05Model:
    return HuM43Attempt05Model(
        family="lambda_rank",
        fold_predictors=tuple(
            LambdaRankFoldPredictor(
                ranker=_Regression(0.0, column=0),
                gain_head=_Probability(),
                tail_p95_head=_Regression(3.0 + index),
                tail_p99_head=_Regression(5.0 + index),
                tail_max_head=_Regression(7.0 + index),
                paired_feature_dim=4 * HU_FEATURE_DIM,
                fold_index=index,
            )
            for index in range(5)
        ),
        runtime_enabled=False,
        winner_frozen=False,
        model_id=ATTEMPT11_FROZEN_MODEL_ID,
    )


def test_attempt11_candidate_mapping_is_unique_unpadded_and_excludes_baseline() -> None:
    sample = _sample()
    built = _model().build_features(sample)
    candidate_keys = [built.action_keys[index] for index in built.candidate_indices]
    assert len(candidate_keys) == 12
    assert len(set(candidate_keys)) == len(candidate_keys)
    assert built.baseline_index not in built.candidate_indices
    assert _model().runtime_contract["candidate_count_is_variable"] is True
    assert _model().runtime_contract["candidate_padding"] is False
    assert _model().runtime_contract["candidate_duplicates"] is False


def test_attempt11_zero_candidate_state_falls_back_to_explicit_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = _sample()
    baseline = sample["baseline_action_row_index"]
    keys = tuple(action_key_from_payload(item) for item in sample["actions"][:2])
    built = Attempt11DistilledFeatures(
        runtime_sample=sample,
        features=np.zeros(
            (2, HU_M43_ATTEMPT11_DISTILLED_FEATURE_DIM), dtype=np.float32
        ),
        action_keys=keys,
        candidate_indices=(),
        baseline_index=0,
    )
    monkeypatch.setattr(
        HuM43Attempt11DistilledModel,
        "build_features",
        lambda self, _sample, baseline_index=None: built,
    )
    heads = _model().predict_heads_sample(sample, baseline_index=baseline)
    assert heads.selected_index == 0
    assert heads.baseline_index == 0
    assert not np.any(heads.candidate_mask)
    assert not np.any(heads.gate_eligible_mask)
    np.testing.assert_array_equal(heads.action_score, np.asarray([0.0, -1.0]))


def test_attempt11_action_key_candidates_ignore_input_enumeration_order() -> None:
    sample = _sample()
    candidate = _order_invariant_candidate()
    model = HuM43Attempt11DistilledModel(
        candidate_generator=candidate,
        fold_predictors=_model().fold_predictors,
    )
    original = model.build_features(sample)
    original_keys = tuple(
        original.action_keys[index] for index in original.candidate_indices
    )

    permuted = dict(sample)
    permuted["actions"] = list(reversed(sample["actions"]))
    permuted["baseline_action_row_index"] = next(
        index
        for index, payload in enumerate(permuted["actions"])
        if action_key_from_payload(payload).to_token() == sample["baseline_action_key"]
    )
    rebuilt = model.build_features(permuted)
    rebuilt_keys = tuple(
        rebuilt.action_keys[index] for index in rebuilt.candidate_indices
    )
    assert rebuilt_keys == original_keys
