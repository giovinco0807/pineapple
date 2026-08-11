from __future__ import annotations

from copy import deepcopy
import pickle
import sys
import types

import numpy as np
import pytest

from ofc_regular.action_key import action_key_from_payload
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.cards import ALL_CARDS
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m43_joint_model_loader import load_hu_m43_joint_action_model
from ofc_regular.hu_m43_joint_model_v6 import HuM43JointModelV6
from ofc_regular.hu_m4_joint_model import (
    ConstantProbabilityEstimator,
    PairedDeltaRiskFoldEstimator,
)
from ofc_regular.hu_m4_t1_policy import HuM4T1SelectiveOverridePolicy
from ofc_regular.hu_turn3_model import HU_FEATURE_DIM, hu_policy_sample
from ofc_regular.state import Board


class _ConstantRegression:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, matrix):
        return np.full(np.asarray(matrix).shape[0], self.value, dtype=np.float64)


class _ExplodingModel:
    def predict_sample(self, _sample):
        raise AssertionError("first-seat fallback must not invoke the v6 model")


class _BaselinePolicy:
    def choose_action_observation(self, observation, **_kwargs):
        return generate_turn_actions(
            observation.hero_board, observation.dealt_cards
        )[0]


def _fold(index: int, *, tail: float = 5.0) -> PairedDeltaRiskFoldEstimator:
    return PairedDeltaRiskFoldEstimator(
        delta_estimator=_ConstantRegression(1.0 + index),
        positive_gain_estimator=_ConstantRegression(0.7),
        downside_p95_estimator=_ConstantRegression(tail),
        downside_p99_estimator=_ConstantRegression(2.0 * tail),
        downside_max_estimator=_ConstantRegression(3.0 * tail),
        paired_feature_dim=4 * HU_FEATURE_DIM,
        fold_index=index,
    )


def _model(
    *, tail: float = 5.0, probability: float | None = 0.8
) -> HuM43JointModelV6:
    return HuM43JointModelV6(
        paired_fold_estimators=tuple(_fold(index, tail=tail) for index in range(5)),
        safety_estimator=(
            None
            if probability is None
            else ConstantProbabilityEstimator(probability)
        ),
        safety_threshold=0.7,
        safety_enabled=probability is not None,
        tail_cushions=(1.0, 2.0, 3.0),
        model_id="v6-test",
        manifest={"current_profile_mutated": False, "runtime_enabled": False},
    )


def _observation(*, seat: str = "second", offset: int = 0) -> ActorObservation:
    hero = Board.from_rows(
        top=["Ah"], middle=["Kd"], bottom=["2s", "3s", "4s"]
    )
    opponent = (
        Board.from_rows(
            top=["Qh"],
            middle=["Jd", "Td"],
            bottom=["5s", "6s", "7s", "8s"],
        )
        if seat == "second"
        else Board.from_rows(
            top=["Qh"], middle=["Jd"], bottom=["5s", "6s", "7s"]
        )
    )
    used = {*hero.all_cards(), *opponent.all_cards()}
    remaining = [card for card in ALL_CARDS if card not in used]
    return ActorObservation(
        hero_board=hero,
        opponent_public_board=opponent,
        dealt_cards=tuple(remaining[offset : offset + 3]),
        hero_private_discards=(),
        seat=seat,
        street="T1",
        to_act_order=seat,
    )


def _sample(*, seat: str = "second") -> dict:
    observation = _observation(seat=seat)
    legal = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    sample = hu_policy_sample(
        observation.hero_board,
        observation.dealt_cards,
        legal,
        opponent_board=observation.opponent_public_board,
        dead_cards=observation.legacy_dead_cards(),
        seat=seat,
        to_act_order=seat,
    )
    sample["baseline_action_row_index"] = len(legal) // 2
    sample["policy_observation"] = observation.to_dict()
    return sample


def _proposal_key(model: HuM43JointModelV6, sample: dict):
    heads = model.predict_heads_sample(sample)
    return action_key_from_payload(sample["actions"][heads.proposal_index])


def test_v6_tie_break_is_action_key_and_is_action_permutation_invariant():
    sample = _sample()
    model = _model()
    baseline = sample["baseline_action_row_index"]
    expected = min(
        (
            action_key_from_payload(action)
            for index, action in enumerate(sample["actions"])
            if index != baseline
        ),
        key=lambda key: key.sort_key(),
    )
    assert _proposal_key(model, sample) == expected

    permuted = deepcopy(sample)
    baseline_key = action_key_from_payload(permuted["actions"][baseline])
    permuted["actions"] = list(reversed(permuted["actions"]))
    permuted["baseline_action_row_index"] = next(
        index
        for index, action in enumerate(permuted["actions"])
        if action_key_from_payload(action) == baseline_key
    )
    assert _proposal_key(model, permuted) == expected


def test_v6_tail_pool_and_safety_are_fail_closed():
    sample = _sample()
    safe = _model()
    heads = safe.predict_heads_sample(sample)
    baseline = sample["baseline_action_row_index"]
    assert heads.proposal_risk_eligible is True
    assert heads.action_score[baseline] == 0.0
    assert heads.action_score[heads.proposal_index] == 1.0
    decision = safe.select_action_index(sample)
    assert decision.override_fired is True
    assert decision.selected_index == heads.proposal_index
    assert decision.safety_probability == pytest.approx(0.8)

    unsafe = _model(tail=30.0)
    unsafe_heads = unsafe.predict_heads_sample(sample)
    assert unsafe_heads.proposal_risk_eligible is False
    assert unsafe_heads.action_score[baseline] == 0.0
    assert np.count_nonzero(unsafe_heads.action_score == 1.0) == 0
    assert unsafe.select_action_index(sample).selected_index == baseline


def test_v6_rejects_baseline_and_candidate_index_contract_violations():
    sample = _sample()
    model = _model()
    baseline = sample["baseline_action_row_index"]
    with pytest.raises(ValueError, match="disagrees"):
        model.predict_heads_sample(sample, baseline_index=(baseline + 1) % len(sample["actions"]))
    with pytest.raises(TypeError, match="candidate index"):
        model.predict_safety_probability(
            sample, candidate_index=True, baseline_index=baseline
        )
    with pytest.raises(IndexError, match="candidate index"):
        model.predict_safety_probability(
            sample, candidate_index=len(sample["actions"]), baseline_index=baseline
        )


def test_v6_runtime_projection_ignores_teacher_only_fields():
    sample = _sample()
    polluted = deepcopy(sample)
    polluted["opponent_private_discards"] = ["As"]
    polluted["teacher_value"] = 999999.0
    for action in polluted["actions"]:
        action["score"] = -999999.0
        action["paired_delta_vs_baseline"] = {"mean": 999999.0}
    model = _model()
    clean = model.predict_heads_sample(sample)
    dirty = model.predict_heads_sample(polluted)
    assert clean.proposal_index == dirty.proposal_index
    np.testing.assert_array_equal(clean.action_score, dirty.action_score)
    np.testing.assert_allclose(clean.base_delta, dirty.base_delta)


@pytest.mark.parametrize("attack", ("missing_observation", "dead_cards", "next_board", "hidden_nested"))
def test_v6_runtime_projection_rejects_untrusted_or_poisoned_inputs(attack):
    sample = _sample()
    if attack == "missing_observation":
        sample.pop("policy_observation")
    elif attack == "dead_cards":
        sample["dead_cards"] = ["As"]
    elif attack == "next_board":
        sample["actions"][0]["next_board"] = {"top": [], "middle": [], "bottom": []}
    else:
        sample["policy_observation"]["opponent_private_discards"] = ["As"]
    with pytest.raises(ValueError):
        _model().predict_heads_sample(sample)


def test_v6_tail_cushions_freeze_before_safety_only():
    raw = _model(probability=None)
    frozen = raw.with_frozen_tail_cushions((2.0, 3.0, 4.0))
    assert frozen.tail_cushions == (2.0, 3.0, 4.0)
    calibrated = frozen.with_frozen_safety(
        ConstantProbabilityEstimator(0.75), threshold=0.7, enabled=True
    )
    assert calibrated.tail_cushions == frozen.tail_cushions
    with pytest.raises(RuntimeError, match="before safety"):
        calibrated.with_frozen_tail_cushions((0.0, 0.0, 0.0))


def test_v6_save_is_no_clobber_and_loader_raw_dispatches(tmp_path):
    path = tmp_path / "candidate_model.pkl"
    model = _model()
    digest = model.save(path)
    loaded = load_hu_m43_joint_action_model(path)
    assert isinstance(loaded, HuM43JointModelV6)
    assert HuM43JointModelV6.load(path, expected_sha256=digest).model_id == "v6-test"
    with pytest.raises(FileExistsError):
        model.save(path)
    with pytest.raises(ValueError, match="together"):
        load_hu_m43_joint_action_model(path, expected_sha256=digest)

    envelope = pickle.loads(path.read_bytes())
    envelope["proposal_schema"] = "poisoned"
    poisoned = tmp_path / "poisoned.pkl"
    poisoned.write_bytes(pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL))
    with pytest.raises(ValueError, match="proposal schema"):
        HuM43JointModelV6.load(poisoned)


def test_v6_loader_dispatches_all_four_bound_inputs(tmp_path, monkeypatch):
    path = tmp_path / "candidate_model.pkl"
    digest = _model().save(path)
    calls = []
    runtime = types.ModuleType("ofc_regular.hu_m43_attempt04_runtime")

    def load_bound(source, **kwargs):
        calls.append((source, kwargs))
        return "bound-v6"

    runtime.load_bound_attempt04_v6_model = load_bound
    monkeypatch.setitem(sys.modules, "ofc_regular.hu_m43_attempt04_runtime", runtime)
    result = load_hu_m43_joint_action_model(
        path,
        expected_sha256=digest,
        freeze_manifest={"schema": "freeze"},
        training_manifest_path=tmp_path / "training.json",
        threshold_lock_path=tmp_path / "threshold.json",
    )
    assert result == "bound-v6"
    assert calls[0][1]["threshold_lock_path"] == tmp_path / "threshold.json"


def test_existing_t1_wrapper_preserves_first_seat_baseline_fallback():
    observation = _observation(seat="first")
    baseline_policy = _BaselinePolicy()
    expected = baseline_policy.choose_action_observation(observation)
    policy = HuM4T1SelectiveOverridePolicy(
        baseline_policy,
        action_value_model=_ExplodingModel(),
        safety_model=_ExplodingModel(),
        safety_probability_threshold=0.7,
        enabled=True,
        allowed_seats=("second",),
    )
    assert policy.choose_action_observation(observation) == expected
