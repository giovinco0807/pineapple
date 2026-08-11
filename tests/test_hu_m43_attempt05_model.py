from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from ofc_regular.action_key import action_key_from_payload
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.cards import ALL_CARDS
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m43_attempt05_model import (
    DeepSetsFoldPredictor,
    HuM43Attempt05Model,
    LambdaRankFoldPredictor,
    build_deepsets_network,
    encode_deepsets_runtime_sample,
)
from ofc_regular.hu_turn3_model import HU_FEATURE_DIM, hu_policy_sample
from ofc_regular.state import Board


class _ConstantRegression:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, matrix):
        return np.full(np.asarray(matrix).shape[0], self.value, dtype=np.float64)


class _ConstantProbability:
    classes_ = np.asarray([0, 1])

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict_proba(self, matrix):
        positive = np.full(np.asarray(matrix).shape[0], self.value)
        return np.column_stack((1.0 - positive, positive))


def _observation(*, seat: str = "second", offset: int = 0) -> ActorObservation:
    hero = Board.from_rows(top=["Ah"], middle=["Kd"], bottom=["2s", "3s", "4s"])
    opponent = (
        Board.from_rows(
            top=["Qh"], middle=["Jd", "Td"], bottom=["5s", "6s", "7s", "8s"]
        )
        if seat == "second"
        else Board.from_rows(top=["Qh"], middle=["Jd"], bottom=["5s", "6s", "7s"])
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
    sample["baseline_action_key"] = action_key_from_payload(
        sample["actions"][sample["baseline_action_row_index"]]
    ).to_token()
    sample["policy_observation"] = observation.to_dict()
    return sample


def _tree_fold(index: int, *, gain: float = 0.9, tail: float = 5.0):
    return LambdaRankFoldPredictor(
        ranker=_ConstantRegression(1.0),
        gain_head=_ConstantProbability(gain),
        tail_p95_head=_ConstantRegression(tail),
        tail_p99_head=_ConstantRegression(2.0 * tail),
        tail_max_head=_ConstantRegression(3.0 * tail),
        paired_feature_dim=4 * HU_FEATURE_DIM,
        fold_index=index,
    )


def _model(*, enabled: bool = False) -> HuM43Attempt05Model:
    return HuM43Attempt05Model(
        family="lambda_rank",
        fold_predictors=tuple(_tree_fold(index) for index in range(5)),
        conformal_cushions=(1.0, 2.0, 3.0),
        gain_threshold=0.7,
        uncertainty_max=0.1,
        winner_frozen=enabled,
        runtime_enabled=enabled,
        model_id="attempt05-test",
    )


def test_attempt05_model_argmax_uses_action_key_and_unfrozen_falls_back():
    sample = _sample()
    model = _model(enabled=False)
    baseline = sample["baseline_action_row_index"]
    expected = min(
        (
            action_key_from_payload(action)
            for index, action in enumerate(sample["actions"])
            if index != baseline
        ),
        key=lambda key: key.sort_key(),
    )
    heads = model.predict_heads_sample(sample)
    assert action_key_from_payload(sample["actions"][heads.proposal_index]) == expected
    assert model.select_action_index(sample).selected_index == baseline
    assert model.runtime_contract["runtime_teacher_ev"] is False
    assert model.runtime_contract["runtime_teacher_lcb"] is False
    assert model.runtime_contract["profile_runtime_feature"] is False

    permuted = deepcopy(sample)
    baseline_key = sample["baseline_action_key"]
    permuted["actions"] = list(reversed(permuted["actions"]))
    permuted["baseline_action_row_index"] = next(
        index
        for index, action in enumerate(permuted["actions"])
        if action_key_from_payload(action).to_token() == baseline_key
    )
    assert action_key_from_payload(
        permuted["actions"][model.predict_heads_sample(permuted).proposal_index]
    ) == expected


def test_attempt05_external_search_candidate_is_gated_not_model_argmax():
    sample = _sample()
    model = _model(enabled=True)
    heads = model.predict_heads_sample(sample)
    baseline = sample["baseline_action_row_index"]
    candidate = next(
        index
        for index in range(len(sample["actions"]))
        if index not in {baseline, heads.proposal_index}
    )
    key = action_key_from_payload(sample["actions"][candidate]).to_token()
    evaluation = model.evaluate_candidate(
        sample, candidate_index=candidate, candidate_action_key=key
    )
    assert evaluation.candidate_index == candidate
    assert evaluation.gate_eligible is True
    decision = model.select_external_candidate_index(
        sample, candidate_index=candidate, candidate_action_key=key
    )
    assert decision.override_fired is True
    assert decision.selected_index == candidate
    assert decision.proposal_index != heads.proposal_index

    wrong_key = action_key_from_payload(sample["actions"][heads.proposal_index]).to_token()
    fail_closed = model.select_external_candidate_index(
        sample, candidate_index=candidate, candidate_action_key=wrong_key
    )
    assert fail_closed.override_fired is False
    assert fail_closed.selected_index == baseline

    missing_key = model.select_external_candidate_index(
        sample, candidate_index=candidate
    )
    assert missing_key.override_fired is False
    assert missing_key.selected_index == baseline
    assert missing_key.reason == "fail_closed_candidate_error"
    with pytest.raises(ValueError, match="candidate ActionKey is required"):
        model.evaluate_candidate(sample, candidate_index=candidate)


def test_attempt05_runtime_requires_exact_baseline_key_and_complete_legal_set():
    sample = _sample()
    missing_key = deepcopy(sample)
    del missing_key["baseline_action_key"]
    with pytest.raises(ValueError, match="baseline ActionKey is required"):
        _model().predict_heads_sample(missing_key)

    incomplete = deepcopy(sample)
    baseline = incomplete["baseline_action_row_index"]
    remove = next(index for index in range(len(incomplete["actions"])) if index != baseline)
    incomplete["actions"].pop(remove)
    if remove < baseline:
        incomplete["baseline_action_row_index"] -= 1
    with pytest.raises(ValueError, match="complete legal action set"):
        _model().predict_heads_sample(incomplete)


def test_attempt05_unauthorized_first_seat_and_poisoned_observation_fall_back():
    first = _sample(seat="first")
    assert _model(enabled=True).select_action_index(first).selected_index == first[
        "baseline_action_row_index"
    ]
    polluted = _sample()
    polluted["policy_observation"]["opponent_private_discards"] = ["As"]
    decision = _model(enabled=True).select_external_candidate_index(
        polluted, candidate_index=0
    )
    assert decision.override_fired is False
    assert decision.selected_index == polluted["baseline_action_row_index"]


def test_attempt05_teacher_fields_do_not_change_runtime_prediction():
    clean = _sample()
    polluted = deepcopy(clean)
    polluted["teacher_ev"] = 999999.0
    polluted["teacher_lcb"] = 999999.0
    polluted["provenance"] = {"root_profile": "oracle_hidden_profile"}
    polluted["opponent_private_discards"] = ["As"]
    for action in polluted["actions"]:
        action["score"] = -999999.0
        action["paired_delta_vs_baseline"] = {"mean": 999999.0}
    clean_heads = _model().predict_heads_sample(clean)
    dirty_heads = _model().predict_heads_sample(polluted)
    assert clean_heads.proposal_index == dirty_heads.proposal_index
    np.testing.assert_allclose(clean_heads.rank_score, dirty_heads.rank_score)
    np.testing.assert_allclose(clean_heads.gain_probability, dirty_heads.gain_probability)


def test_deepsets_encoder_is_card_order_invariant_and_baseline_explicit():
    sample = _sample()
    baseline = sample["baseline_action_row_index"]
    state, actions, context = encode_deepsets_runtime_sample(
        sample, baseline_index=baseline
    )
    permuted = deepcopy(sample)
    permuted["actions"] = list(reversed(permuted["actions"]))
    baseline_key = sample["baseline_action_key"]
    permuted_baseline = next(
        index
        for index, action in enumerate(permuted["actions"])
        if action_key_from_payload(action).to_token() == baseline_key
    )
    permuted_state, permuted_actions, permuted_context = encode_deepsets_runtime_sample(
        permuted, baseline_index=permuted_baseline
    )
    np.testing.assert_allclose(np.sum(state, axis=0), np.sum(permuted_state, axis=0))
    np.testing.assert_allclose(context[0], permuted_context[0])
    original_by_key = {
        action_key_from_payload(action).to_token(): np.sum(actions[index], axis=0)
        for index, action in enumerate(sample["actions"])
    }
    permuted_by_key = {
        action_key_from_payload(action).to_token(): np.sum(permuted_actions[index], axis=0)
        for index, action in enumerate(permuted["actions"])
    }
    assert original_by_key.keys() == permuted_by_key.keys()
    for key in original_by_key:
        np.testing.assert_allclose(original_by_key[key], permuted_by_key[key])

    other_baseline = next(index for index in range(len(sample["actions"])) if index != baseline)
    other_state, _actions, _context = encode_deepsets_runtime_sample(
        sample, baseline_index=other_baseline
    )
    assert not np.array_equal(np.sum(state, axis=0), np.sum(other_state, axis=0))


def test_deepsets_fold_forward_shapes_and_artifact_no_clobber(tmp_path):
    torch = pytest.importorskip("torch")
    sample = _sample()
    net = build_deepsets_network(
        torch, token_hidden_dim=32, trunk_hidden_dim=64, dropout=0.05
    )
    state_dict = {key: value.detach().cpu() for key, value in net.state_dict().items()}
    fold = DeepSetsFoldPredictor(state_dict=state_dict, fold_index=0)
    output = fold.predict(sample, baseline_index=sample["baseline_action_row_index"])
    assert output.rank_score.shape == (len(sample["actions"]),)
    assert np.all((output.gain_probability >= 0.0) & (output.gain_probability <= 1.0))

    path = tmp_path / "attempt05.pkl"
    model = _model()
    digest = model.save(path)
    loaded = HuM43Attempt05Model.load(path, expected_sha256=digest)
    assert loaded.model_id == model.model_id
    with pytest.raises(FileExistsError):
        model.save(path)
    with pytest.raises(ValueError, match="SHA-256"):
        HuM43Attempt05Model.load(path, expected_sha256="0" * 64)

    deep_model = HuM43Attempt05Model(
        family="deepsets",
        fold_predictors=tuple(
            DeepSetsFoldPredictor(state_dict=state_dict, fold_index=index)
            for index in range(5)
        ),
        runtime_enabled=False,
        winner_frozen=False,
        model_id="attempt05-deep-pickle-test",
    )
    # Populate every lazy cache before serialization; caches must be excluded.
    deep_model.predict_heads_sample(sample)
    deep_path = tmp_path / "attempt05_deep.pkl"
    deep_digest = deep_model.save(deep_path)
    loaded_deep = HuM43Attempt05Model.load(
        deep_path, expected_sha256=deep_digest
    )
    assert loaded_deep.family == "deepsets"
    assert loaded_deep.predict_heads_sample(sample).rank_score.shape == output.rank_score.shape
