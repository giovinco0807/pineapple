from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from ofc_regular.action_key import action_key_from_payload
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.cards import ALL_CARDS
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m43_joint_model_v4 import (
    HU_M43_V4_ACTION_SCORE_MODE,
    HU_M43_V4_SELECTION_SCORE,
    HuM43JointModelV4,
    V4CrossFitResult,
    V4FoldWorkerConfig,
    V4OofStatePrediction,
    V4PrecalibrationGateConfig,
    V4SafetyFitResult,
    build_v4_fold_training_arrays,
    build_v4_oof_safety_dataset,
    build_v4_training_manifest,
    evaluate_v4_precalibration_oof_gate,
    fit_v4_fold_worker_compatible,
    fit_v4_nested_crossfit,
    fit_v4_safety_calibrator,
    make_v4_fold_estimator_provider,
    select_v4_threshold,
    write_v4_training_manifest,
)
from ofc_regular.hu_m4_joint_model import (
    ConstantProbabilityEstimator,
    PairedDeltaRiskFoldEstimator,
    build_paired_action_features,
)
from ofc_regular.hu_m4_t1_policy import HuM4T1SelectiveOverridePolicy
from ofc_regular.hu_turn3_model import HU_FEATURE_DIM
from ofc_regular.policy import action_to_json
from ofc_regular.state import Board
from ofc_regular.train_hu_m4_joint_model import (
    M43FoldJobSpec,
    prepare_teacher_sample,
)


class _ConstantRegression:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, matrix):
        return np.full(np.asarray(matrix).shape[0], self.value, dtype=np.float64)


class _DifferenceSumRegression:
    def __init__(self, scale: float = 1.0) -> None:
        self.scale = float(scale)

    def predict(self, matrix):
        values = np.asarray(matrix, dtype=np.float64)
        difference = values[:, 2 * HU_FEATURE_DIM : 3 * HU_FEATURE_DIM]
        return self.scale * np.sum(difference, axis=1)


def _fold(index: int, *, varying_delta: bool = True) -> PairedDeltaRiskFoldEstimator:
    delta = (
        _DifferenceSumRegression(0.01 * (index + 1))
        if varying_delta
        else _ConstantRegression(float(index))
    )
    return PairedDeltaRiskFoldEstimator(
        delta_estimator=delta,
        positive_gain_estimator=_ConstantRegression(0.6),
        downside_p95_estimator=_ConstantRegression(5.0),
        downside_p99_estimator=_ConstantRegression(10.0),
        downside_max_estimator=_ConstantRegression(20.0),
        paired_feature_dim=4 * HU_FEATURE_DIM,
        fold_index=index,
    )


def _model(
    *, safety_probability: float | None = None, threshold: float = 1.0
) -> HuM43JointModelV4:
    estimator = (
        ConstantProbabilityEstimator(safety_probability)
        if safety_probability is not None
        else None
    )
    return HuM43JointModelV4(
        paired_fold_estimators=tuple(_fold(index) for index in range(5)),
        safety_estimator=estimator,
        safety_threshold=threshold,
        safety_enabled=estimator is not None,
        model_id="v4-test",
        manifest={"current_profile_mutated": False, "runtime_enabled": False},
    )


def _observation(offset: int, *, seat: str = "second") -> ActorObservation:
    hero = Board.from_rows(
        top=["Ah"], middle=["Kd"], bottom=["2s", "3s", "4s"]
    )
    opponent = Board.from_rows(
        top=["Qh"], middle=["Jd", "Td"], bottom=["5s", "6s", "7s", "8s"]
    )
    used = {*hero.all_cards(), *opponent.all_cards()}
    remaining = [card for card in ALL_CARDS if card not in used]
    dealt = tuple(remaining[offset : offset + 3])
    return ActorObservation(
        hero_board=hero,
        opponent_public_board=opponent,
        dealt_cards=dealt,
        hero_private_discards=(),
        seat=seat,  # type: ignore[arg-type]
        street="T1",
        to_act_order=seat,  # type: ignore[arg-type]
    )


def _paired_sample(seed: int, *, offset: int = 0):
    observation = _observation(offset)
    legal = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    actions = []
    for index, action in enumerate(legal):
        payload = action_to_json(observation.hero_board, action)
        # Non-monotone but deterministic labels; the model never consumes them.
        payload["score"] = float((index % 7) - 0.25 * (index % 3))
        payload["score_se"] = float(0.5 + 0.1 * (index % 4))
        actions.append(payload)
    baseline = len(actions) // 2
    baseline_score = float(actions[baseline]["score"])
    for index, action in enumerate(actions):
        delta = float(action["score"]) - baseline_score
        if index == baseline:
            se = p05 = p01 = minimum = 0.0
        else:
            se = 0.75 + 0.05 * (index % 4)
            p05 = delta - 5.0
            p01 = delta - 10.0
            minimum = delta - 20.0
        action["delta_se_vs_baseline"] = se
        action["paired_delta_vs_baseline"] = {
            "mean": delta,
            "standard_error": se,
            "p05": p05,
            "p01": p01,
            "min": minimum,
        }
    return prepare_teacher_sample(
        {
            "root_seed": seed,
            "hand_seed": seed + 100_000,
            "policy_observation": observation.to_dict(),
            "actions": actions,
            "baseline_action_row_index": baseline,
        }
    )


def _samples(count: int) -> list:
    return [_paired_sample(10_000 + index, offset=index * 3) for index in range(count)]


def _labeled_samples(
    count: int,
    *,
    seed_start: int,
    offset_start: int,
    safe_top: bool,
) -> list:
    model = _model()
    adjusted = []
    for index in range(count):
        sample = _paired_sample(
            seed_start + index, offset=offset_start + index
        )
        heads = model.predict_heads_sample(
            sample.policy_sample, baseline_index=sample.baseline_index
        )
        proposal = model.top_nonbaseline_index(
            sample.policy_sample,
            baseline_index=sample.baseline_index,
            predictions=heads,
        )
        action_count = len(sample.teacher_scores)
        delta = np.full(action_count, -1.0, dtype=np.float64)
        p95 = np.full(action_count, 30.0, dtype=np.float64)
        p99 = np.full(action_count, 45.0, dtype=np.float64)
        maximum = np.full(action_count, 55.0, dtype=np.float64)
        # Keep both safety classes available to the fit split.
        for action_index in range(action_count):
            if action_index != sample.baseline_index and action_index % 2 == 0:
                delta[action_index] = 1.0
                p95[action_index] = 5.0
                p99[action_index] = 10.0
                maximum[action_index] = 20.0
        if safe_top:
            delta[proposal] = 2.0
            p95[proposal] = 5.0
            p99[proposal] = 10.0
            maximum[proposal] = 20.0
        else:
            delta[proposal] = -1.0
            p95[proposal] = 30.0
            p99[proposal] = 45.0
            maximum[proposal] = 55.0
        delta[sample.baseline_index] = 0.0
        p95[sample.baseline_index] = 0.0
        p99[sample.baseline_index] = 0.0
        maximum[sample.baseline_index] = 0.0
        adjusted.append(
            replace(
                sample,
                teacher_scores=delta.copy(),
                teacher_paired_delta_mean=delta,
                downside_loss_p95=p95,
                downside_loss_p99=p99,
                downside_loss_max=maximum,
            )
        )
    return adjusted


def _spec(samples, *, index: int = 0, fold: int = 0, seed: int = 17):
    return M43FoldJobSpec(
        job_index=index,
        kind="outer_runtime",
        outer_fold=fold,
        inner_fold=None,
        estimator_fold_index=fold,
        estimator_seed=seed,
        fit_samples=len(samples),
        fit_identity_sha256="0" * 64,
        outer_validation_samples=1,
        outer_validation_identity_sha256="1" * 64,
        inner_validation_samples=0,
        inner_validation_identity_sha256=None,
        outer_assignment_sha256="2" * 64,
        inner_assignment_sha256=None,
    )


def test_v4_fold_arrays_exclude_baseline_and_balance_each_state():
    samples = _samples(4)
    arrays = build_v4_fold_training_arrays(samples, paired_se_floor=0.5)

    assert arrays.manifest["baseline_rows_included"] is False
    assert arrays.features.shape[0] == sum(
        len(sample.teacher_scores) - 1 for sample in samples
    )
    assert np.all(arrays.action_indices != arrays.baseline_indices)
    for index in range(len(samples)):
        assert np.sum(arrays.state_indices == index) == len(samples[index].teacher_scores) - 1
        assert np.sum(arrays.weights[arrays.state_indices == index]) == pytest.approx(1.0)


def test_v4_fold_worker_is_existing_artifact_type_and_deterministic():
    samples = _samples(3)
    spec = _spec(samples, fold=2, seed=31)
    config = V4FoldWorkerConfig(iterations=2, max_leaf_nodes=3, learning_rate=0.1)

    first = fit_v4_fold_worker_compatible(spec, samples, config=config)
    second = fit_v4_fold_worker_compatible(spec, samples, config=config)
    assert isinstance(first, PairedDeltaRiskFoldEstimator)
    assert first.fold_index == 2
    matrix = build_paired_action_features(
        samples[0].policy_sample, baseline_index=samples[0].baseline_index
    )
    for left, right in zip(first.predict(matrix), second.predict(matrix), strict=True):
        np.testing.assert_array_equal(left, right)

    provider = make_v4_fold_estimator_provider(config)
    provided = provider(spec, samples)
    assert isinstance(provided, PairedDeltaRiskFoldEstimator)


def test_v4_proposal_is_always_nonbaseline_and_duck_runtime_fails_closed():
    sample = _samples(1)[0]
    baseline = sample.baseline_index
    safe = _model(safety_probability=0.8, threshold=0.7)
    heads = safe.predict_heads_sample(
        sample.policy_sample, baseline_index=baseline
    )

    assert heads.proposal_score[baseline] == 0.0
    assert heads.action_score[baseline] == 0.0
    assert np.max(np.delete(heads.action_score, baseline)) == HU_M43_V4_SELECTION_SCORE
    proposal = safe.top_nonbaseline_index(
        sample.policy_sample,
        baseline_index=baseline,
        predictions=heads,
    )
    assert proposal != baseline
    assert safe.should_override(
        sample.policy_sample,
        candidate_index=proposal,
        baseline_index=baseline,
    )
    decision = safe.select_action_index(
        sample.policy_sample, baseline_index=baseline
    )
    assert decision.proposal_index != baseline
    assert decision.selected_index == proposal
    assert decision.override_fired

    closed = _model()
    closed_decision = closed.select_action_index(
        sample.policy_sample, baseline_index=baseline
    )
    assert closed_decision.proposal_index != baseline
    assert closed_decision.selected_index == baseline
    assert not closed_decision.override_fired


def test_v4_action_scores_and_safety_are_permutation_invariant():
    sample = _samples(1)[0]
    model = _model(safety_probability=0.75, threshold=0.7)
    original = model.predict_heads_sample(
        sample.policy_sample, baseline_index=sample.baseline_index
    )
    original_proposal = model.top_nonbaseline_index(
        sample.policy_sample,
        baseline_index=sample.baseline_index,
        predictions=original,
    )
    original_keys = [
        action_key_from_payload(action).to_token()
        for action in sample.policy_sample["actions"]
    ]

    reversed_sample = deepcopy(sample.policy_sample)
    reversed_sample["actions"] = list(reversed(reversed_sample["actions"]))
    reversed_sample["baseline_action_row_index"] = (
        len(reversed_sample["actions"]) - 1 - sample.baseline_index
    )
    reversed_sample["best_action"] = (
        len(reversed_sample["actions"]) - 1 - int(reversed_sample["best_action"])
    )
    reversed_heads = model.predict_heads_sample(reversed_sample)
    reversed_proposal = model.top_nonbaseline_index(
        reversed_sample,
        baseline_index=reversed_sample["baseline_action_row_index"],
        predictions=reversed_heads,
    )
    reversed_keys = [
        action_key_from_payload(action).to_token()
        for action in reversed_sample["actions"]
    ]

    for field in ("proposal_score", "action_score", "delta_vs_baseline"):
        left = dict(zip(original_keys, getattr(original, field), strict=True))
        right = dict(zip(reversed_keys, getattr(reversed_heads, field), strict=True))
        assert left.keys() == right.keys()
        for key in left:
            assert left[key] == pytest.approx(right[key], abs=1e-9)
    assert original_keys[original_proposal] == reversed_keys[reversed_proposal]
    left_probability = model.predict_safety_probability(
        sample.policy_sample,
        candidate_index=original_proposal,
        baseline_index=sample.baseline_index,
    )
    right_probability = model.predict_safety_probability(
        reversed_sample,
        candidate_index=reversed_proposal,
        baseline_index=reversed_sample["baseline_action_row_index"],
    )
    assert left_probability == pytest.approx(right_probability)


def test_v4_runtime_predictions_ignore_teacher_values_and_lcb():
    sample = _samples(1)[0]
    model = _model()
    assert model.action_score_mode == HU_M43_V4_ACTION_SCORE_MODE
    original = model.predict_heads_sample(
        sample.policy_sample, baseline_index=sample.baseline_index
    )
    changed = deepcopy(sample.policy_sample)
    changed["teacher_ev"] = 1.0e9
    changed["teacher_lcb"] = 1.0e9
    for index, action in enumerate(changed["actions"]):
        action["score"] = float(100_000 - index)
        action["teacher_lcb"] = float(index)
    altered = model.predict_heads_sample(
        changed, baseline_index=sample.baseline_index
    )
    np.testing.assert_array_equal(original.proposal_score, altered.proposal_score)
    np.testing.assert_array_equal(original.action_score, altered.action_score)


def test_v4_serialization_hash_and_prediction_roundtrip(tmp_path):
    sample = _samples(1)[0]
    model = _model(safety_probability=0.8, threshold=0.7)
    first = tmp_path / "v4.pkl"
    digest = model.save(first)
    assert digest == hashlib.sha256(first.read_bytes()).hexdigest()
    loaded = HuM43JointModelV4.load(first, expected_sha256=digest)
    second = tmp_path / "v4-second.pkl"
    # Repeated publication of the same frozen in-memory object is byte stable.
    # A load/re-pickle cycle is not used as an artifact identity operation;
    # runtime verifies the hash of the original frozen bytes instead.
    second_digest = model.save(second)
    assert second_digest == digest
    assert second.read_bytes() == first.read_bytes()
    np.testing.assert_array_equal(
        loaded.predict_sample_with_baseline(
            sample.policy_sample, baseline_index=sample.baseline_index
        ),
        model.predict_sample_with_baseline(
            sample.policy_sample, baseline_index=sample.baseline_index
        ),
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        HuM43JointModelV4.load(first, expected_sha256="f" * 64)


def _gate_samples_and_oof(count: int = 5):
    base = _samples(count)
    model = _model()
    adjusted = []
    oof = []
    for index, sample in enumerate(base):
        heads = model.predict_heads_sample(
            sample.policy_sample, baseline_index=sample.baseline_index
        )
        proposal = model.top_nonbaseline_index(
            sample.policy_sample,
            baseline_index=sample.baseline_index,
            predictions=heads,
        )
        action_count = len(sample.teacher_scores)
        delta = np.full(action_count, -1.0, dtype=np.float64)
        p95 = np.full(action_count, 30.0, dtype=np.float64)
        p99 = np.full(action_count, 45.0, dtype=np.float64)
        maximum = np.full(action_count, 55.0, dtype=np.float64)
        # Ensure both classes exist in every state, while four of five top
        # proposals are positive and tail-safe.
        for action_index in range(action_count):
            if action_index != sample.baseline_index and action_index % 2 == 0:
                delta[action_index] = 1.0
                p95[action_index] = 5.0
                p99[action_index] = 10.0
                maximum[action_index] = 20.0
        if index < count - 1:
            delta[proposal] = 2.0
            p95[proposal] = 5.0
            p99[proposal] = 10.0
            maximum[proposal] = 20.0
        else:
            delta[proposal] = -1.0
        delta[sample.baseline_index] = 0.0
        p95[sample.baseline_index] = 0.0
        p99[sample.baseline_index] = 0.0
        maximum[sample.baseline_index] = 0.0
        adjusted_sample = replace(
            sample,
            teacher_scores=delta.copy(),
            teacher_paired_delta_mean=delta,
            downside_loss_p95=p95,
            downside_loss_p99=p99,
            downside_loss_max=maximum,
        )
        adjusted.append(adjusted_sample)
        oof.append(
            V4OofStatePrediction(
                sample_index=index,
                fold_index=index,
                predictions=heads,
                identity_excluded_from_fit=True,
            )
        )
    return adjusted, oof


def test_v4_all_action_oof_safety_rows_and_precalibration_gate():
    samples, oof = _gate_samples_and_oof()
    dataset = build_v4_oof_safety_dataset(samples, oof)
    assert dataset.manifest["baseline_rows"] == 0
    assert dataset.manifest["row_source"] == "all_nonbaseline_oof_actions"
    for state in range(len(samples)):
        indices = [
            index for index, row in enumerate(dataset.rows) if row["state_index"] == state
        ]
        assert np.sum(dataset.weights[indices]) == pytest.approx(1.0)
        assert all(
            dataset.rows[index]["action_index"]
            != dataset.rows[index]["baseline_index"]
            for index in indices
        )
    config = V4PrecalibrationGateConfig(
        expected_states=5,
        expected_folds=5,
        minimum_states_per_fold=1,
        minimum_proposal_positive_rate=0.60,
        minimum_tail_safe_proposals=3,
        minimum_tail_unsafe_proposals=1,
        minimum_all_action_safe_rows=5,
        minimum_all_action_unsafe_rows=5,
    )
    report = evaluate_v4_precalibration_oof_gate(
        samples, oof, dataset, config=config
    )
    json.dumps(report)
    assert report["status"] == "go"
    assert report["metrics"]["nonbaseline_proposals"] == 5
    assert report["metrics"]["proposal_positive_count"] == 4
    assert report["gates"]["baseline_score_exact_zero"]
    assert report["calibration_opened"] is False

    leaked = list(oof)
    leaked[0] = replace(leaked[0], identity_excluded_from_fit=False)
    failed = evaluate_v4_precalibration_oof_gate(
        samples,
        leaked,
        build_v4_oof_safety_dataset(samples, leaked),
        config=config,
    )
    assert failed["status"] == "no_go"
    assert not failed["gates"]["all_oof_identities_excluded_from_fit"]


def test_v4_exact_fold_plan_accepts_existing_provider_interface():
    samples = _samples(10)
    calls = []

    def provider(spec, fit_samples):
        calls.append((spec.job_index, len(fit_samples)))
        return _fold(spec.estimator_fold_index, varying_delta=False)

    gate = V4PrecalibrationGateConfig(
        expected_states=10,
        expected_folds=2,
        minimum_states_per_fold=4,
        minimum_proposal_positive_rate=0.0,
        minimum_tail_safe_proposals=0,
        minimum_tail_unsafe_proposals=0,
        minimum_all_action_safe_rows=0,
        minimum_all_action_unsafe_rows=0,
        require_positive_mean_teacher_delta=False,
    )
    result = fit_v4_nested_crossfit(
        samples,
        cross_fit_folds=2,
        gate_config=gate,
        fold_estimator_provider=provider,
    )
    assert len(calls) == 6
    assert sorted(index for index, _count in calls) == list(range(6))
    assert result.report["exact_outer_inner_job_grid"]
    assert len(result.oof_predictions) == len(samples)
    assert result.safety_dataset.manifest["states"] == len(samples)


class _BaselinePolicy:
    def choose_action_observation(self, observation, **_kwargs):
        return generate_turn_actions(
            observation.hero_board, observation.dealt_cards
        )[0]


def test_v4_duck_methods_work_with_existing_opt_in_policy_without_default_change():
    observation = _observation(0)
    model = _model(safety_probability=0.8, threshold=0.7)
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
    assert chosen != baseline
    assert decisions[-1]["override_fired"] is True
    assert decisions[-1]["candidate_action_index"] != decisions[-1]["baseline_action_index"]


def test_v4_canonical_tie_break_uses_semantic_action_key():
    sample = _samples(1)[0]
    model = HuM43JointModelV4(
        paired_fold_estimators=tuple(
            _fold(index, varying_delta=False) for index in range(5)
        )
    )
    heads = model.predict_heads_sample(
        sample.policy_sample, baseline_index=sample.baseline_index
    )
    proposal = model.top_nonbaseline_index(
        sample.policy_sample,
        baseline_index=sample.baseline_index,
        predictions=heads,
    )
    expected = min(
        (
            index
            for index in range(len(sample.teacher_scores))
            if index != sample.baseline_index
        ),
        key=lambda index: action_key_from_payload(
            sample.policy_sample["actions"][index]
        ).sort_key(),
    )
    assert proposal == expected


def test_v4_safety_fit_uses_train_oof_plus_fresh_safety_fit_only():
    samples, oof = _gate_samples_and_oof()
    train_oof = build_v4_oof_safety_dataset(samples, oof)
    safety_fit_samples = _labeled_samples(
        5, seed_start=20_000, offset_start=20, safe_top=True
    )

    result = fit_v4_safety_calibrator(
        _model(), train_oof, safety_fit_samples
    )

    assert result.report["status"] == "fit_threshold_unselected"
    assert result.report["sources"]["train_oof"]["states"] == 5
    assert result.report["sources"]["calibration.safety_fit"]["states"] == 5
    assert result.report["sources"]["calibration.threshold_lock"] == {
        "used": False,
        "labels_opened": False,
    }
    assert result.report["sources"]["locked_holdout"] == {
        "used": False,
        "labels_opened": False,
    }
    assert result.report["state_balanced_weights"] is True
    assert result.model.safety_estimator is not None
    assert result.model.safety_enabled is False
    assert result.model.safety_threshold == 1.0

    with pytest.raises(ValueError, match="identity overlap"):
        fit_v4_safety_calibrator(_model(), train_oof, samples)


def test_v4_threshold_lock_selects_fixed_grid_and_fails_closed_on_no_go():
    samples, oof = _gate_samples_and_oof()
    train_oof = build_v4_oof_safety_dataset(samples, oof)
    base_fit = V4SafetyFitResult(
        model=_model().with_frozen_safety(
            ConstantProbabilityEstimator(0.8), threshold=1.0, enabled=False
        ),
        report={"status": "test_fit"},
        fit_seed_values=train_oof.seed_values,
        fit_observation_fingerprints=train_oof.observation_fingerprints,
    )
    lock = _labeled_samples(
        12, seed_start=30_000, offset_start=25, safe_top=True
    )

    selected = select_v4_threshold(
        base_fit,
        lock,
        thresholds=(1.0, 0.0, 0.9, 0.5, 0.5),
        minimum_fires=10,
    )
    assert selected.report["status"] == "go"
    assert selected.report["thresholds"] == [0.0, 0.5, 0.9, 1.0]
    assert selected.report["selected_threshold"] == 0.5
    assert selected.report["selected_metrics"]["fires"] == 12
    assert selected.report["selected_metrics"]["false_positive_rate"] == 0.0
    assert selected.report["selected_metrics"]["max_loss"] == 20.0
    assert selected.model.safety_enabled is True
    assert selected.model.safety_threshold == 0.5
    assert selected.report["locked_holdout"] == {
        "used": False,
        "labels_opened": False,
    }

    no_go = select_v4_threshold(
        base_fit,
        lock,
        thresholds=(0.9, 1.0),
        minimum_fires=10,
    )
    assert no_go.report["status"] == "no_go"
    assert no_go.report["selected_threshold"] == 1.0
    assert no_go.report["selected_metrics"]["fires"] == 0
    assert no_go.model.safety_enabled is False

    with pytest.raises(ValueError, match="identity overlap"):
        select_v4_threshold(
            base_fit,
            samples,
            thresholds=(0.5,),
            minimum_fires=1,
        )


def test_v4_training_manifest_remains_opt_in_and_locked_unopened(tmp_path):
    samples, oof = _gate_samples_and_oof()
    safety = build_v4_oof_safety_dataset(samples, oof)
    crossfit = V4CrossFitResult(
        model=_model(),
        oof_predictions=tuple(oof),
        safety_dataset=safety,
        precalibration_report={"status": "go"},
        report={"schema": "test-crossfit", "status": "pass"},
    )
    fit = V4SafetyFitResult(
        model=_model().with_frozen_safety(
            ConstantProbabilityEstimator(0.8), threshold=1.0, enabled=False
        ),
        report={"status": "fit_threshold_unselected"},
        fit_seed_values=safety.seed_values,
        fit_observation_fingerprints=safety.observation_fingerprints,
    )
    lock = _labeled_samples(
        12, seed_start=40_000, offset_start=25, safe_top=True
    )
    selected = select_v4_threshold(
        fit, lock, thresholds=(0.5, 1.0), minimum_fires=10
    )

    manifest = build_v4_training_manifest(crossfit, fit, selected)
    assert manifest["promotion_status"] == "candidate_ready_for_freeze"
    assert manifest["baseline_training_rows_included"] is False
    assert manifest["runtime_teacher_inputs"] is False
    assert manifest["locked_holdout"] == {
        "status": "not_evaluated_pre_freeze",
        "labels_opened": False,
    }
    assert manifest["runtime"] == {
        "current_profile_mutated": False,
        "policy_activated": False,
        "full_replacement": False,
    }
    destination = tmp_path / "v4-training-manifest.json"
    digest = write_v4_training_manifest(destination, manifest)
    assert digest == hashlib.sha256(destination.read_bytes()).hexdigest()
