"""Opt-in M4.3 attempt-02 joint model with proposal/safety separation.

This module is intentionally independent from the frozen v3 implementation.
It reuses the versioned paired-fold estimator payload so the existing 30-job
fold artifact transport can carry v4 estimators, but changes the training and
runtime semantics:

* the exact baseline row is excluded from every paired regressor fit;
* proposal ordering is the mean centered paired-delta prediction only;
* a non-baseline proposal is always emitted, even when every expected delta is
  negative;
* predicted tails, positive probability, and fold disagreement are safety
  features and never subtract directly from the proposal score;
* OOF safety rows cover every non-baseline action with equal total weight per
  information set.

Teacher values and paired-future labels appear only in the offline dataset and
pre-calibration audit helpers.  Runtime prediction accepts the policy sample
and explicit baseline mapping only; teacher EV/LCB is not an input or gate.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .action_key import action_key_from_payload
from .hu_m4_joint_model import (
    ConstantProbabilityEstimator,
    HU_M4_PAIRED_ACTION_FEATURE_SCHEMA,
    PairedDeltaRiskFoldEstimator,
    build_paired_action_features_matrix,
)
from .hu_turn3_model import HU_FEATURE_DIM, sample_to_matrix
from .train_hu_m4_joint_model import (
    M43FoldEstimatorProvider,
    M43FoldJobSpec,
    PreparedTeacherSample,
    build_m43_fold_training_plan,
)


HU_M43_V4_MODEL_SCHEMA = "hu_m43_t1_joint_model_v4"
HU_M43_V4_ARTIFACT_SCHEMA = "hu_m43_t1_joint_model_v4_pickle_v1"
HU_M43_V4_PROPOSAL_SCHEMA = "centered_fold_mean_delta_v1"
HU_M43_V4_ACTION_SCORE_MODE = "centered_fold_mean_delta_then_safety_gate_v4"
HU_M43_V4_SAFETY_FEATURE_SCHEMA = "hu_m43_t1_all_action_safety_features_v4"
HU_M43_V4_SAFETY_DATASET_SCHEMA = "hu_m43_t1_oof_safety_rows_v4"
HU_M43_V4_PRECAL_GATE_SCHEMA = "hu_m43_t1_precalibration_oof_gate_v4"
HU_M43_V4_CROSSFIT_SCHEMA = "hu_m43_t1_nested_crossfit_v4"
HU_M43_V4_WORKER_SCHEMA = "hu_m43_t1_fold_worker_config_v4"
HU_M43_V4_SAFETY_FIT_SCHEMA = "hu_m43_t1_safety_calibrator_fit_v4"
HU_M43_V4_THRESHOLD_SCHEMA = "hu_m43_t1_threshold_lock_selection_v4"
HU_M43_V4_TRAINING_MANIFEST_SCHEMA = "hu_m43_t1_training_manifest_v4"

HU_M43_V4_SAFETY_FEATURE_DIM = 15
HU_M43_V4_SELECTION_SCORE = 1.0
HU_M43_V4_TEACHER_STATUS = (
    "offline_diagnostic_only_not_realized_match_ev_not_runtime_gate"
)


@dataclass(frozen=True)
class V4FoldTrainingArrays:
    """Baseline-free arrays for one v4 paired-fold fit."""

    features: np.ndarray
    delta: np.ndarray
    soft_positive: np.ndarray
    downside_p95: np.ndarray
    downside_p99: np.ndarray
    downside_max: np.ndarray
    weights: np.ndarray
    state_indices: np.ndarray
    action_indices: np.ndarray
    baseline_indices: np.ndarray
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class V4FoldWorkerConfig:
    """Frozen hyperparameters accepted by the worker-compatible fit call."""

    paired_se_floor: float = 0.50
    huber_alpha: float = 0.90
    iterations: int = 150
    max_leaf_nodes: int = 31
    learning_rate: float = 0.05
    schema: str = HU_M43_V4_WORKER_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != HU_M43_V4_WORKER_SCHEMA:
            raise ValueError(f"unsupported v4 worker schema: {self.schema!r}")
        if not math.isfinite(self.paired_se_floor) or self.paired_se_floor <= 0.0:
            raise ValueError("paired_se_floor must be finite and positive")
        if not 0.5 <= self.huber_alpha < 1.0:
            raise ValueError("huber_alpha must be in [0.5, 1.0)")
        if self.iterations < 1 or self.max_leaf_nodes < 2:
            raise ValueError("iterations/max_leaf_nodes are invalid")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "paired_se_floor": float(self.paired_se_floor),
            "huber_alpha": float(self.huber_alpha),
            "iterations": int(self.iterations),
            "max_leaf_nodes": int(self.max_leaf_nodes),
            "learning_rate": float(self.learning_rate),
            "baseline_training_rows_included": False,
            "proposal_score": HU_M43_V4_PROPOSAL_SCHEMA,
            "downside_objective": "huber_on_declared_tail_target",
        }


def build_v4_fold_training_arrays(
    samples: Sequence[PreparedTeacherSample],
    *,
    paired_se_floor: float = 0.50,
) -> V4FoldTrainingArrays:
    """Build per-action targets while excluding every baseline row.

    Precision weights are normalized over the non-baseline actions of each
    state, so every information set contributes total weight one.  A baseline
    with paired SE zero therefore cannot absorb most of the training mass.
    """

    if not samples:
        raise ValueError("v4 fold training requires samples")
    if not math.isfinite(paired_se_floor) or paired_se_floor <= 0.0:
        raise ValueError("paired_se_floor must be finite and positive")
    feature_blocks: list[np.ndarray] = []
    delta_blocks: list[np.ndarray] = []
    positive_blocks: list[np.ndarray] = []
    p95_blocks: list[np.ndarray] = []
    p99_blocks: list[np.ndarray] = []
    max_blocks: list[np.ndarray] = []
    weight_blocks: list[np.ndarray] = []
    state_blocks: list[np.ndarray] = []
    action_blocks: list[np.ndarray] = []
    baseline_blocks: list[np.ndarray] = []
    state_rows: list[dict[str, Any]] = []
    for state_index, sample in enumerate(samples):
        _require_v4_targets(sample)
        matrix, _targets = sample_to_matrix(sample.policy_sample)
        paired = build_paired_action_features_matrix(
            matrix, baseline_index=sample.baseline_index
        )
        action_count = paired.shape[0]
        if action_count < 2:
            raise ValueError("v4 requires at least one non-baseline legal action")
        nonbaseline = np.asarray(
            [index for index in range(action_count) if index != sample.baseline_index],
            dtype=np.int32,
        )
        delta = np.asarray(sample.teacher_paired_delta_mean, dtype=np.float64)
        paired_se = np.asarray(
            sample.teacher_delta_se_vs_baseline, dtype=np.float64
        )
        p95 = np.asarray(sample.downside_loss_p95, dtype=np.float64)
        p99 = np.asarray(sample.downside_loss_p99, dtype=np.float64)
        maximum = np.asarray(sample.downside_loss_max, dtype=np.float64)
        expected_shape = (action_count,)
        if any(
            values.shape != expected_shape
            for values in (delta, paired_se, p95, p99, maximum)
        ):
            raise ValueError("v4 target/action shape mismatch")
        if any(
            not np.isfinite(values).all()
            for values in (delta, paired_se, p95, p99, maximum)
        ) or np.any(paired_se < 0.0):
            raise ValueError("v4 targets must be finite and SE non-negative")
        baseline = sample.baseline_index
        if any(
            float(values[baseline]) != 0.0
            for values in (delta, paired_se, p95, p99, maximum)
        ):
            raise ValueError("v4 baseline paired targets must be exactly zero")
        scale = np.maximum(paired_se[nonbaseline], paired_se_floor)
        standardized = np.clip(delta[nonbaseline] / scale, -40.0, 40.0)
        soft_positive = 1.0 / (1.0 + np.exp(-standardized))
        precision = 1.0 / np.square(scale)
        precision /= float(np.sum(precision))
        if not math.isclose(
            float(np.sum(precision)), 1.0, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise AssertionError("v4 state weights do not sum to one")

        feature_blocks.append(paired[nonbaseline])
        delta_blocks.append(delta[nonbaseline])
        positive_blocks.append(soft_positive)
        p95_blocks.append(p95[nonbaseline])
        p99_blocks.append(p99[nonbaseline])
        max_blocks.append(maximum[nonbaseline])
        weight_blocks.append(precision)
        state_blocks.append(np.full(nonbaseline.size, state_index, dtype=np.int32))
        action_blocks.append(nonbaseline)
        baseline_blocks.append(
            np.full(nonbaseline.size, baseline, dtype=np.int32)
        )
        state_rows.append(
            {
                "state_index": state_index,
                "legal_actions": action_count,
                "fit_actions": int(nonbaseline.size),
                "baseline_action_index": baseline,
                "baseline_rows_in_fit": 0,
                "state_weight_sum": float(np.sum(precision)),
            }
        )

    features = np.vstack(feature_blocks).astype(np.float32, copy=False)
    result = V4FoldTrainingArrays(
        features=features,
        delta=np.concatenate(delta_blocks),
        soft_positive=np.concatenate(positive_blocks),
        downside_p95=np.concatenate(p95_blocks),
        downside_p99=np.concatenate(p99_blocks),
        downside_max=np.concatenate(max_blocks),
        weights=np.concatenate(weight_blocks),
        state_indices=np.concatenate(state_blocks),
        action_indices=np.concatenate(action_blocks),
        baseline_indices=np.concatenate(baseline_blocks),
        manifest={
            "schema": "hu_m43_t1_baseline_free_fold_arrays_v4",
            "states": len(samples),
            "rows": int(features.shape[0]),
            "feature_dim": int(features.shape[1]),
            "baseline_rows_included": False,
            "state_weighting": "nonbaseline_inverse_variance_sum_one_per_state",
            "paired_se_floor": float(paired_se_floor),
            "state_rows": state_rows,
        },
    )
    _validate_v4_fold_arrays(result, expected_states=len(samples))
    return result


def fit_v4_fold_estimator(
    samples: Sequence[PreparedTeacherSample],
    *,
    fold_index: int,
    seed: int,
    config: V4FoldWorkerConfig = V4FoldWorkerConfig(),
) -> PairedDeltaRiskFoldEstimator:
    """Fit one fold while returning the existing transport-compatible type."""

    if fold_index < 0:
        raise ValueError("fold_index must be non-negative")
    arrays = build_v4_fold_training_arrays(
        samples, paired_se_floor=config.paired_se_floor
    )
    return PairedDeltaRiskFoldEstimator(
        delta_estimator=_fit_huber(
            arrays.features,
            arrays.delta,
            arrays.weights,
            config=config,
            seed=seed,
        ),
        positive_gain_estimator=_fit_huber(
            arrays.features,
            arrays.soft_positive,
            arrays.weights,
            config=config,
            seed=seed + 1,
        ),
        downside_p95_estimator=_fit_huber(
            arrays.features,
            arrays.downside_p95,
            arrays.weights,
            config=config,
            seed=seed + 2,
        ),
        downside_p99_estimator=_fit_huber(
            arrays.features,
            arrays.downside_p99,
            arrays.weights,
            config=config,
            seed=seed + 3,
        ),
        downside_max_estimator=_fit_huber(
            arrays.features,
            arrays.downside_max,
            arrays.weights,
            config=config,
            seed=seed + 4,
        ),
        paired_feature_dim=4 * HU_FEATURE_DIM,
        fold_index=fold_index,
        feature_schema=HU_M4_PAIRED_ACTION_FEATURE_SCHEMA,
    )


def fit_v4_fold_worker_compatible(
    spec: M43FoldJobSpec,
    samples: Sequence[PreparedTeacherSample],
    *,
    config: V4FoldWorkerConfig = V4FoldWorkerConfig(),
) -> PairedDeltaRiskFoldEstimator:
    """Existing ``M43FoldEstimatorProvider``-compatible v4 worker entrypoint."""

    if len(samples) != spec.fit_samples:
        raise ValueError("v4 worker sample count disagrees with job spec")
    return fit_v4_fold_estimator(
        samples,
        fold_index=spec.estimator_fold_index,
        seed=spec.estimator_seed,
        config=config,
    )


def make_v4_fold_estimator_provider(
    config: V4FoldWorkerConfig = V4FoldWorkerConfig(),
) -> M43FoldEstimatorProvider:
    """Return a provider callable accepted by the existing fold assembler."""

    def provider(
        spec: M43FoldJobSpec,
        samples: Sequence[PreparedTeacherSample],
    ) -> PairedDeltaRiskFoldEstimator:
        return fit_v4_fold_worker_compatible(spec, samples, config=config)

    return provider


@dataclass(frozen=True)
class V4ActionPredictions:
    """Per-action v4 heads plus a legacy-policy-compatible selection score."""

    policy_probability: np.ndarray
    value: np.ndarray
    delta_vs_baseline: np.ndarray
    predicted_absolute_residual: np.ndarray
    action_score: np.ndarray
    proposal_score: np.ndarray
    downside_p95: np.ndarray
    downside_p99: np.ndarray
    downside_max: np.ndarray
    delta_disagreement: np.ndarray


@dataclass(frozen=True)
class V4RuntimeDecision:
    baseline_index: int
    proposal_index: int
    selected_index: int
    safety_probability: float
    override_fired: bool


@dataclass(frozen=True)
class HuM43JointModelV4:
    """Opt-in v4 runtime model; no serving/default registration is performed."""

    paired_fold_estimators: tuple[PairedDeltaRiskFoldEstimator, ...]
    safety_estimator: Any | None = None
    safety_threshold: float = 1.0
    safety_enabled: bool = False
    model_id: str = "hu-m43-t1-v4-unfrozen"
    manifest: Mapping[str, Any] = field(default_factory=dict)
    schema: str = HU_M43_V4_MODEL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != HU_M43_V4_MODEL_SCHEMA:
            raise ValueError(f"unsupported v4 model schema: {self.schema!r}")
        if not self.model_id:
            raise ValueError("v4 model_id must be non-empty")
        if len(self.paired_fold_estimators) < 2:
            raise ValueError("v4 runtime requires at least two fold estimators")
        indices: list[int] = []
        for estimator in self.paired_fold_estimators:
            if not isinstance(estimator, PairedDeltaRiskFoldEstimator):
                raise TypeError("v4 fold estimator has incompatible type")
            estimator.__post_init__()
            indices.append(estimator.fold_index)
        if len(indices) != len(set(indices)):
            raise ValueError("v4 fold estimator indices must be unique")
        threshold = float(self.safety_threshold)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("v4 safety_threshold must be in [0, 1]")
        if self.safety_enabled and self.safety_estimator is None:
            raise ValueError("enabled v4 safety requires an estimator")

    @property
    def action_score_mode(self) -> str:
        """Stable runtime identity consumed by population provenance checks."""

        return HU_M43_V4_ACTION_SCORE_MODE

    def predict_heads_sample(
        self,
        sample: dict[str, Any],
        *,
        baseline_index: int | None = None,
    ) -> V4ActionPredictions:
        matrix, _targets = sample_to_matrix(sample)
        baseline = _resolve_baseline_index(sample, baseline_index, matrix.shape[0])
        paired = build_paired_action_features_matrix(
            matrix, baseline_index=baseline
        )
        outputs = [estimator.predict(paired) for estimator in self.paired_fold_estimators]
        fold_delta = np.vstack([output[0] for output in outputs])
        fold_positive = np.vstack([output[1] for output in outputs])
        fold_p95 = np.vstack([output[2] for output in outputs])
        fold_p99 = np.vstack([output[3] for output in outputs])
        fold_max = np.vstack([output[4] for output in outputs])
        centered_delta = fold_delta - fold_delta[:, [baseline]]
        delta = np.mean(centered_delta, axis=0)
        positive = np.mean(fold_positive, axis=0)
        p95 = np.mean(fold_p95, axis=0)
        p99 = np.mean(fold_p99, axis=0)
        maximum = np.mean(fold_max, axis=0)
        disagreement = np.std(centered_delta, axis=0)
        proposal = delta.copy()
        selection = _selection_scores(proposal, baseline_index=baseline)
        delta[baseline] = 0.0
        proposal[baseline] = 0.0
        positive[baseline] = 0.5
        p95[baseline] = 0.0
        p99[baseline] = 0.0
        maximum[baseline] = 0.0
        disagreement[baseline] = 0.0
        selection[baseline] = 0.0
        arrays = (
            delta,
            proposal,
            positive,
            p95,
            p99,
            maximum,
            disagreement,
            selection,
        )
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("v4 prediction contains non-finite values")
        if proposal[baseline] != 0.0 or selection[baseline] != 0.0:
            raise AssertionError("v4 baseline score is not exactly zero")
        return V4ActionPredictions(
            policy_probability=positive,
            value=delta.copy(),
            delta_vs_baseline=delta,
            predicted_absolute_residual=disagreement,
            action_score=selection,
            proposal_score=proposal,
            downside_p95=p95,
            downside_p99=p99,
            downside_max=maximum,
            delta_disagreement=disagreement,
        )

    def predict_sample_with_baseline(
        self,
        sample: dict[str, Any],
        *,
        baseline_index: int,
    ) -> np.ndarray:
        """Duck-typed score used by the existing opt-in policy wrapper.

        Baseline is exactly zero and the best non-baseline action is exactly
        one.  This score is a selection encoding, not predicted EV; expected
        paired delta remains available as ``proposal_score``/``value``.
        """

        return self.predict_heads_sample(
            sample, baseline_index=baseline_index
        ).action_score

    def predict_sample(self, sample: dict[str, Any]) -> np.ndarray:
        return self.predict_heads_sample(sample).action_score

    def top_nonbaseline_index(
        self,
        sample: Mapping[str, Any],
        *,
        baseline_index: int,
        predictions: V4ActionPredictions | None = None,
    ) -> int:
        heads = predictions or self.predict_heads_sample(
            dict(sample), baseline_index=baseline_index
        )
        return _canonical_argmax_excluding(
            sample,
            heads.proposal_score,
            excluded_index=baseline_index,
        )

    def predict_safety_probability(
        self,
        sample: dict[str, Any],
        *,
        candidate_index: int,
        baseline_index: int,
    ) -> float:
        if self.safety_estimator is None:
            raise RuntimeError("v4 artifact has no safety estimator")
        heads = self.predict_heads_sample(sample, baseline_index=baseline_index)
        features = build_v4_safety_features(
            heads,
            candidate_index=candidate_index,
            baseline_index=baseline_index,
            seat=str(sample.get("seat", "")),
        ).reshape(1, -1)
        return _positive_probability(self.safety_estimator, features)

    def should_override(
        self,
        sample: dict[str, Any],
        *,
        candidate_index: int,
        baseline_index: int,
    ) -> bool:
        if (
            not self.safety_enabled
            or self.safety_estimator is None
            or candidate_index == baseline_index
        ):
            return False
        return self.predict_safety_probability(
            sample,
            candidate_index=candidate_index,
            baseline_index=baseline_index,
        ) >= float(self.safety_threshold)

    def select_action_index(
        self,
        sample: dict[str, Any],
        *,
        baseline_index: int | None = None,
    ) -> V4RuntimeDecision:
        action_count = len(sample.get("actions", ()))
        baseline = _resolve_baseline_index(sample, baseline_index, action_count)
        heads = self.predict_heads_sample(sample, baseline_index=baseline)
        proposal = self.top_nonbaseline_index(
            sample,
            baseline_index=baseline,
            predictions=heads,
        )
        probability = (
            self.predict_safety_probability(
                sample,
                candidate_index=proposal,
                baseline_index=baseline,
            )
            if self.safety_estimator is not None
            else 0.0
        )
        fired = bool(
            self.safety_enabled
            and probability >= float(self.safety_threshold)
        )
        return V4RuntimeDecision(
            baseline_index=baseline,
            proposal_index=proposal,
            selected_index=proposal if fired else baseline,
            safety_probability=float(probability),
            override_fired=fired,
        )

    def with_frozen_safety(
        self,
        estimator: Any,
        *,
        threshold: float,
        enabled: bool,
        manifest: Mapping[str, Any] | None = None,
    ) -> "HuM43JointModelV4":
        return replace(
            self,
            safety_estimator=estimator,
            safety_threshold=float(threshold),
            safety_enabled=bool(enabled),
            manifest=self.manifest if manifest is None else dict(manifest),
        )

    def save(self, path: str | Path) -> str:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "artifact_schema": HU_M43_V4_ARTIFACT_SCHEMA,
            "model_schema": HU_M43_V4_MODEL_SCHEMA,
            "proposal_schema": HU_M43_V4_PROPOSAL_SCHEMA,
            "model": self,
        }
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        return _file_sha256(destination)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
    ) -> "HuM43JointModelV4":
        source = Path(path)
        encoded = source.read_bytes()
        actual = hashlib.sha256(encoded).hexdigest()
        if expected_sha256 is not None and actual != _sha256_text(expected_sha256):
            raise ValueError("v4 artifact SHA-256 mismatch")
        payload = pickle.loads(encoded)
        if not isinstance(payload, dict):
            raise TypeError("v4 artifact must be a mapping")
        if payload.get("artifact_schema") != HU_M43_V4_ARTIFACT_SCHEMA:
            raise ValueError("unsupported v4 artifact schema")
        if payload.get("model_schema") != HU_M43_V4_MODEL_SCHEMA:
            raise ValueError("unsupported v4 model schema")
        if payload.get("proposal_schema") != HU_M43_V4_PROPOSAL_SCHEMA:
            raise ValueError("unsupported v4 proposal schema")
        model = payload.get("model")
        if not isinstance(model, cls):
            raise TypeError("v4 artifact contains wrong model type")
        model.__post_init__()
        return model


def load_hu_m43_joint_model_v4(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> HuM43JointModelV4:
    return HuM43JointModelV4.load(path, expected_sha256=expected_sha256)


def build_v4_safety_features(
    predictions: V4ActionPredictions,
    *,
    candidate_index: int,
    baseline_index: int,
    seat: str,
) -> np.ndarray:
    """Build action-order-invariant runtime features without teacher fields."""

    if seat not in {"first", "second"}:
        raise ValueError(f"invalid seat: {seat!r}")
    count = predictions.proposal_score.size
    if not 0 <= candidate_index < count or not 0 <= baseline_index < count:
        raise IndexError("v4 safety candidate/baseline index is invalid")
    if candidate_index == baseline_index:
        raise ValueError("v4 safety features require a non-baseline candidate")
    nonbaseline = np.asarray(
        [index for index in range(count) if index != baseline_index], dtype=np.int32
    )
    proposal = predictions.proposal_score[nonbaseline]
    candidate_score = float(predictions.proposal_score[candidate_index])
    other = np.asarray(
        [
            predictions.proposal_score[index]
            for index in nonbaseline
            if index != candidate_index
        ],
        dtype=np.float64,
    )
    next_margin = (
        candidate_score - float(np.max(other)) if other.size else 0.0
    )
    rank_fraction = (
        float(np.mean(proposal < candidate_score)) if proposal.size > 1 else 1.0
    )
    features = np.asarray(
        [
            candidate_score,
            float(predictions.policy_probability[candidate_index]),
            float(predictions.downside_p95[candidate_index]),
            float(predictions.downside_p99[candidate_index]),
            float(predictions.downside_max[candidate_index]),
            float(predictions.delta_disagreement[candidate_index]),
            next_margin,
            rank_fraction,
            float(np.mean(proposal)),
            float(np.std(proposal)),
            float(np.max(proposal)),
            float(np.min(proposal)),
            float(nonbaseline.size),
            float(seat == "first"),
            float(seat == "second"),
        ],
        dtype=np.float32,
    )
    if features.shape != (HU_M43_V4_SAFETY_FEATURE_DIM,):
        raise AssertionError("v4 safety feature dimension changed")
    if not np.isfinite(features).all():
        raise ValueError("v4 safety features are non-finite")
    return features


@dataclass(frozen=True)
class V4OofStatePrediction:
    sample_index: int
    fold_index: int
    predictions: V4ActionPredictions
    identity_excluded_from_fit: bool


@dataclass(frozen=True)
class V4OofSafetyDataset:
    features: np.ndarray
    labels: np.ndarray
    weights: np.ndarray
    rows: tuple[Mapping[str, Any], ...]
    manifest: Mapping[str, Any]
    seed_values: frozenset[str]
    observation_fingerprints: frozenset[str]


def build_v4_oof_safety_dataset(
    samples: Sequence[PreparedTeacherSample],
    oof_predictions: Sequence[V4OofStatePrediction],
    *,
    maximum_p95_loss: float = 25.0,
    maximum_p99_loss: float = 40.0,
    maximum_max_loss: float = 50.0,
) -> V4OofSafetyDataset:
    """Create all-action, state-balanced offline safety examples."""

    ordered = _ordered_oof_predictions(samples, oof_predictions)
    features: list[np.ndarray] = []
    labels: list[int] = []
    weights: list[float] = []
    rows: list[dict[str, Any]] = []
    state_weight_sums: list[float] = []
    for state_index, (sample, oof) in enumerate(zip(samples, ordered, strict=True)):
        _require_v4_targets(sample)
        heads = oof.predictions
        action_count = len(sample.policy_sample.get("actions", ()))
        _validate_prediction_shape(heads, action_count)
        baseline = sample.baseline_index
        nonbaseline = [index for index in range(action_count) if index != baseline]
        row_weight = 1.0 / len(nonbaseline)
        state_sum = 0.0
        for action_index in nonbaseline:
            teacher_delta = float(sample.teacher_paired_delta_mean[action_index])
            p95 = float(sample.downside_loss_p95[action_index])
            p99 = float(sample.downside_loss_p99[action_index])
            maximum = float(sample.downside_loss_max[action_index])
            safe = bool(
                teacher_delta > 0.0
                and p95 <= maximum_p95_loss
                and p99 <= maximum_p99_loss
                and maximum <= maximum_max_loss
            )
            features.append(
                build_v4_safety_features(
                    heads,
                    candidate_index=action_index,
                    baseline_index=baseline,
                    seat=sample.seat,
                )
            )
            labels.append(int(safe))
            weights.append(row_weight)
            state_sum += row_weight
            rows.append(
                {
                    "state_index": state_index,
                    "fold_index": oof.fold_index,
                    "action_index": action_index,
                    "baseline_index": baseline,
                    "teacher_delta": teacher_delta,
                    "downside_loss_p95": p95,
                    "downside_loss_p99": p99,
                    "downside_loss_max": maximum,
                    "safe_label": int(safe),
                    "sample_weight": row_weight,
                    "identity_excluded_from_fit": bool(
                        oof.identity_excluded_from_fit
                    ),
                    "teacher_value_status": HU_M43_V4_TEACHER_STATUS,
                }
            )
        state_weight_sums.append(state_sum)
    matrix = np.vstack(features).astype(np.float32, copy=False)
    label_array = np.asarray(labels, dtype=np.int8)
    weight_array = np.asarray(weights, dtype=np.float64)
    if not all(
        math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1.0e-12)
        for value in state_weight_sums
    ):
        raise AssertionError("v4 safety rows are not state-balanced")
    return V4OofSafetyDataset(
        features=matrix,
        labels=label_array,
        weights=weight_array,
        rows=tuple(rows),
        manifest={
            "schema": HU_M43_V4_SAFETY_DATASET_SCHEMA,
            "states": len(samples),
            "rows": len(rows),
            "feature_dim": HU_M43_V4_SAFETY_FEATURE_DIM,
            "safe_rows": int(np.sum(label_array == 1)),
            "unsafe_rows": int(np.sum(label_array == 0)),
            "baseline_rows": 0,
            "row_source": "all_nonbaseline_oof_actions",
            "state_weighting": "sum_one_per_information_set",
            "safe_label": {
                "minimum_teacher_delta": "strictly_greater_than_zero",
                "maximum_p95_loss": float(maximum_p95_loss),
                "maximum_p99_loss": float(maximum_p99_loss),
                "maximum_max_loss": float(maximum_max_loss),
            },
            "teacher_value_status": HU_M43_V4_TEACHER_STATUS,
            "runtime_teacher_inputs": False,
        },
        seed_values=frozenset().union(
            *(sample.root_seed_values for sample in samples)
        ),
        observation_fingerprints=frozenset(
            sample.observation_fingerprint for sample in samples
        ),
    )


def build_v4_independent_safety_dataset(
    model: HuM43JointModelV4,
    samples: Sequence[PreparedTeacherSample],
    *,
    source: str,
    maximum_p95_loss: float = 25.0,
    maximum_p99_loss: float = 40.0,
    maximum_max_loss: float = 50.0,
) -> V4OofSafetyDataset:
    """Build state-balanced rows for a split independent of model training.

    This is used only for ``calibration.safety_fit``.  The source string is
    explicit so a threshold-lock or locked-holdout caller cannot silently be
    mislabeled as a fit source.
    """

    if source != "calibration.safety_fit":
        raise ValueError("v4 independent fit rows require calibration.safety_fit")
    predictions = tuple(
        V4OofStatePrediction(
            sample_index=index,
            fold_index=-1,
            predictions=model.predict_heads_sample(
                sample.policy_sample, baseline_index=sample.baseline_index
            ),
            identity_excluded_from_fit=True,
        )
        for index, sample in enumerate(samples)
    )
    dataset = build_v4_oof_safety_dataset(
        samples,
        predictions,
        maximum_p95_loss=maximum_p95_loss,
        maximum_p99_loss=maximum_p99_loss,
        maximum_max_loss=maximum_max_loss,
    )
    rows = tuple({**dict(row), "source": source} for row in dataset.rows)
    manifest = {
        **dict(dataset.manifest),
        "row_source": "all_nonbaseline_calibration_safety_fit_actions",
        "source": source,
        "model_training_identity_excluded": True,
        "used_for_threshold_selection": False,
    }
    return replace(dataset, rows=rows, manifest=manifest)


@dataclass(frozen=True)
class V4SafetyFitResult:
    """Safety estimator fit without threshold-lock label access."""

    model: HuM43JointModelV4
    report: Mapping[str, Any]
    fit_seed_values: frozenset[str]
    fit_observation_fingerprints: frozenset[str]


def fit_v4_safety_calibrator(
    model: HuM43JointModelV4,
    train_oof_dataset: V4OofSafetyDataset,
    safety_fit_samples: Sequence[PreparedTeacherSample],
    *,
    safety_calibrator_c: float = 0.25,
    seed: int = 2026071805,
    maximum_p95_loss: float = 25.0,
    maximum_p99_loss: float = 40.0,
    maximum_max_loss: float = 50.0,
) -> V4SafetyFitResult:
    """Fit low-capacity safety probability from train OOF + safety-fit only."""

    if not safety_fit_samples:
        raise ValueError("v4 safety calibrator requires safety-fit samples")
    if not math.isfinite(safety_calibrator_c) or safety_calibrator_c <= 0.0:
        raise ValueError("safety_calibrator_c must be finite and positive")
    if train_oof_dataset.features.shape[1] != HU_M43_V4_SAFETY_FEATURE_DIM:
        raise ValueError("v4 train OOF safety feature dimension mismatch")
    safety_fit = build_v4_independent_safety_dataset(
        model,
        safety_fit_samples,
        source="calibration.safety_fit",
        maximum_p95_loss=maximum_p95_loss,
        maximum_p99_loss=maximum_p99_loss,
        maximum_max_loss=maximum_max_loss,
    )
    seed_overlap = train_oof_dataset.seed_values & safety_fit.seed_values
    fingerprint_overlap = (
        train_oof_dataset.observation_fingerprints
        & safety_fit.observation_fingerprints
    )
    if seed_overlap or fingerprint_overlap:
        raise ValueError("v4 train OOF/safety-fit identity overlap")
    features = np.vstack(
        (train_oof_dataset.features, safety_fit.features)
    ).astype(np.float32, copy=False)
    labels = np.concatenate(
        (train_oof_dataset.labels, safety_fit.labels)
    ).astype(np.int8, copy=False)
    weights = np.concatenate(
        (train_oof_dataset.weights, safety_fit.weights)
    ).astype(np.float64, copy=False)
    unique = np.unique(labels)
    if unique.size == 1:
        estimator: Any = ConstantProbabilityEstimator(float(unique[0]))
        family = "constant_probability_one_class"
    else:
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=safety_calibrator_c,
                solver="lbfgs",
                max_iter=500,
                random_state=seed,
            ),
        ).fit(features, labels, logisticregression__sample_weight=weights)
        family = "standardized_l2_logistic_low_capacity"
    # Threshold is intentionally not selected here.  The provisional model is
    # fail-closed until the one fixed threshold-lock sweep returns Go.
    provisional = model.with_frozen_safety(
        estimator,
        threshold=1.0,
        enabled=False,
        manifest=model.manifest,
    )
    fit_seeds = train_oof_dataset.seed_values | safety_fit.seed_values
    fit_fingerprints = (
        train_oof_dataset.observation_fingerprints
        | safety_fit.observation_fingerprints
    )
    report = {
        "schema": HU_M43_V4_SAFETY_FIT_SCHEMA,
        "status": "fit_threshold_unselected",
        "estimator_family": family,
        "safety_calibrator_c": float(safety_calibrator_c),
        "seed": int(seed),
        "feature_schema": HU_M43_V4_SAFETY_FEATURE_SCHEMA,
        "feature_dim": HU_M43_V4_SAFETY_FEATURE_DIM,
        "label": {
            "teacher_delta": "strictly_greater_than_zero",
            "maximum_p95_loss": float(maximum_p95_loss),
            "maximum_p99_loss": float(maximum_p99_loss),
            "maximum_max_loss": float(maximum_max_loss),
        },
        "sources": {
            "train_oof": {
                "states": int(train_oof_dataset.manifest["states"]),
                "rows": int(train_oof_dataset.features.shape[0]),
                "safe_rows": int(np.sum(train_oof_dataset.labels == 1)),
                "unsafe_rows": int(np.sum(train_oof_dataset.labels == 0)),
            },
            "calibration.safety_fit": {
                "states": len(safety_fit_samples),
                "rows": int(safety_fit.features.shape[0]),
                "safe_rows": int(np.sum(safety_fit.labels == 1)),
                "unsafe_rows": int(np.sum(safety_fit.labels == 0)),
            },
            "calibration.threshold_lock": {
                "used": False,
                "labels_opened": False,
            },
            "locked_holdout": {"used": False, "labels_opened": False},
        },
        "combined_rows": int(features.shape[0]),
        "combined_safe_rows": int(np.sum(labels == 1)),
        "combined_unsafe_rows": int(np.sum(labels == 0)),
        "state_balanced_weights": True,
        "fit_identity_overlap": {
            "seed_values": 0,
            "observation_fingerprints": 0,
        },
        "safety_enabled": False,
        "threshold_selected": False,
        "runtime_teacher_inputs": False,
        "teacher_value_status": HU_M43_V4_TEACHER_STATUS,
    }
    return V4SafetyFitResult(
        model=provisional,
        report=report,
        fit_seed_values=frozenset(fit_seeds),
        fit_observation_fingerprints=frozenset(fit_fingerprints),
    )


@dataclass(frozen=True)
class V4ThresholdSelectionResult:
    model: HuM43JointModelV4
    report: Mapping[str, Any]


def select_v4_threshold(
    safety_fit: V4SafetyFitResult,
    threshold_lock_samples: Sequence[PreparedTeacherSample],
    *,
    thresholds: Sequence[float],
    minimum_fires: int = 10,
    maximum_false_positive_rate: float = 0.30,
    maximum_p95_loss: float = 25.0,
    maximum_p99_loss: float = 40.0,
    maximum_max_loss: float = 50.0,
) -> V4ThresholdSelectionResult:
    """Run the one fixed threshold-lock sweep and fail closed on No-Go."""

    if not threshold_lock_samples:
        raise ValueError("v4 threshold lock requires samples")
    if safety_fit.model.safety_estimator is None:
        raise ValueError("v4 threshold lock requires a fitted safety estimator")
    normalized = _normalize_thresholds(thresholds)
    if minimum_fires < 1:
        raise ValueError("minimum_fires must be positive")
    lock_seeds = frozenset().union(
        *(sample.root_seed_values for sample in threshold_lock_samples)
    )
    lock_fingerprints = frozenset(
        sample.observation_fingerprint for sample in threshold_lock_samples
    )
    seed_overlap = safety_fit.fit_seed_values & lock_seeds
    fingerprint_overlap = (
        safety_fit.fit_observation_fingerprints & lock_fingerprints
    )
    if seed_overlap or fingerprint_overlap:
        raise ValueError("v4 safety-fit/threshold-lock identity overlap")
    proposal_rows: list[dict[str, Any]] = []
    for state_index, sample in enumerate(threshold_lock_samples):
        _require_v4_targets(sample)
        baseline = sample.baseline_index
        heads = safety_fit.model.predict_heads_sample(
            sample.policy_sample, baseline_index=baseline
        )
        proposal = safety_fit.model.top_nonbaseline_index(
            sample.policy_sample,
            baseline_index=baseline,
            predictions=heads,
        )
        probability = safety_fit.model.predict_safety_probability(
            sample.policy_sample,
            candidate_index=proposal,
            baseline_index=baseline,
        )
        proposal_rows.append(
            {
                "state_index": state_index,
                "proposal_index": proposal,
                "baseline_index": baseline,
                "probability": probability,
                "teacher_delta": float(
                    sample.teacher_paired_delta_mean[proposal]
                ),
                "downside_loss_p95": float(sample.downside_loss_p95[proposal]),
                "downside_loss_p99": float(sample.downside_loss_p99[proposal]),
                "downside_loss_max": float(sample.downside_loss_max[proposal]),
                "teacher_value_status": HU_M43_V4_TEACHER_STATUS,
            }
        )
    sweep: list[dict[str, Any]] = []
    for threshold in normalized:
        selected = [
            row for row in proposal_rows if float(row["probability"]) >= threshold
        ]
        sweep.append(
            _v4_threshold_metrics(
                selected,
                total_states=len(threshold_lock_samples),
                threshold=threshold,
            )
        )
    eligible = [
        row
        for row in sweep
        if row["fires"] >= minimum_fires
        and row["teacher_mean_delta_per_fire"] > 0.0
        and row["false_positive_rate"] <= maximum_false_positive_rate
        and row["p95_loss"] <= maximum_p95_loss
        and row["p99_loss"] <= maximum_p99_loss
        and row["max_loss"] <= maximum_max_loss
    ]
    if eligible:
        selected = max(
            eligible,
            key=lambda row: (row["teacher_delta_per_state"], row["threshold"]),
        )
        status = "go"
        enabled = True
    else:
        selected = next(row for row in sweep if row["threshold"] == max(normalized))
        status = "no_go"
        enabled = False
    final_model = safety_fit.model.with_frozen_safety(
        safety_fit.model.safety_estimator,
        threshold=float(selected["threshold"]),
        enabled=enabled,
        manifest=safety_fit.model.manifest,
    )
    report = {
        "schema": HU_M43_V4_THRESHOLD_SCHEMA,
        "status": status,
        "source": "fresh_calibration.threshold_lock_only",
        "states": len(threshold_lock_samples),
        "proposal_rows": len(proposal_rows),
        "thresholds": normalized,
        "selected_threshold": float(selected["threshold"]),
        "selected_metrics": selected,
        "threshold_sweep": sweep,
        "constraints": {
            "minimum_fires": int(minimum_fires),
            "maximum_false_positive_rate": float(maximum_false_positive_rate),
            "maximum_p95_loss": float(maximum_p95_loss),
            "maximum_p99_loss": float(maximum_p99_loss),
            "maximum_max_loss": float(maximum_max_loss),
        },
        "identity_overlap_with_fit": {
            "seed_values": 0,
            "observation_fingerprints": 0,
        },
        "threshold_adaptation_after_selection": False,
        "safety_enabled": enabled,
        "locked_holdout": {"used": False, "labels_opened": False},
        "runtime_teacher_inputs": False,
        "teacher_value_status": HU_M43_V4_TEACHER_STATUS,
    }
    return V4ThresholdSelectionResult(model=final_model, report=report)


def build_v4_training_manifest(
    crossfit: V4CrossFitResult,
    safety_fit: V4SafetyFitResult,
    threshold_selection: V4ThresholdSelectionResult,
    *,
    model_sha256: str | None = None,
) -> dict[str, Any]:
    """Compose the immutable pre-locked-evaluation v4 training manifest."""

    if model_sha256 is not None:
        model_sha256 = _sha256_text(model_sha256)
    manifest = {
        "schema": HU_M43_V4_TRAINING_MANIFEST_SCHEMA,
        "model_schema": HU_M43_V4_MODEL_SCHEMA,
        "model_id": threshold_selection.model.model_id,
        "proposal_schema": HU_M43_V4_PROPOSAL_SCHEMA,
        "model_sha256": model_sha256,
        "promotion_status": (
            "candidate_ready_for_freeze"
            if threshold_selection.report["status"] == "go"
            else "no_go_calibration"
        ),
        "crossfit": dict(crossfit.report),
        "safety_fit": dict(safety_fit.report),
        "threshold_selection": dict(threshold_selection.report),
        "baseline_training_rows_included": False,
        "baseline_runtime_score_exact_zero": True,
        "all_action_state_balanced_oof_safety": True,
        "runtime_teacher_inputs": False,
        "teacher_value_status": HU_M43_V4_TEACHER_STATUS,
        "locked_holdout": {
            "status": "not_evaluated_pre_freeze",
            "labels_opened": False,
        },
        "runtime": {
            "current_profile_mutated": False,
            "policy_activated": False,
            "full_replacement": False,
        },
    }
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    return manifest


def write_v4_training_manifest(
    path: str | Path, manifest: Mapping[str, Any]
) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return _file_sha256(destination)


@dataclass(frozen=True)
class V4PrecalibrationGateConfig:
    """Frozen attempt-02 train-OOF gate; never derived from calibration."""

    expected_states: int = 200
    expected_folds: int = 5
    minimum_states_per_fold: int = 30
    minimum_proposal_positive_rate: float = 0.40
    minimum_tail_safe_proposals: int = 30
    minimum_tail_unsafe_proposals: int = 30
    minimum_all_action_safe_rows: int = 100
    minimum_all_action_unsafe_rows: int = 100
    require_positive_mean_teacher_delta: bool = True

    def __post_init__(self) -> None:
        if self.expected_states < 1 or self.expected_folds < 2:
            raise ValueError("v4 precal expected size/folds are invalid")
        if self.minimum_states_per_fold < 1:
            raise ValueError("minimum_states_per_fold must be positive")
        if not 0.0 <= self.minimum_proposal_positive_rate <= 1.0:
            raise ValueError("minimum_proposal_positive_rate must be in [0, 1]")
        for name in (
            "minimum_tail_safe_proposals",
            "minimum_tail_unsafe_proposals",
            "minimum_all_action_safe_rows",
            "minimum_all_action_unsafe_rows",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    def to_manifest(self) -> dict[str, Any]:
        return {
            "expected_states": self.expected_states,
            "expected_folds": self.expected_folds,
            "minimum_states_per_fold": self.minimum_states_per_fold,
            "minimum_proposal_positive_rate": self.minimum_proposal_positive_rate,
            "minimum_tail_safe_proposals": self.minimum_tail_safe_proposals,
            "minimum_tail_unsafe_proposals": self.minimum_tail_unsafe_proposals,
            "minimum_all_action_safe_rows": self.minimum_all_action_safe_rows,
            "minimum_all_action_unsafe_rows": self.minimum_all_action_unsafe_rows,
            "require_positive_mean_teacher_delta": (
                self.require_positive_mean_teacher_delta
            ),
        }


def evaluate_v4_precalibration_oof_gate(
    samples: Sequence[PreparedTeacherSample],
    oof_predictions: Sequence[V4OofStatePrediction],
    safety_dataset: V4OofSafetyDataset,
    *,
    config: V4PrecalibrationGateConfig = V4PrecalibrationGateConfig(),
) -> dict[str, Any]:
    """Audit proposal viability before any calibration labels are opened."""

    ordered = _ordered_oof_predictions(samples, oof_predictions)
    fold_counts = {
        fold: sum(row.fold_index == fold for row in ordered)
        for fold in sorted({row.fold_index for row in ordered})
    }
    proposal_rows: list[dict[str, Any]] = []
    baseline_exact = True
    all_identity_excluded = True
    for state_index, (sample, oof) in enumerate(zip(samples, ordered, strict=True)):
        heads = oof.predictions
        baseline = sample.baseline_index
        baseline_exact = baseline_exact and (
            heads.proposal_score[baseline] == 0.0
            and heads.action_score[baseline] == 0.0
        )
        all_identity_excluded = (
            all_identity_excluded and oof.identity_excluded_from_fit
        )
        proposal = _canonical_argmax_excluding(
            sample.policy_sample,
            heads.proposal_score,
            excluded_index=baseline,
        )
        delta = float(sample.teacher_paired_delta_mean[proposal])
        p95 = float(sample.downside_loss_p95[proposal])
        p99 = float(sample.downside_loss_p99[proposal])
        maximum = float(sample.downside_loss_max[proposal])
        tail_safe = bool(
            delta > 0.0 and p95 <= 25.0 and p99 <= 40.0 and maximum <= 50.0
        )
        proposal_rows.append(
            {
                "state_index": state_index,
                "fold_index": oof.fold_index,
                "proposal_index": proposal,
                "baseline_index": baseline,
                "teacher_delta": delta,
                "teacher_positive": delta > 0.0,
                "tail_safe": tail_safe,
                "teacher_value_status": HU_M43_V4_TEACHER_STATUS,
            }
        )
    deltas = np.asarray(
        [row["teacher_delta"] for row in proposal_rows], dtype=np.float64
    )
    positive_count = int(np.sum(deltas > 0.0))
    tail_safe_count = sum(bool(row["tail_safe"]) for row in proposal_rows)
    fold_set = set(fold_counts)
    gates = {
        "exact_state_count": len(samples) == config.expected_states,
        "oof_state_coverage_exactly_once": (
            len(ordered) == len(samples)
            and [row.sample_index for row in ordered] == list(range(len(samples)))
        ),
        "expected_fold_set": fold_set == set(range(config.expected_folds)),
        "minimum_states_per_fold": bool(fold_counts)
        and min(fold_counts.values()) >= config.minimum_states_per_fold,
        "all_oof_identities_excluded_from_fit": all_identity_excluded,
        "baseline_score_exact_zero": baseline_exact,
        "nonbaseline_proposal_every_state": len(proposal_rows) == len(samples),
        "minimum_proposal_positive_rate": (
            positive_count / len(samples) >= config.minimum_proposal_positive_rate
            if samples
            else False
        ),
        "positive_mean_teacher_delta": (
            float(np.mean(deltas)) > 0.0
            if config.require_positive_mean_teacher_delta and deltas.size
            else not config.require_positive_mean_teacher_delta
        ),
        "minimum_tail_safe_proposals": (
            tail_safe_count >= config.minimum_tail_safe_proposals
        ),
        "minimum_tail_unsafe_proposals": (
            len(samples) - tail_safe_count >= config.minimum_tail_unsafe_proposals
        ),
        "minimum_all_action_safe_rows": (
            int(np.sum(safety_dataset.labels == 1))
            >= config.minimum_all_action_safe_rows
        ),
        "minimum_all_action_unsafe_rows": (
            int(np.sum(safety_dataset.labels == 0))
            >= config.minimum_all_action_unsafe_rows
        ),
    }
    # NumPy scalar comparisons (notably the exact baseline check) can produce
    # ``np.bool_``.  Training manifests are JSON artifacts, so normalize every
    # gate here instead of relying on a permissive JSON encoder downstream.
    gates = {name: bool(value) for name, value in gates.items()}
    return {
        "schema": HU_M43_V4_PRECAL_GATE_SCHEMA,
        "status": "go" if all(gates.values()) else "no_go",
        "config": config.to_manifest(),
        "config_sha256": _canonical_sha256(config.to_manifest()),
        "metrics": {
            "states": len(samples),
            "nonbaseline_proposals": len(proposal_rows),
            "proposal_positive_count": positive_count,
            "proposal_positive_rate": (
                float(positive_count / len(samples)) if samples else 0.0
            ),
            "proposal_mean_teacher_delta": (
                float(np.mean(deltas)) if deltas.size else 0.0
            ),
            "tail_safe_proposals": tail_safe_count,
            "tail_unsafe_proposals": len(samples) - tail_safe_count,
            "all_action_safe_rows": int(np.sum(safety_dataset.labels == 1)),
            "all_action_unsafe_rows": int(np.sum(safety_dataset.labels == 0)),
            "fold_counts": {str(key): value for key, value in fold_counts.items()},
        },
        "gates": gates,
        "proposal_rows": proposal_rows,
        "teacher_value_status": HU_M43_V4_TEACHER_STATUS,
        "runtime_teacher_inputs": False,
        "calibration_opened": False,
    }


@dataclass(frozen=True)
class V4CrossFitResult:
    model: HuM43JointModelV4
    oof_predictions: tuple[V4OofStatePrediction, ...]
    safety_dataset: V4OofSafetyDataset
    precalibration_report: Mapping[str, Any]
    report: Mapping[str, Any]


def fit_v4_nested_crossfit(
    samples: Sequence[PreparedTeacherSample],
    *,
    cross_fit_folds: int = 5,
    seed: int = 2026071801,
    model_id: str = "hu-m43-t1-v4-unfrozen",
    worker_config: V4FoldWorkerConfig = V4FoldWorkerConfig(),
    gate_config: V4PrecalibrationGateConfig = V4PrecalibrationGateConfig(),
    fold_estimator_provider: M43FoldEstimatorProvider | None = None,
) -> V4CrossFitResult:
    """Fit/assemble the exact outer+inner grid and create nested OOF rows."""

    plan = build_m43_fold_training_plan(
        samples, cross_fit_folds=cross_fit_folds, seed=seed
    )
    jobs = {job.spec.job_index: job for job in plan.jobs}

    def obtain(job_index: int) -> PairedDeltaRiskFoldEstimator:
        job = jobs[job_index]
        estimator = (
            fit_v4_fold_worker_compatible(
                job.spec, job.fit_samples, config=worker_config
            )
            if fold_estimator_provider is None
            else fold_estimator_provider(job.spec, job.fit_samples)
        )
        if not isinstance(estimator, PairedDeltaRiskFoldEstimator):
            raise TypeError("v4 fold provider returned incompatible estimator")
        if estimator.fold_index != job.spec.estimator_fold_index:
            raise ValueError("v4 fold estimator index disagrees with job spec")
        return estimator

    runtime_folds: list[PairedDeltaRiskFoldEstimator] = []
    oof: list[V4OofStatePrediction | None] = [None] * len(plan.ordered_samples)
    fold_audits: list[dict[str, Any]] = []
    for outer_fold in range(cross_fit_folds):
        base_job = outer_fold * (cross_fit_folds + 1)
        runtime_folds.append(obtain(base_job))
        inner = tuple(obtain(base_job + 1 + index) for index in range(cross_fit_folds))
        fold_model = HuM43JointModelV4(
            paired_fold_estimators=inner,
            model_id=f"{model_id}:nested-oof-{outer_fold}",
        )
        validation = [
            index
            for index, assigned in enumerate(plan.outer_fold_ids)
            if assigned == outer_fold
        ]
        training = [
            index
            for index, assigned in enumerate(plan.outer_fold_ids)
            if assigned != outer_fold
        ]
        train_seeds = set().union(
            *(plan.ordered_samples[index].root_seed_values for index in training)
        )
        validation_seeds = set().union(
            *(plan.ordered_samples[index].root_seed_values for index in validation)
        )
        train_fingerprints = {
            plan.ordered_samples[index].observation_fingerprint for index in training
        }
        validation_fingerprints = {
            plan.ordered_samples[index].observation_fingerprint for index in validation
        }
        identity_excluded = not (
            train_seeds & validation_seeds
            or train_fingerprints & validation_fingerprints
        )
        if not identity_excluded:
            raise AssertionError("v4 outer fold identity leakage")
        for index in validation:
            sample = plan.ordered_samples[index]
            oof[index] = V4OofStatePrediction(
                sample_index=index,
                fold_index=outer_fold,
                predictions=fold_model.predict_heads_sample(
                    sample.policy_sample, baseline_index=sample.baseline_index
                ),
                identity_excluded_from_fit=True,
            )
        fold_audits.append(
            {
                "fold": outer_fold,
                "training_states": len(training),
                "validation_states": len(validation),
                "inner_fold_estimators": len(inner),
                "outer_validation_identity_excluded": identity_excluded,
            }
        )
    if any(row is None for row in oof):
        raise AssertionError("v4 OOF coverage is incomplete")
    clean_oof = tuple(row for row in oof if row is not None)
    runtime_model = HuM43JointModelV4(
        paired_fold_estimators=tuple(runtime_folds), model_id=model_id
    )
    safety = build_v4_oof_safety_dataset(plan.ordered_samples, clean_oof)
    precal = evaluate_v4_precalibration_oof_gate(
        plan.ordered_samples,
        clean_oof,
        safety,
        config=gate_config,
    )
    report = {
        "schema": HU_M43_V4_CROSSFIT_SCHEMA,
        "status": "pass" if precal["status"] == "go" else "no_go_precalibration",
        "states": len(plan.ordered_samples),
        "folds": cross_fit_folds,
        "fold_jobs": len(plan.jobs),
        "exact_outer_inner_job_grid": (
            len(plan.jobs) == cross_fit_folds * (cross_fit_folds + 1)
        ),
        "proposal_schema": HU_M43_V4_PROPOSAL_SCHEMA,
        "baseline_training_rows_included": False,
        "runtime_teacher_inputs": False,
        "calibration_opened": False,
        "worker_config": worker_config.to_manifest(),
        "precalibration_gate": precal,
        "fold_audits": fold_audits,
    }
    return V4CrossFitResult(
        model=runtime_model,
        oof_predictions=clean_oof,
        safety_dataset=safety,
        precalibration_report=precal,
        report=report,
    )


def _fit_huber(
    features: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    *,
    config: V4FoldWorkerConfig,
    seed: int,
) -> Any:
    if np.allclose(targets, targets[0]):
        return DummyRegressor(strategy="constant", constant=float(targets[0])).fit(
            features, targets, sample_weight=weights
        )
    estimator = GradientBoostingRegressor(
        loss="huber",
        alpha=config.huber_alpha,
        n_estimators=config.iterations,
        max_leaf_nodes=config.max_leaf_nodes,
        learning_rate=config.learning_rate,
        random_state=seed,
    )
    return estimator.fit(features, targets, sample_weight=weights)


def _selection_scores(proposal: np.ndarray, *, baseline_index: int) -> np.ndarray:
    """Encode top-nonbaseline selection for the existing baseline-inclusive argmax."""

    values = np.asarray(proposal, dtype=np.float64).reshape(-1)
    if values.size < 2 or not 0 <= baseline_index < values.size:
        raise ValueError("v4 selection score baseline/action count is invalid")
    nonbaseline = np.asarray(
        [index for index in range(values.size) if index != baseline_index],
        dtype=np.int32,
    )
    maximum = float(np.max(values[nonbaseline]))
    scores = values - maximum + HU_M43_V4_SELECTION_SCORE
    scores[baseline_index] = 0.0
    if float(np.max(scores[nonbaseline])) != HU_M43_V4_SELECTION_SCORE:
        raise AssertionError("v4 top nonbaseline selection score changed")
    return scores


def _canonical_argmax_excluding(
    sample: Mapping[str, Any],
    values: Sequence[float] | np.ndarray,
    *,
    excluded_index: int,
) -> int:
    actions = sample.get("actions", ())
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if (
        isinstance(actions, (str, bytes))
        or not actions
        or len(actions) != array.size
        or not 0 <= excluded_index < array.size
    ):
        raise ValueError("v4 action/value/exclusion mapping is invalid")
    candidates = [index for index in range(array.size) if index != excluded_index]
    if not candidates or not np.isfinite(array).all():
        raise ValueError("v4 proposal values are invalid")
    best = max(float(array[index]) for index in candidates)
    tied = [index for index in candidates if float(array[index]) == best]
    return int(
        min(
            tied,
            key=lambda index: action_key_from_payload(actions[index]).sort_key(),
        )
    )


def _resolve_baseline_index(
    sample: Mapping[str, Any], explicit: int | None, action_count: int
) -> int:
    declared = sample.get("baseline_action_row_index")
    if explicit is None:
        explicit = declared
    elif declared is not None and declared != explicit:
        raise ValueError("explicit v4 baseline disagrees with sample mapping")
    if isinstance(explicit, bool) or not isinstance(explicit, (int, np.integer)):
        raise ValueError("v4 requires an explicit or declared baseline index")
    baseline = int(explicit)
    if not 0 <= baseline < action_count:
        raise IndexError("v4 baseline index is outside actions")
    return baseline


def _positive_probability(estimator: Any, features: np.ndarray) -> float:
    if hasattr(estimator, "predict_proba"):
        raw = np.asarray(estimator.predict_proba(features), dtype=np.float64)
        classes = list(getattr(estimator, "classes_", ()))
        if raw.ndim != 2 or raw.shape[0] != 1:
            raise ValueError("invalid v4 safety probability shape")
        if 1 in classes:
            probability = float(raw[0, classes.index(1)])
        elif classes == [0] and raw.shape[1] == 1:
            probability = 0.0
        else:
            raise ValueError("v4 safety estimator has no positive class")
    elif hasattr(estimator, "decision_function"):
        decision = float(
            np.asarray(estimator.decision_function(features), dtype=np.float64).reshape(-1)[
                0
            ]
        )
        probability = 1.0 / (1.0 + math.exp(-decision))
    else:
        raise TypeError("v4 safety estimator lacks a probability interface")
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("invalid v4 safety probability")
    return probability


def _require_v4_targets(sample: PreparedTeacherSample) -> None:
    if any(
        value is None
        for value in (
            sample.teacher_delta_se_vs_baseline,
            sample.teacher_paired_delta_mean,
            sample.downside_loss_p95,
            sample.downside_loss_p99,
            sample.downside_loss_max,
        )
    ):
        raise ValueError("v4 requires complete paired delta/tail targets")


def _normalize_thresholds(thresholds: Sequence[float]) -> list[float]:
    """Return one deterministic, validated threshold grid."""

    normalized = sorted({float(value) for value in thresholds})
    if not normalized or any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in normalized
    ):
        raise ValueError("v4 safety thresholds must be finite values in [0, 1]")
    return normalized


def _v4_threshold_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    total_states: int,
    threshold: float,
) -> dict[str, Any]:
    """Summarize fired top proposals using paired-future downside bounds.

    The tail gates are deliberately conservative: the reported p95/p99/max
    values are the maxima of the corresponding within-state paired-future
    targets over every fired proposal.  They are not quantiles of the small
    threshold-lock split and therefore keep the same meaning as the teacher
    contract.
    """

    if total_states < 0:
        raise ValueError("total_states must be non-negative")
    deltas = np.asarray(
        [float(row["teacher_delta"]) for row in rows], dtype=np.float64
    )
    p95 = np.asarray(
        [float(row["downside_loss_p95"]) for row in rows], dtype=np.float64
    )
    p99 = np.asarray(
        [float(row["downside_loss_p99"]) for row in rows], dtype=np.float64
    )
    maximum = np.asarray(
        [float(row["downside_loss_max"]) for row in rows], dtype=np.float64
    )
    if not all(
        np.isfinite(values).all()
        for values in (deltas, p95, p99, maximum)
    ):
        raise ValueError("v4 threshold rows contain non-finite values")
    if any(np.any(values < 0.0) for values in (p95, p99, maximum)):
        raise ValueError("v4 threshold downside targets must be non-negative")
    fires = int(deltas.size)
    return {
        "threshold": float(threshold),
        "fires": fires,
        "fire_rate": float(fires / total_states) if total_states else 0.0,
        "teacher_mean_delta_per_fire": (
            float(np.mean(deltas)) if fires else 0.0
        ),
        "teacher_delta_per_state": (
            float(np.sum(deltas) / total_states) if total_states else 0.0
        ),
        "false_positive_rate": (
            float(np.mean(deltas <= 0.0)) if fires else 0.0
        ),
        "p95_loss": float(np.max(p95)) if fires else 0.0,
        "p99_loss": float(np.max(p99)) if fires else 0.0,
        "max_loss": float(np.max(maximum)) if fires else 0.0,
        "loss_metric_source": "selected_action_paired_p05_p01_min_maxima",
        "teacher_value_status": HU_M43_V4_TEACHER_STATUS,
    }


def _validate_v4_fold_arrays(
    arrays: V4FoldTrainingArrays, *, expected_states: int
) -> None:
    rows = arrays.features.shape[0]
    if arrays.features.shape[1] != 4 * HU_FEATURE_DIM:
        raise ValueError("v4 paired feature dimension mismatch")
    for values in (
        arrays.delta,
        arrays.soft_positive,
        arrays.downside_p95,
        arrays.downside_p99,
        arrays.downside_max,
        arrays.weights,
        arrays.state_indices,
        arrays.action_indices,
        arrays.baseline_indices,
    ):
        if values.shape != (rows,):
            raise ValueError("v4 fold array row counts disagree")
    if not all(
        np.isfinite(values).all()
        for values in (
            arrays.features,
            arrays.delta,
            arrays.soft_positive,
            arrays.downside_p95,
            arrays.downside_p99,
            arrays.downside_max,
            arrays.weights,
        )
    ):
        raise ValueError("v4 fold arrays are non-finite")
    if np.any(arrays.action_indices == arrays.baseline_indices):
        raise AssertionError("v4 baseline row entered a paired regressor")
    if set(arrays.state_indices.tolist()) != set(range(expected_states)):
        raise ValueError("v4 fold arrays lost a state")
    for state in range(expected_states):
        if not math.isclose(
            float(np.sum(arrays.weights[arrays.state_indices == state])),
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise AssertionError("v4 per-state fold weight changed")


def _validate_prediction_shape(
    predictions: V4ActionPredictions, action_count: int
) -> None:
    for value in (
        predictions.policy_probability,
        predictions.value,
        predictions.delta_vs_baseline,
        predictions.predicted_absolute_residual,
        predictions.action_score,
        predictions.proposal_score,
        predictions.downside_p95,
        predictions.downside_p99,
        predictions.downside_max,
        predictions.delta_disagreement,
    ):
        if np.asarray(value).shape != (action_count,):
            raise ValueError("v4 prediction/action shape mismatch")


def _ordered_oof_predictions(
    samples: Sequence[PreparedTeacherSample],
    oof_predictions: Sequence[V4OofStatePrediction],
) -> tuple[V4OofStatePrediction, ...]:
    if len(oof_predictions) != len(samples):
        raise ValueError("v4 OOF prediction count mismatch")
    by_index: dict[int, V4OofStatePrediction] = {}
    for row in oof_predictions:
        if row.sample_index in by_index:
            raise ValueError("v4 OOF sample predicted more than once")
        by_index[row.sample_index] = row
    if set(by_index) != set(range(len(samples))):
        raise ValueError("v4 OOF sample coverage is incomplete")
    return tuple(by_index[index] for index in range(len(samples)))


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError("invalid SHA-256")
    return text


__all__ = [
    "HU_M43_V4_ACTION_SCORE_MODE",
    "HU_M43_V4_ARTIFACT_SCHEMA",
    "HU_M43_V4_MODEL_SCHEMA",
    "HU_M43_V4_PRECAL_GATE_SCHEMA",
    "HU_M43_V4_PROPOSAL_SCHEMA",
    "HU_M43_V4_SAFETY_DATASET_SCHEMA",
    "HU_M43_V4_SAFETY_FIT_SCHEMA",
    "HU_M43_V4_SAFETY_FEATURE_DIM",
    "HU_M43_V4_SAFETY_FEATURE_SCHEMA",
    "HU_M43_V4_SELECTION_SCORE",
    "HU_M43_V4_TEACHER_STATUS",
    "HU_M43_V4_THRESHOLD_SCHEMA",
    "HU_M43_V4_TRAINING_MANIFEST_SCHEMA",
    "HU_M43_V4_WORKER_SCHEMA",
    "HuM43JointModelV4",
    "V4ActionPredictions",
    "V4CrossFitResult",
    "V4FoldTrainingArrays",
    "V4FoldWorkerConfig",
    "V4OofSafetyDataset",
    "V4OofStatePrediction",
    "V4PrecalibrationGateConfig",
    "V4RuntimeDecision",
    "V4SafetyFitResult",
    "V4ThresholdSelectionResult",
    "build_v4_fold_training_arrays",
    "build_v4_independent_safety_dataset",
    "build_v4_oof_safety_dataset",
    "build_v4_safety_features",
    "build_v4_training_manifest",
    "evaluate_v4_precalibration_oof_gate",
    "fit_v4_fold_estimator",
    "fit_v4_fold_worker_compatible",
    "fit_v4_nested_crossfit",
    "fit_v4_safety_calibrator",
    "load_hu_m43_joint_model_v4",
    "make_v4_fold_estimator_provider",
    "select_v4_threshold",
    "write_v4_training_manifest",
]
