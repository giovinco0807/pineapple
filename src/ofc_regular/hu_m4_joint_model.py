"""Versioned, seat-aware joint action model for the M4 HU T1 policy.

The model deliberately consumes the existing HU ``sample_to_matrix`` feature
contract.  Teacher records are converted to that contract by the trainer from
``policy_observation`` only; replay truth and the opponent's private discards
never enter this module's feature matrix.

``predict_sample`` returns one fixed-composition score per legal action and is
therefore compatible with the existing action-value runtime interface.
Teacher values remain offline diagnostics and are not a runtime EV gate.
"""

from __future__ import annotations

import math
import hashlib
import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .action_key import action_key_from_payload
from .hu_turn3_model import HU_FEATURE_DIM, sample_to_matrix


HU_M4_JOINT_MODEL_SCHEMA = "hu_m4_t1_joint_action_model_v1"
HU_M4_JOINT_ARTIFACT_SCHEMA = "hu_m4_t1_joint_action_model_pickle_v1"
HU_M4_JOINT_FEATURE_SCHEMA = "hu_sample_to_matrix_policy_observation_v1"
HU_M4_JOINT_SAFETY_FEATURE_SCHEMA = "hu_m4_t1_joint_safety_features_v1"
HU_M4_META_RANK_FEATURE_SCHEMA = "hu_m4_t1_meta_rank_features_v2"
HU_M4_PAIRED_ACTION_FEATURE_SCHEMA = "hu_m4_t1_candidate_baseline_pair_v1"

LEGACY_ACTION_SCORE_MODE = "legacy_fixed_composition_v1"
NEGATIVE_REGRET_ACTION_SCORE_MODE = "negative_regret_ranker_v2"
PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE = (
    "baseline_paired_delta_risk_ensemble_v3"
)
ACTION_SCORE_MODES = frozenset(
    {
        LEGACY_ACTION_SCORE_MODE,
        NEGATIVE_REGRET_ACTION_SCORE_MODE,
        PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE,
    }
)

# Values are in teacher-score units.  Policy probability is centered so its
# contribution cannot add a state-wide constant to every action.
VALUE_SCORE_WEIGHT = 1.0
DELTA_SCORE_WEIGHT = 0.5
POLICY_SCORE_WEIGHT = 0.25
POLICY_SCORE_CENTER = 0.5

DEFAULT_POSITIVE_GAIN_SCORE_WEIGHT = 0.25
DEFAULT_DOWNSIDE_RISK_SCORE_WEIGHT = 0.50
DEFAULT_ENSEMBLE_DISAGREEMENT_SCORE_WEIGHT = 0.25


@dataclass(frozen=True)
class JointHeadPredictions:
    """Per-action outputs from every M4 joint-model head."""

    policy_probability: np.ndarray
    value: np.ndarray
    delta_vs_baseline: np.ndarray
    predicted_absolute_residual: np.ndarray
    action_score: np.ndarray


@dataclass(frozen=True)
class ConstantProbabilityEstimator:
    """Small pickle-stable fallback for a one-class safety calibration set."""

    probability: float
    classes_: tuple[int, int] = (0, 1)

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.probability)) or not 0.0 <= float(
            self.probability
        ) <= 1.0:
            raise ValueError("constant probability must be in [0, 1]")

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        rows = np.asarray(features).shape[0]
        positive = np.full(rows, float(self.probability), dtype=np.float64)
        return np.column_stack((1.0 - positive, positive))


@dataclass(frozen=True)
class PairedDeltaRiskFoldEstimator:
    """One identity-fold model used directly by the M4.3 runtime ensemble.

    All three heads consume candidate-versus-explicit-baseline features.  The
    robust delta head estimates paired gain, the soft-positive head estimates
    pairwise win probability, and the downside head estimates adverse-tail
    magnitude.  These estimators never receive teacher fields at runtime.
    """

    delta_estimator: Any
    positive_gain_estimator: Any
    downside_p95_estimator: Any
    downside_p99_estimator: Any
    downside_max_estimator: Any
    paired_feature_dim: int
    fold_index: int
    feature_schema: str = HU_M4_PAIRED_ACTION_FEATURE_SCHEMA

    def __post_init__(self) -> None:
        if self.feature_schema != HU_M4_PAIRED_ACTION_FEATURE_SCHEMA:
            raise ValueError(
                f"unsupported paired action feature schema: {self.feature_schema!r}"
            )
        if self.paired_feature_dim != 4 * HU_FEATURE_DIM:
            raise ValueError(
                "paired feature dimension mismatch: "
                f"{self.paired_feature_dim} != {4 * HU_FEATURE_DIM}"
            )
        if self.fold_index < 0:
            raise ValueError("fold_index must be non-negative")
        for name in (
            "delta_estimator",
            "positive_gain_estimator",
            "downside_p95_estimator",
            "downside_p99_estimator",
            "downside_max_estimator",
        ):
            if getattr(self, name) is None:
                raise ValueError(f"{name} must not be None")

    def predict(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        matrix = np.asarray(features, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.paired_feature_dim:
            raise ValueError(
                "paired fold feature shape mismatch: "
                f"{matrix.shape} expected (*, {self.paired_feature_dim})"
            )
        delta = _regression_prediction(self.delta_estimator, matrix, "paired delta")
        positive = np.clip(
            _regression_prediction(
                self.positive_gain_estimator, matrix, "soft positive gain"
            ),
            0.0,
            1.0,
        )
        downside_p95 = np.maximum(
            0.0,
            _regression_prediction(
                self.downside_p95_estimator, matrix, "downside p95"
            ),
        )
        downside_p99 = np.maximum(
            0.0,
            _regression_prediction(
                self.downside_p99_estimator, matrix, "downside p99"
            ),
        )
        downside_max = np.maximum(
            0.0,
            _regression_prediction(
                self.downside_max_estimator, matrix, "downside max"
            ),
        )
        return delta, positive, downside_p95, downside_p99, downside_max


@dataclass(frozen=True)
class HuM4JointActionModel:
    """Single versioned artifact containing policy/value/delta/risk heads.

    ``safety_estimator`` is learned from its manifest-declared safety-fit
    sources.  ``safety_threshold`` is frozen using only the explicit
    threshold-lock subset.  The locked holdout may be reported in the manifest
    but cannot alter either value.
    """

    policy_estimator: Any
    value_estimator: Any
    delta_estimator: Any
    uncertainty_estimator: Any
    safety_estimator: Any | None = None
    safety_threshold: float = 1.0
    safety_enabled: bool = False
    feature_dim: int = HU_FEATURE_DIM
    model_id: str = "hu-m4-t1-joint-v1"
    manifest: Mapping[str, Any] = field(default_factory=dict)
    schema: str = HU_M4_JOINT_MODEL_SCHEMA
    # Appended after every v1 field so positional construction remains
    # backward compatible as well as pickle loading.
    meta_rank_estimator: Any | None = None
    action_score_mode: str = LEGACY_ACTION_SCORE_MODE
    paired_fold_estimators: tuple[PairedDeltaRiskFoldEstimator, ...] = ()
    positive_gain_score_weight: float = DEFAULT_POSITIVE_GAIN_SCORE_WEIGHT
    downside_risk_score_weight: float = DEFAULT_DOWNSIDE_RISK_SCORE_WEIGHT
    ensemble_disagreement_score_weight: float = (
        DEFAULT_ENSEMBLE_DISAGREEMENT_SCORE_WEIGHT
    )

    def __post_init__(self) -> None:
        # Pickles produced before M4.2 do not have these two attributes.  Fill
        # them with the exact legacy behavior instead of changing the artifact
        # schema or requiring a migration.
        if not hasattr(self, "meta_rank_estimator"):
            object.__setattr__(self, "meta_rank_estimator", None)
        if not hasattr(self, "action_score_mode"):
            object.__setattr__(self, "action_score_mode", LEGACY_ACTION_SCORE_MODE)
        if not hasattr(self, "paired_fold_estimators"):
            object.__setattr__(self, "paired_fold_estimators", ())
        if not hasattr(self, "positive_gain_score_weight"):
            object.__setattr__(
                self,
                "positive_gain_score_weight",
                DEFAULT_POSITIVE_GAIN_SCORE_WEIGHT,
            )
        if not hasattr(self, "downside_risk_score_weight"):
            object.__setattr__(
                self,
                "downside_risk_score_weight",
                DEFAULT_DOWNSIDE_RISK_SCORE_WEIGHT,
            )
        if not hasattr(self, "ensemble_disagreement_score_weight"):
            object.__setattr__(
                self,
                "ensemble_disagreement_score_weight",
                DEFAULT_ENSEMBLE_DISAGREEMENT_SCORE_WEIGHT,
            )
        if self.schema != HU_M4_JOINT_MODEL_SCHEMA:
            raise ValueError(f"unsupported M4 joint model schema: {self.schema!r}")
        if not isinstance(self.feature_dim, int) or isinstance(self.feature_dim, bool):
            raise TypeError("feature_dim must be an integer")
        if self.feature_dim != HU_FEATURE_DIM:
            raise ValueError(
                f"M4 joint feature dimension mismatch: {self.feature_dim} != {HU_FEATURE_DIM}"
            )
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("model_id must be a non-empty string")
        threshold = float(self.safety_threshold)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("safety_threshold must be in [0, 1]")
        if self.safety_enabled and self.safety_estimator is None:
            raise ValueError("enabled safety requires a safety estimator")
        if self.action_score_mode not in ACTION_SCORE_MODES:
            raise ValueError(
                f"unsupported M4 action score mode: {self.action_score_mode!r}"
            )
        if (
            self.action_score_mode
            == PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE
            and not self.paired_fold_estimators
        ):
            raise ValueError("paired-delta runtime mode requires fold estimators")
        for index, fold in enumerate(self.paired_fold_estimators):
            if not isinstance(fold, PairedDeltaRiskFoldEstimator):
                raise TypeError(
                    f"paired_fold_estimators[{index}] must be "
                    "PairedDeltaRiskFoldEstimator"
                )
            fold.__post_init__()
        for name in (
            "positive_gain_score_weight",
            "downside_risk_score_weight",
            "ensemble_disagreement_score_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in (
            "policy_estimator",
            "value_estimator",
            "delta_estimator",
            "uncertainty_estimator",
        ):
            if getattr(self, name) is None:
                raise ValueError(f"{name} must not be None")

    @staticmethod
    def combine_scores(
        value: Sequence[float] | np.ndarray,
        delta_vs_baseline: Sequence[float] | np.ndarray,
        policy_probability: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        value_array = np.asarray(value, dtype=np.float64).reshape(-1)
        delta_array = np.asarray(delta_vs_baseline, dtype=np.float64).reshape(-1)
        policy_array = np.asarray(policy_probability, dtype=np.float64).reshape(-1)
        if not (
            value_array.shape == delta_array.shape == policy_array.shape
        ):
            raise ValueError("joint head output shapes disagree")
        score = (
            VALUE_SCORE_WEIGHT * value_array
            + DELTA_SCORE_WEIGHT * delta_array
            + POLICY_SCORE_WEIGHT * (policy_array - POLICY_SCORE_CENTER)
        )
        if not np.isfinite(score).all():
            raise ValueError("joint action scores are non-finite")
        return score

    def predict_heads_matrix(self, features: np.ndarray) -> JointHeadPredictions:
        matrix = np.asarray(features, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.feature_dim:
            raise ValueError(
                "M4 joint feature shape mismatch: "
                f"{matrix.shape} expected (*, {self.feature_dim})"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("M4 joint features must be finite")
        if self.action_score_mode == PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE:
            raise ValueError(
                "paired-delta runtime prediction requires an explicit baseline index"
            )
        policy = _positive_probability(self.policy_estimator, matrix)
        value = _regression_prediction(self.value_estimator, matrix, "value")
        delta = _regression_prediction(self.delta_estimator, matrix, "delta")
        uncertainty = np.maximum(
            0.0,
            _regression_prediction(
                self.uncertainty_estimator, matrix, "uncertainty"
            ),
        )
        legacy_score = self.combine_scores(value, delta, policy)
        if self.action_score_mode == NEGATIVE_REGRET_ACTION_SCORE_MODE:
            rank_features = build_meta_rank_features(
                policy_probability=policy,
                value=value,
                delta_vs_baseline=delta,
                legacy_score=legacy_score,
            )
            score = (
                _regression_prediction(
                    self.meta_rank_estimator, rank_features, "meta rank"
                )
                if self.meta_rank_estimator is not None
                else legacy_score - float(np.mean(legacy_score))
            )
        else:
            score = legacy_score
        return JointHeadPredictions(
            policy_probability=policy,
            value=value,
            delta_vs_baseline=delta,
            predicted_absolute_residual=uncertainty,
            action_score=score,
        )

    def predict_heads_sample(
        self,
        sample: dict[str, Any],
        *,
        baseline_index: int | None = None,
    ) -> JointHeadPredictions:
        features, _targets = sample_to_matrix(sample)
        if self.action_score_mode == PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE:
            baseline = _resolve_baseline_index(sample, baseline_index, features.shape[0])
            return self._predict_paired_heads_matrix(features, baseline_index=baseline)
        return self.predict_heads_matrix(features)

    def _predict_paired_heads_matrix(
        self,
        features: np.ndarray,
        *,
        baseline_index: int,
    ) -> JointHeadPredictions:
        matrix = np.asarray(features, dtype=np.float32)
        paired = build_paired_action_features_matrix(
            matrix, baseline_index=baseline_index
        )
        fold_outputs = [fold.predict(paired) for fold in self.paired_fold_estimators]
        fold_deltas = np.vstack([output[0] for output in fold_outputs])
        fold_positive = np.vstack([output[1] for output in fold_outputs])
        fold_downside_p95 = np.vstack([output[2] for output in fold_outputs])
        fold_downside_p99 = np.vstack([output[3] for output in fold_outputs])
        fold_downside_max = np.vstack([output[4] for output in fold_outputs])

        # Center *each fold's entire composite* on that fold's baseline.  It is
        # not sufficient to center only delta: a state-wide positive/risk-head
        # bias could otherwise manufacture a non-baseline override after the
        # baseline entry was merely overwritten with zero.
        fold_deltas = fold_deltas - fold_deltas[:, [baseline_index]]
        fold_downside = (
            0.50 * fold_downside_p95
            + 0.30 * fold_downside_p99
            + 0.20 * fold_downside_max
        )
        fold_composite = (
            fold_deltas
            + float(self.positive_gain_score_weight) * (2.0 * fold_positive - 1.0)
            - float(self.downside_risk_score_weight) * fold_downside
        )
        fold_composite = fold_composite - fold_composite[:, [baseline_index]]
        delta = np.mean(fold_deltas, axis=0)
        positive = np.mean(fold_positive, axis=0)
        downside_p95 = np.mean(fold_downside_p95, axis=0)
        downside_p99 = np.mean(fold_downside_p99, axis=0)
        downside_max = np.mean(fold_downside_max, axis=0)
        downside = 0.50 * downside_p95 + 0.30 * downside_p99 + 0.20 * downside_max
        disagreement = np.std(fold_composite, axis=0)
        uncertainty = downside + disagreement
        score = np.mean(fold_composite, axis=0) - float(
            self.ensemble_disagreement_score_weight
        ) * disagreement
        delta[baseline_index] = 0.0
        positive[baseline_index] = 0.5
        downside[baseline_index] = 0.0
        downside_p95[baseline_index] = 0.0
        downside_p99[baseline_index] = 0.0
        downside_max[baseline_index] = 0.0
        disagreement[baseline_index] = 0.0
        uncertainty[baseline_index] = 0.0
        score[baseline_index] = 0.0
        if not all(
            np.isfinite(values).all()
            for values in (
                delta,
                positive,
                downside_p95,
                downside_p99,
                downside_max,
                downside,
                disagreement,
                uncertainty,
                score,
            )
        ):
            raise ValueError("paired-delta ensemble returned non-finite values")
        return JointHeadPredictions(
            policy_probability=positive,
            # M4.3 is deliberately baseline-relative; it has no absolute-value
            # runtime head.  Keep the compatibility slot relative as well.
            value=delta.copy(),
            delta_vs_baseline=delta,
            predicted_absolute_residual=uncertainty,
            action_score=score,
        )

    def predict_matrix(self, features: np.ndarray) -> np.ndarray:
        """Return the fixed-composition action score (runtime compatibility)."""

        return self.predict_heads_matrix(features).action_score

    def predict_sample(self, sample: dict[str, Any]) -> np.ndarray:
        """Return one sortable score per legal action.

        Runtime callers must build *sample* from ``ActorObservation``.  The M4
        selective policy already follows that boundary.
        """

        return self.predict_heads_sample(sample).action_score

    def predict_sample_with_baseline(
        self,
        sample: dict[str, Any],
        *,
        baseline_index: int,
    ) -> np.ndarray:
        """Return action scores relative to an explicitly resolved baseline.

        Legacy/M4.2 modes accept the argument but preserve their exact former
        prediction path.  M4.3 requires it and guarantees the corresponding
        score is exactly ``0.0``.
        """

        if self.action_score_mode != PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE:
            return self.predict_sample(sample)
        return self.predict_heads_sample(
            sample, baseline_index=baseline_index
        ).action_score

    def choose_action_index(self, sample: dict[str, Any]) -> int:
        scores = self.predict_sample(sample)
        return canonical_action_argmax(sample, scores)

    def predict_safety_probability(
        self,
        sample: dict[str, Any],
        *,
        candidate_index: int,
        baseline_index: int,
    ) -> float:
        if self.safety_estimator is None:
            raise RuntimeError("M4 joint artifact has no calibrated safety head")
        heads = self.predict_heads_sample(sample, baseline_index=baseline_index)
        features = build_joint_safety_features(
            heads,
            candidate_index=candidate_index,
            baseline_index=baseline_index,
            seat=str(sample.get("seat", "")),
        ).reshape(1, -1)
        probability = float(_positive_probability(self.safety_estimator, features)[0])
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("invalid safety probability")
        return probability

    def should_override(
        self,
        sample: dict[str, Any],
        *,
        candidate_index: int,
        baseline_index: int,
    ) -> bool:
        if not self.safety_enabled or candidate_index == baseline_index:
            return False
        return (
            self.predict_safety_probability(
                sample,
                candidate_index=candidate_index,
                baseline_index=baseline_index,
            )
            >= self.safety_threshold
        )

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "artifact_schema": HU_M4_JOINT_ARTIFACT_SCHEMA,
            "model_schema": HU_M4_JOINT_MODEL_SCHEMA,
            "feature_schema": HU_M4_JOINT_FEATURE_SCHEMA,
            "model": self,
        }
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
        freeze_manifest: str | Path | Mapping[str, Any] | None = None,
        training_manifest_path: str | Path | None = None,
    ) -> "HuM4JointActionModel":
        source = Path(path)
        encoded = source.read_bytes()
        actual_sha256 = hashlib.sha256(encoded).hexdigest()
        freeze = _load_freeze_manifest(freeze_manifest)
        if freeze is not None:
            if freeze.get("schema") != "hu_m43_model_threshold_freeze_v1":
                raise ValueError("unsupported M4.3 freeze manifest schema")
            if freeze.get("status") != "model_and_threshold_frozen_locked_unopened":
                raise ValueError("M4.3 freeze manifest is not in frozen state")
            frozen_sha = _sha256_text(
                freeze.get("model_sha256"), "freeze model SHA-256"
            )
        else:
            frozen_sha = None
        if expected_sha256 is not None:
            expected = _sha256_text(expected_sha256, "expected model SHA-256")
            if actual_sha256 != expected:
                raise ValueError("M4 joint artifact SHA-256 mismatch")
            if frozen_sha is not None and frozen_sha != expected:
                raise ValueError("expected artifact SHA disagrees with freeze manifest")
        if frozen_sha is not None and actual_sha256 != frozen_sha:
            raise ValueError("M4 joint artifact SHA disagrees with freeze manifest")
        if training_manifest_path is not None:
            if freeze is None:
                raise ValueError(
                    "training_manifest_path requires a frozen runtime manifest"
                )
            actual_training_sha = hashlib.sha256(
                Path(training_manifest_path).read_bytes()
            ).hexdigest()
            frozen_training_sha = _sha256_text(
                freeze.get("training_manifest_sha256"),
                "freeze training manifest SHA-256",
            )
            if actual_training_sha != frozen_training_sha:
                raise ValueError("training manifest SHA disagrees with freeze")
        payload = pickle.loads(encoded)
        if not isinstance(payload, dict):
            raise TypeError("M4 joint artifact must be a mapping")
        if payload.get("artifact_schema") != HU_M4_JOINT_ARTIFACT_SCHEMA:
            raise ValueError("unsupported M4 joint artifact schema")
        if payload.get("model_schema") != HU_M4_JOINT_MODEL_SCHEMA:
            raise ValueError("unsupported M4 joint model schema")
        if payload.get("feature_schema") != HU_M4_JOINT_FEATURE_SCHEMA:
            raise ValueError("unsupported M4 joint feature schema")
        model = payload.get("model")
        if not isinstance(model, cls):
            raise TypeError(f"expected {cls.__name__}, got {type(model).__name__}")
        model.__post_init__()
        if freeze is not None:
            if str(freeze.get("model_id", "")) != model.model_id:
                raise ValueError("M4 joint model_id disagrees with freeze manifest")
            frozen_threshold = float(freeze.get("frozen_threshold"))
            if not math.isclose(
                frozen_threshold,
                float(model.safety_threshold),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError("M4 safety threshold disagrees with freeze manifest")
        return model


def load_hu_m4_joint_action_model(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    freeze_manifest: str | Path | Mapping[str, Any] | None = None,
    training_manifest_path: str | Path | None = None,
) -> HuM4JointActionModel:
    return HuM4JointActionModel.load(
        path,
        expected_sha256=expected_sha256,
        freeze_manifest=freeze_manifest,
        training_manifest_path=training_manifest_path,
    )


def canonical_action_argmax(
    sample: Mapping[str, Any], values: Sequence[float] | np.ndarray
) -> int:
    """Enumeration-independent argmax for JSON teacher/runtime actions."""

    actions = sample.get("actions", ())
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if isinstance(actions, (str, bytes)) or len(actions) != array.shape[0] or not actions:
        raise ValueError("action/value length mismatch")
    if not np.isfinite(array).all():
        raise ValueError("action scores must be finite")
    best = float(array.max())
    tied = np.flatnonzero(array == best)
    return int(
        min(
            tied,
            key=lambda index: action_key_from_payload(actions[int(index)]).sort_key(),
        )
    )


def build_joint_safety_features(
    heads: JointHeadPredictions,
    *,
    candidate_index: int,
    baseline_index: int,
    seat: str,
) -> np.ndarray:
    """Build compact safety features from predictions, never teacher values."""

    if seat not in {"first", "second"}:
        raise ValueError(f"invalid seat: {seat!r}")
    count = heads.action_score.shape[0]
    if not 0 <= candidate_index < count or not 0 <= baseline_index < count:
        raise IndexError("candidate/baseline index is outside the legal action list")

    matrix = np.column_stack(
        (
            heads.policy_probability,
            heads.value,
            heads.delta_vs_baseline,
            heads.predicted_absolute_residual,
            heads.action_score,
        )
    ).astype(np.float64, copy=False)
    candidate = matrix[candidate_index]
    baseline = matrix[baseline_index]
    other = np.delete(heads.action_score, candidate_index)
    next_margin = (
        float(heads.action_score[candidate_index] - np.max(other))
        if other.size
        else 0.0
    )
    result = np.concatenate(
        (
            candidate,
            baseline,
            candidate - baseline,
            np.asarray(
                [next_margin, float(seat == "first"), float(seat == "second")],
                dtype=np.float64,
            ),
        )
    ).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("joint safety features must be finite")
    return result


def build_paired_action_features_matrix(
    features: np.ndarray,
    *,
    baseline_index: int,
) -> np.ndarray:
    """Encode every candidate against the exact baseline action row.

    The layout is ``candidate, baseline, candidate-baseline, abs(delta)``.
    It is independent of legal-action enumeration order once the semantic
    baseline mapping has been resolved by the caller.
    """

    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != HU_FEATURE_DIM or not matrix.shape[0]:
        raise ValueError(
            "paired action source shape mismatch: "
            f"{matrix.shape} expected (N, {HU_FEATURE_DIM})"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("paired action source features must be finite")
    if isinstance(baseline_index, bool) or not isinstance(
        baseline_index, (int, np.integer)
    ):
        raise TypeError("baseline_index must be an integer")
    baseline = int(baseline_index)
    if not 0 <= baseline < matrix.shape[0]:
        raise IndexError("baseline_index is outside the legal action list")
    baseline_rows = np.repeat(matrix[[baseline]], matrix.shape[0], axis=0)
    difference = matrix - baseline_rows
    result = np.concatenate(
        (matrix, baseline_rows, difference, np.abs(difference)), axis=1
    ).astype(np.float32, copy=False)
    if result.shape != (matrix.shape[0], 4 * HU_FEATURE_DIM):
        raise AssertionError("paired action feature dimension changed unexpectedly")
    if not np.isfinite(result).all():
        raise ValueError("paired action features must be finite")
    return result


def build_paired_action_features(
    sample: dict[str, Any],
    *,
    baseline_index: int,
) -> np.ndarray:
    matrix, _targets = sample_to_matrix(sample)
    return build_paired_action_features_matrix(
        matrix, baseline_index=baseline_index
    )


def build_meta_rank_features(
    *,
    policy_probability: Sequence[float] | np.ndarray,
    value: Sequence[float] | np.ndarray,
    delta_vs_baseline: Sequence[float] | np.ndarray,
    legacy_score: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Build state-relative features for the opt-in negative-regret ranker.

    Every column is invariant to a state-wide additive offset.  This prevents
    the meta estimator from recovering the absolute teacher-value level and
    makes its output solely an ordering score within one legal-action set.
    """

    policy = np.asarray(policy_probability, dtype=np.float64).reshape(-1)
    value_array = np.asarray(value, dtype=np.float64).reshape(-1)
    delta = np.asarray(delta_vs_baseline, dtype=np.float64).reshape(-1)
    if not (policy.shape == value_array.shape == delta.shape) or not policy.size:
        raise ValueError("meta rank head output shapes disagree")
    legacy = (
        np.asarray(legacy_score, dtype=np.float64).reshape(-1)
        if legacy_score is not None
        else HuM4JointActionModel.combine_scores(value_array, delta, policy)
    )
    if legacy.shape != policy.shape:
        raise ValueError("meta rank legacy score shape disagrees")
    if not all(np.isfinite(array).all() for array in (policy, value_array, delta, legacy)):
        raise ValueError("meta rank inputs must be finite")

    columns: list[np.ndarray] = []
    for array in (policy, value_array, delta, legacy):
        centered = array - float(np.mean(array))
        scale = max(float(np.std(centered)), 1.0e-6)
        columns.extend(
            (
                centered,
                centered / scale,
                array - float(np.max(array)),
            )
        )
    result = np.column_stack(columns).astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise ValueError("meta rank features must be finite")
    return result


def _resolve_baseline_index(
    sample: Mapping[str, Any],
    explicit: int | None,
    action_count: int,
) -> int:
    declared = sample.get("baseline_action_row_index")
    if explicit is None:
        explicit = declared
    elif declared is not None and declared != explicit:
        raise ValueError("explicit baseline index disagrees with sample mapping")
    if isinstance(explicit, bool) or not isinstance(explicit, (int, np.integer)):
        raise ValueError(
            "paired-delta runtime requires baseline_action_row_index or an "
            "explicit baseline_index"
        )
    baseline = int(explicit)
    if not 0 <= baseline < action_count:
        raise IndexError("baseline index is outside the legal action list")
    return baseline


def _positive_probability(estimator: Any, features: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        raw = np.asarray(estimator.predict_proba(features), dtype=np.float64)
        if raw.ndim != 2 or raw.shape[0] != features.shape[0]:
            raise ValueError("invalid predict_proba output shape")
        classes = list(getattr(estimator, "classes_", ()))
        if 1 in classes:
            index = classes.index(1)
        elif raw.shape[1] == 1 and classes == [0]:
            return np.zeros(features.shape[0], dtype=np.float64)
        else:
            index = raw.shape[1] - 1
        probability = raw[:, index]
    elif hasattr(estimator, "decision_function"):
        decision = np.asarray(estimator.decision_function(features), dtype=np.float64).reshape(-1)
        probability = np.empty_like(decision)
        positive = decision >= 0.0
        probability[positive] = 1.0 / (1.0 + np.exp(-decision[positive]))
        exp_value = np.exp(decision[~positive])
        probability[~positive] = exp_value / (1.0 + exp_value)
    else:
        prediction = np.asarray(estimator.predict(features), dtype=np.float64).reshape(-1)
        probability = prediction
    if probability.shape != (features.shape[0],):
        raise ValueError("probability head output length mismatch")
    if not np.isfinite(probability).all() or np.any(probability < 0.0) or np.any(
        probability > 1.0
    ):
        raise ValueError("probability head returned values outside [0, 1]")
    return probability


def _regression_prediction(estimator: Any, features: np.ndarray, name: str) -> np.ndarray:
    prediction = np.asarray(estimator.predict(features), dtype=np.float64).reshape(-1)
    if prediction.shape != (features.shape[0],):
        raise ValueError(f"{name} head output length mismatch")
    if not np.isfinite(prediction).all():
        raise ValueError(f"{name} head returned non-finite values")
    return prediction


def _load_freeze_manifest(
    source: str | Path | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if source is None:
        return None
    if isinstance(source, Mapping):
        return dict(source)
    value = json.loads(Path(source).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("M4.3 freeze manifest must be a mapping")
    return value


def _sha256_text(value: Any, location: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{location} must be a SHA-256 string")
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{location} must be a SHA-256 string")
    return normalized


__all__ = [
    "ACTION_SCORE_MODES",
    "ConstantProbabilityEstimator",
    "DEFAULT_DOWNSIDE_RISK_SCORE_WEIGHT",
    "DEFAULT_ENSEMBLE_DISAGREEMENT_SCORE_WEIGHT",
    "DEFAULT_POSITIVE_GAIN_SCORE_WEIGHT",
    "DELTA_SCORE_WEIGHT",
    "HU_M4_JOINT_ARTIFACT_SCHEMA",
    "HU_M4_JOINT_FEATURE_SCHEMA",
    "HU_M4_JOINT_MODEL_SCHEMA",
    "HU_M4_PAIRED_ACTION_FEATURE_SCHEMA",
    "HU_M4_JOINT_SAFETY_FEATURE_SCHEMA",
    "HU_M4_META_RANK_FEATURE_SCHEMA",
    "HuM4JointActionModel",
    "JointHeadPredictions",
    "LEGACY_ACTION_SCORE_MODE",
    "NEGATIVE_REGRET_ACTION_SCORE_MODE",
    "PAIRED_DELTA_RISK_ENSEMBLE_ACTION_SCORE_MODE",
    "PairedDeltaRiskFoldEstimator",
    "POLICY_SCORE_CENTER",
    "POLICY_SCORE_WEIGHT",
    "VALUE_SCORE_WEIGHT",
    "build_joint_safety_features",
    "build_meta_rank_features",
    "build_paired_action_features",
    "build_paired_action_features_matrix",
    "canonical_action_argmax",
    "load_hu_m4_joint_action_model",
]
