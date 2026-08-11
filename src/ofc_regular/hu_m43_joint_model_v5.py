"""Opt-in M4.3 Attempt03 stacked proposal model.

The v5 proposal is deliberately separate from the frozen v4 implementation.
Five paired-fold base estimators and the fixed Stage18 action scorer produce a
22-column, action-order-independent feature matrix.  A low-capacity LightGBM
Huber regressor is fit on every nonbaseline action.  Runtime first chooses one
meta-ranked proposal from every nonbaseline action, then accepts that one
proposal only when its centered paired-delta prediction is positive in at
least three folds.  It never falls through to a lower-ranked eligible action.

Teacher values are accepted only by :func:`fit_v5_meta_ranker`.  Runtime
prediction builds a strict policy-only projection before invoking either the
base feature encoder or the injected Stage18 scorer; action ``score``, LCB,
paired-future labels, and other teacher fields never cross that boundary.

LightGBM is an explicit dependency for fitting.  There is intentionally no
silent sklearn fallback because changing the estimator would invalidate the
Attempt03 pre-registration.  Loading and inference need only the estimator
stored in the artifact.
"""

from __future__ import annotations

import hashlib
import math
import os
import pickle
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .action_key import ActionKey, action_key_from_payload
from .hu_m4_joint_model import (
    PairedDeltaRiskFoldEstimator,
    build_paired_action_features_matrix,
)
from .hu_turn3_model import sample_to_matrix


HU_M43_V5_MODEL_SCHEMA = "hu_m43_t1_joint_model_v5"
HU_M43_V5_ARTIFACT_SCHEMA = "hu_m43_t1_joint_model_v5_pickle_v1"
HU_M43_V5_PROPOSAL_SCHEMA = "stage18_stacked_meta_ranker_v1"
HU_M43_V5_ACTION_SCORE_MODE = "eligible_stage18_stacked_meta_ranker_v5"
HU_M43_V5_FEATURE_SCHEMA = "hu_m43_t1_stage18_stacked_features_v5"
HU_M43_V5_SAFETY_FEATURE_SCHEMA = HU_M43_V5_FEATURE_SCHEMA
HU_M43_V5_TRAINING_SCHEMA = "hu_m43_t1_stage18_meta_ranker_fit_v5"
HU_M43_V5_REQUIRED_FOLDS = 5
HU_M43_V5_MIN_POSITIVE_VOTES = 3
HU_M43_V5_META_FEATURE_DIM = 22
HU_M43_V5_SELECTION_SCORE = 1.0
HU_M43_V5_INELIGIBLE_SCORE = -1.0
HU_M43_V5_FIT_ROW_SCOPE = "all_nonbaseline_actions"

HU_M43_V5_FEATURE_NAMES = (
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

if len(HU_M43_V5_FEATURE_NAMES) != HU_M43_V5_META_FEATURE_DIM:
    raise AssertionError("v5 feature schema dimension changed")


@dataclass(frozen=True)
class V5MetaRankerConfig:
    """Frozen low-capacity LightGBM Huber configuration for Attempt03."""

    objective: str = "huber"
    n_estimators: int = 180
    learning_rate: float = 0.035
    num_leaves: int = 7
    min_child_samples: int = 50
    reg_lambda: float = 8.0
    reg_alpha: float = 0.5
    huber_alpha: float = 0.90
    random_state: int = 20260714

    def __post_init__(self) -> None:
        expected = {
            "objective": "huber",
            "n_estimators": 180,
            "learning_rate": 0.035,
            "num_leaves": 7,
            "min_child_samples": 50,
            "reg_lambda": 8.0,
            "reg_alpha": 0.5,
            "huber_alpha": 0.90,
        }
        for name, value in expected.items():
            actual = getattr(self, name)
            if isinstance(value, float):
                valid = math.isfinite(float(actual)) and float(actual) == value
            else:
                valid = actual == value
            if not valid:
                raise ValueError(
                    f"Attempt03 v5 freezes {name}={value!r}; got {actual!r}"
                )
        if isinstance(self.random_state, bool) or not isinstance(
            self.random_state, int
        ):
            raise TypeError("v5 random_state must be an integer")

    def lightgbm_params(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "reg_lambda": self.reg_lambda,
            "reg_alpha": self.reg_alpha,
            "alpha": self.huber_alpha,
            "random_state": self.random_state,
            "n_jobs": 1,
            "deterministic": True,
            "force_col_wise": True,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "verbosity": -1,
        }

    def to_manifest(self) -> dict[str, Any]:
        return {
            "library": "lightgbm",
            **self.lightgbm_params(),
            "loss": "huber",
            "feature_schema": HU_M43_V5_FEATURE_SCHEMA,
            "feature_dim": HU_M43_V5_META_FEATURE_DIM,
            "state_balanced_weights": True,
            "fit_row_scope": HU_M43_V5_FIT_ROW_SCOPE,
            "eligibility_after_canonical_argmax": True,
            "eligible_rerank_allowed": False,
            "positive_vote_eligibility": {
                "folds": HU_M43_V5_REQUIRED_FOLDS,
                "minimum_votes": HU_M43_V5_MIN_POSITIVE_VOTES,
                "predicate": "centered_fold_delta_strictly_greater_than_zero",
                "application_order": (
                    "after_all_nonbaseline_meta_argmax_no_eligible_rerank"
                ),
            },
        }


@dataclass(frozen=True)
class V5BaseHeadPredictions:
    fold_centered_delta: np.ndarray
    delta: np.ndarray
    positive: np.ndarray
    downside_p95: np.ndarray
    downside_p99: np.ndarray
    downside_max: np.ndarray
    delta_disagreement: np.ndarray
    delta_positive_votes: np.ndarray
    eligible_mask: np.ndarray


@dataclass(frozen=True)
class V5StackedFeatures:
    features: np.ndarray
    base: V5BaseHeadPredictions
    stage18_score: np.ndarray
    action_keys: tuple[ActionKey, ...]
    baseline_index: int


@dataclass(frozen=True)
class V5ActionPredictions:
    meta_features: np.ndarray
    meta_score: np.ndarray
    action_score: np.ndarray
    base_delta: np.ndarray
    base_positive: np.ndarray
    downside_p95: np.ndarray
    downside_p99: np.ndarray
    downside_max: np.ndarray
    delta_disagreement: np.ndarray
    delta_positive_votes: np.ndarray
    eligible_mask: np.ndarray
    stage18_score: np.ndarray
    proposal_index: int
    proposal_eligible: bool


@dataclass(frozen=True)
class V5RuntimeDecision:
    baseline_index: int
    proposal_index: int
    selected_index: int
    eligibility_passed: bool
    eligible_action_count: int
    positive_vote_count: int
    safety_probability: float
    override_fired: bool


@dataclass(frozen=True)
class V5MetaRankerFitResult:
    estimator: Any
    sample_weights: np.ndarray
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class HuM43JointModelV5:
    """Opt-in v5 proposal/safety artifact; never registered by this module."""

    paired_fold_estimators: tuple[PairedDeltaRiskFoldEstimator, ...]
    stage18_scorer: Any
    meta_ranker: Any
    safety_estimator: Any | None = None
    safety_threshold: float = 1.0
    safety_enabled: bool = False
    model_id: str = "hu-m43-t1-v5-unfrozen"
    manifest: Mapping[str, Any] = field(default_factory=dict)
    schema: str = HU_M43_V5_MODEL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != HU_M43_V5_MODEL_SCHEMA:
            raise ValueError(f"unsupported v5 model schema: {self.schema!r}")
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("v5 model_id must be non-empty")
        _validate_five_folds(self.paired_fold_estimators)
        if not (
            hasattr(self.stage18_scorer, "predict_sample")
            or callable(self.stage18_scorer)
        ):
            raise TypeError("v5 Stage18 scorer must be callable or expose predict_sample")
        if self.meta_ranker is None or not hasattr(self.meta_ranker, "predict"):
            raise TypeError("v5 meta ranker must expose predict")
        threshold = float(self.safety_threshold)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("v5 safety_threshold must be in [0, 1]")
        if self.safety_enabled and self.safety_estimator is None:
            raise ValueError("enabled v5 safety requires an estimator")

    @property
    def action_score_mode(self) -> str:
        return HU_M43_V5_ACTION_SCORE_MODE

    @property
    def runtime_contract(self) -> dict[str, Any]:
        return {
            "fit_row_scope": HU_M43_V5_FIT_ROW_SCOPE,
            "eligibility_after_canonical_argmax": True,
            "eligible_rerank_allowed": False,
            "minimum_positive_votes": HU_M43_V5_MIN_POSITIVE_VOTES,
            "folds": HU_M43_V5_REQUIRED_FOLDS,
        }

    def predict_heads_sample(
        self,
        sample: Mapping[str, Any],
        *,
        baseline_index: int | None = None,
    ) -> V5ActionPredictions:
        stacked = build_v5_stacked_features(
            sample,
            paired_fold_estimators=self.paired_fold_estimators,
            stage18_scorer=self.stage18_scorer,
            baseline_index=baseline_index,
        )
        meta_score = _regression_prediction(
            self.meta_ranker,
            stacked.features,
            label="v5 meta ranker",
        )
        meta_score = meta_score.astype(np.float64, copy=True)
        meta_score[stacked.baseline_index] = 0.0
        proposal = _canonical_nonbaseline_argmax(
            meta_score,
            action_keys=stacked.action_keys,
            baseline_index=stacked.baseline_index,
        )
        proposal_eligible = bool(stacked.base.eligible_mask[proposal])
        action_score = _selection_scores(
            action_count=meta_score.size,
            proposal_index=proposal,
            proposal_eligible=proposal_eligible,
            baseline_index=stacked.baseline_index,
        )
        return V5ActionPredictions(
            meta_features=stacked.features,
            meta_score=meta_score,
            action_score=action_score,
            base_delta=stacked.base.delta,
            base_positive=stacked.base.positive,
            downside_p95=stacked.base.downside_p95,
            downside_p99=stacked.base.downside_p99,
            downside_max=stacked.base.downside_max,
            delta_disagreement=stacked.base.delta_disagreement,
            delta_positive_votes=stacked.base.delta_positive_votes,
            eligible_mask=stacked.base.eligible_mask,
            stage18_score=stacked.stage18_score,
            proposal_index=proposal,
            proposal_eligible=proposal_eligible,
        )

    def predict_sample_with_baseline(
        self, sample: Mapping[str, Any], *, baseline_index: int
    ) -> np.ndarray:
        return self.predict_heads_sample(
            sample, baseline_index=baseline_index
        ).action_score

    def predict_sample(self, sample: Mapping[str, Any]) -> np.ndarray:
        return self.predict_heads_sample(sample).action_score

    def top_nonbaseline_index(
        self,
        sample: Mapping[str, Any],
        *,
        baseline_index: int,
        predictions: V5ActionPredictions | None = None,
    ) -> int:
        heads = predictions or self.predict_heads_sample(
            sample, baseline_index=baseline_index
        )
        return int(heads.proposal_index)

    def predict_safety_probability(
        self,
        sample: Mapping[str, Any],
        *,
        candidate_index: int,
        baseline_index: int,
    ) -> float:
        if self.safety_estimator is None:
            raise RuntimeError("v5 artifact has no safety estimator")
        heads = self.predict_heads_sample(sample, baseline_index=baseline_index)
        if (
            isinstance(candidate_index, bool)
            or not isinstance(candidate_index, (int, np.integer))
            or not 0 <= int(candidate_index) < heads.eligible_mask.size
        ):
            raise IndexError("v5 safety candidate index is outside actions")
        candidate = int(candidate_index)
        if (
            candidate == baseline_index
            or candidate != heads.proposal_index
            or not heads.proposal_eligible
        ):
            raise ValueError(
                "v5 safety accepts only the eligibility-passing top proposal"
            )
        return _positive_probability(
            self.safety_estimator,
            heads.meta_features[candidate].reshape(1, -1),
        )

    def should_override(
        self,
        sample: Mapping[str, Any],
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
        try:
            probability = self.predict_safety_probability(
                sample,
                candidate_index=candidate_index,
                baseline_index=baseline_index,
            )
        except (IndexError, ValueError):
            return False
        return probability >= float(self.safety_threshold)

    def select_action_index(
        self,
        sample: Mapping[str, Any],
        *,
        baseline_index: int | None = None,
    ) -> V5RuntimeDecision:
        action_count = _action_count(sample)
        baseline = _resolve_baseline_index(sample, baseline_index, action_count)
        heads = self.predict_heads_sample(sample, baseline_index=baseline)
        candidate = heads.proposal_index
        if not heads.proposal_eligible:
            return V5RuntimeDecision(
                baseline_index=baseline,
                proposal_index=candidate,
                selected_index=baseline,
                eligibility_passed=False,
                eligible_action_count=int(np.sum(heads.eligible_mask)),
                positive_vote_count=int(heads.delta_positive_votes[candidate]),
                safety_probability=0.0,
                override_fired=False,
            )
        probability = (
            _positive_probability(
                self.safety_estimator,
                heads.meta_features[candidate].reshape(1, -1),
            )
            if self.safety_estimator is not None
            else 0.0
        )
        fired = bool(
            self.safety_enabled
            and self.safety_estimator is not None
            and probability >= float(self.safety_threshold)
        )
        return V5RuntimeDecision(
            baseline_index=baseline,
            proposal_index=candidate,
            selected_index=candidate if fired else baseline,
            eligibility_passed=True,
            eligible_action_count=int(np.sum(heads.eligible_mask)),
            positive_vote_count=int(heads.delta_positive_votes[candidate]),
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
    ) -> "HuM43JointModelV5":
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
            "artifact_schema": HU_M43_V5_ARTIFACT_SCHEMA,
            "model_schema": HU_M43_V5_MODEL_SCHEMA,
            "proposal_schema": HU_M43_V5_PROPOSAL_SCHEMA,
            "feature_schema": HU_M43_V5_FEATURE_SCHEMA,
            "fit_row_scope": HU_M43_V5_FIT_ROW_SCOPE,
            "eligibility_after_canonical_argmax": True,
            "eligible_rerank_allowed": False,
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
    ) -> "HuM43JointModelV5":
        source = Path(path)
        encoded = source.read_bytes()
        actual = hashlib.sha256(encoded).hexdigest()
        if expected_sha256 is not None and actual != _require_sha256(
            expected_sha256
        ):
            raise ValueError("v5 artifact SHA-256 mismatch")
        payload = pickle.loads(encoded)
        if not isinstance(payload, Mapping):
            raise TypeError("v5 artifact must be a mapping")
        expected = {
            "artifact_schema": HU_M43_V5_ARTIFACT_SCHEMA,
            "model_schema": HU_M43_V5_MODEL_SCHEMA,
            "proposal_schema": HU_M43_V5_PROPOSAL_SCHEMA,
            "feature_schema": HU_M43_V5_FEATURE_SCHEMA,
            "fit_row_scope": HU_M43_V5_FIT_ROW_SCOPE,
            "eligibility_after_canonical_argmax": True,
            "eligible_rerank_allowed": False,
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError("unsupported v5 artifact identity")
        model = payload.get("model")
        if not isinstance(model, cls):
            raise TypeError("v5 artifact contains wrong model type")
        model.__post_init__()
        return model


def build_v5_stacked_features(
    sample: Mapping[str, Any],
    *,
    paired_fold_estimators: Sequence[PairedDeltaRiskFoldEstimator],
    stage18_scorer: Any,
    baseline_index: int | None = None,
) -> V5StackedFeatures:
    """Build the fixed 22-column runtime matrix without teacher inputs."""

    _validate_five_folds(paired_fold_estimators)
    action_count = _action_count(sample)
    baseline = _resolve_baseline_index(sample, baseline_index, action_count)
    runtime_sample = _runtime_policy_projection(sample)
    action_keys = _action_keys(runtime_sample)
    matrix, _unused_targets = sample_to_matrix(runtime_sample)
    if matrix.shape[0] != action_count or not np.isfinite(matrix).all():
        raise ValueError("v5 runtime policy feature matrix is invalid")
    paired = build_paired_action_features_matrix(matrix, baseline_index=baseline)
    ordered_folds = sorted(paired_fold_estimators, key=lambda fold: fold.fold_index)
    outputs = [fold.predict(paired) for fold in ordered_folds]
    fold_delta = np.vstack([output[0] for output in outputs]).astype(np.float64)
    fold_positive = np.vstack([output[1] for output in outputs]).astype(np.float64)
    fold_p95 = np.vstack([output[2] for output in outputs]).astype(np.float64)
    fold_p99 = np.vstack([output[3] for output in outputs]).astype(np.float64)
    fold_max = np.vstack([output[4] for output in outputs]).astype(np.float64)
    fold_centered_delta = fold_delta - fold_delta[:, [baseline]]
    base_delta = np.mean(fold_centered_delta, axis=0)
    base_positive = np.mean(fold_positive, axis=0)
    base_p95 = np.mean(fold_p95, axis=0)
    base_p99 = np.mean(fold_p99, axis=0)
    base_max = np.mean(fold_max, axis=0)
    disagreement = np.std(fold_centered_delta, axis=0)
    votes = np.sum(fold_centered_delta > 0.0, axis=0).astype(np.int8)
    eligible = votes >= HU_M43_V5_MIN_POSITIVE_VOTES
    eligible[baseline] = False

    stage18 = _stage18_predictions(stage18_scorer, runtime_sample, action_count)
    nonbaseline = np.asarray(
        [index for index in range(action_count) if index != baseline], dtype=np.int32
    )
    if nonbaseline.size == 0:
        raise ValueError("v5 requires at least one nonbaseline action")
    stage18_nb = stage18[nonbaseline]
    delta_nb = base_delta[nonbaseline]
    stage18_summary = _summary(stage18_nb)
    delta_summary = _summary(delta_nb)
    feature_rows = np.empty(
        (action_count, HU_M43_V5_META_FEATURE_DIM), dtype=np.float32
    )
    stage18_baseline = float(stage18[baseline])
    for index in range(action_count):
        feature_rows[index] = np.asarray(
            [
                base_delta[index],
                base_positive[index],
                base_p95[index],
                base_p99[index],
                base_max[index],
                disagreement[index],
                stage18[index],
                stage18[index] - stage18_baseline,
                _rank_fraction(stage18[index], stage18_nb),
                _next_margin(index, stage18, nonbaseline),
                *stage18_summary,
                _rank_fraction(base_delta[index], delta_nb),
                _next_margin(index, base_delta, nonbaseline),
                *delta_summary,
                float(nonbaseline.size),
                stage18_baseline,
            ],
            dtype=np.float32,
        )
    if feature_rows.shape != (action_count, HU_M43_V5_META_FEATURE_DIM):
        raise AssertionError("v5 stacked feature shape changed")
    arrays = (
        fold_centered_delta,
        base_delta,
        base_positive,
        base_p95,
        base_p99,
        base_max,
        disagreement,
        stage18,
        feature_rows,
    )
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("v5 stacked prediction contains non-finite values")
    # Baseline-relative quantities are exact, not merely numerically close.
    fold_centered_delta[:, baseline] = 0.0
    base_delta[baseline] = 0.0
    disagreement[baseline] = 0.0
    votes[baseline] = 0
    base = V5BaseHeadPredictions(
        fold_centered_delta=fold_centered_delta,
        delta=base_delta,
        positive=base_positive,
        downside_p95=base_p95,
        downside_p99=base_p99,
        downside_max=base_max,
        delta_disagreement=disagreement,
        delta_positive_votes=votes,
        eligible_mask=eligible,
    )
    return V5StackedFeatures(
        features=feature_rows,
        base=base,
        stage18_score=stage18,
        action_keys=action_keys,
        baseline_index=baseline,
    )


def build_v5_state_balanced_weights(
    state_indices: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Give every represented information set total weight exactly one."""

    states = np.asarray(state_indices)
    if states.ndim != 1 or states.size == 0:
        raise ValueError("v5 state indices must be a non-empty vector")
    if states.dtype.kind not in "iu" or np.any(states < 0):
        raise ValueError("v5 state indices must be non-negative integers")
    weights = np.empty(states.size, dtype=np.float64)
    for state in np.unique(states):
        mask = states == state
        weights[mask] = 1.0 / float(np.sum(mask))
    for state in np.unique(states):
        if not math.isclose(
            float(np.sum(weights[states == state])),
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise AssertionError("v5 state weights do not sum to one")
    return weights


def fit_v5_meta_ranker(
    features: np.ndarray,
    teacher_delta_targets: Sequence[float] | np.ndarray,
    state_indices: Sequence[int] | np.ndarray,
    *,
    config: V5MetaRankerConfig = V5MetaRankerConfig(),
    row_scope: str = HU_M43_V5_FIT_ROW_SCOPE,
) -> V5MetaRankerFitResult:
    """Fit the offline-only stacked ranker using state-balanced weights.

    The reproduced diagnostic fit used every nonbaseline OOF action, not only
    actions passing the 3/5 vote.  Eligibility is applied once to the single
    all-action argmax after prediction.  The explicit ``row_scope`` guard keeps
    later callers from silently moving that filter into training.  The function
    deliberately names its target argument as a teacher input while no runtime
    method has any such parameter.
    """

    if row_scope != HU_M43_V5_FIT_ROW_SCOPE:
        raise ValueError(
            "v5 meta fit row scope is frozen to every nonbaseline action"
        )
    matrix = np.asarray(features, dtype=np.float32)
    targets = np.asarray(teacher_delta_targets, dtype=np.float64).reshape(-1)
    states = np.asarray(state_indices)
    if matrix.ndim != 2 or matrix.shape[1] != HU_M43_V5_META_FEATURE_DIM:
        raise ValueError(
            "v5 meta features must have shape "
            f"(*, {HU_M43_V5_META_FEATURE_DIM})"
        )
    if matrix.shape[0] == 0 or targets.shape != (matrix.shape[0],):
        raise ValueError("v5 meta target/feature row count mismatch")
    if states.shape != (matrix.shape[0],):
        raise ValueError("v5 meta state/feature row count mismatch")
    if not np.isfinite(matrix).all() or not np.isfinite(targets).all():
        raise ValueError("v5 meta training inputs must be finite")
    weights = build_v5_state_balanced_weights(states)
    try:
        from lightgbm import LGBMRegressor
        import lightgbm
    except ImportError as exc:  # pragma: no cover - installed in CI/local GPU env.
        raise RuntimeError(
            "Attempt03 v5 fitting requires lightgbm; no estimator fallback is allowed"
        ) from exc
    estimator = LGBMRegressor(**config.lightgbm_params())
    estimator.fit(
        matrix,
        targets,
        sample_weight=weights,
    )
    prediction = _regression_prediction(estimator, matrix, label="fitted v5 ranker")
    manifest = {
        "schema": HU_M43_V5_TRAINING_SCHEMA,
        "status": "fit_train_only_not_runtime_activated",
        "rows": int(matrix.shape[0]),
        "states": int(np.unique(states).size),
        "fit_row_scope": HU_M43_V5_FIT_ROW_SCOPE,
        "eligibility_after_canonical_argmax": True,
        "eligible_rerank_allowed": False,
        "eligibility_application": (
            "post_all_nonbaseline_argmax_no_eligible_rerank"
        ),
        "feature_names": list(HU_M43_V5_FEATURE_NAMES),
        "config": config.to_manifest(),
        "lightgbm_version": str(lightgbm.__version__),
        "state_weight_sum_min": float(
            min(np.sum(weights[states == state]) for state in np.unique(states))
        ),
        "state_weight_sum_max": float(
            max(np.sum(weights[states == state]) for state in np.unique(states))
        ),
        "prediction_mean": float(np.mean(prediction)),
        "runtime_teacher_inputs": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    return V5MetaRankerFitResult(
        estimator=estimator,
        sample_weights=weights,
        manifest=manifest,
    )


def load_hu_m43_joint_model_v5(
    path: str | Path, *, expected_sha256: str | None = None
) -> HuM43JointModelV5:
    return HuM43JointModelV5.load(path, expected_sha256=expected_sha256)


def _runtime_policy_projection(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Copy only fields consumed by policy feature encoding.

    This allow-list is the runtime teacher-isolation boundary.  In particular,
    action score/SE/paired-delta/LCB fields and root best-action labels are not
    copied even when a teacher row is supplied accidentally.
    """

    allowed_root = (
        "rule_set",
        "schema",
        "phase",
        "seat",
        "to_act_order",
        "board",
        "opponent_board",
        "dead_cards",
        "dealt",
    )
    projected = {key: sample[key] for key in allowed_root if key in sample}
    actions = sample.get("actions")
    if isinstance(actions, (str, bytes)) or not isinstance(actions, Sequence):
        raise ValueError("v5 runtime sample actions must be a sequence")
    projected_actions: list[dict[str, Any]] = []
    for action in actions:
        if not isinstance(action, Mapping):
            raise ValueError("v5 runtime action must be a mapping")
        projected_actions.append(
            {
                key: action[key]
                for key in ("placements", "discards", "next_board")
                if key in action
            }
        )
    projected["actions"] = projected_actions
    return projected


def _validate_five_folds(
    folds: Sequence[PairedDeltaRiskFoldEstimator],
) -> None:
    if len(folds) != HU_M43_V5_REQUIRED_FOLDS:
        raise ValueError("v5 requires exactly five paired-fold estimators")
    indices: list[int] = []
    for index, fold in enumerate(folds):
        if not isinstance(fold, PairedDeltaRiskFoldEstimator):
            raise TypeError(
                f"v5 paired_fold_estimators[{index}] has incompatible type"
            )
        fold.__post_init__()
        indices.append(fold.fold_index)
    if set(indices) != set(range(HU_M43_V5_REQUIRED_FOLDS)):
        raise ValueError("v5 fold indices must be exactly 0..4")


def _stage18_predictions(
    scorer: Any, sample: Mapping[str, Any], action_count: int
) -> np.ndarray:
    if hasattr(scorer, "predict_sample"):
        raw = scorer.predict_sample(dict(sample))
    elif callable(scorer):
        raw = scorer(dict(sample))
    else:
        raise TypeError("v5 Stage18 scorer is unavailable")
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.shape != (action_count,) or not np.isfinite(values).all():
        raise ValueError("v5 Stage18 score/action mapping is invalid")
    return values


def _action_count(sample: Mapping[str, Any]) -> int:
    actions = sample.get("actions")
    if isinstance(actions, (str, bytes)) or not isinstance(actions, Sequence):
        raise ValueError("v5 sample actions must be a sequence")
    if len(actions) < 2:
        raise ValueError("v5 requires at least two legal actions")
    return len(actions)


def _action_keys(sample: Mapping[str, Any]) -> tuple[ActionKey, ...]:
    actions = sample.get("actions", ())
    keys = tuple(action_key_from_payload(action) for action in actions)
    if len(set(keys)) != len(keys):
        raise ValueError("v5 legal action mapping contains duplicate ActionKeys")
    return keys


def _resolve_baseline_index(
    sample: Mapping[str, Any], explicit: int | None, action_count: int
) -> int:
    declared = sample.get("baseline_action_row_index")
    if declared is not None and (
        isinstance(declared, bool)
        or not isinstance(declared, (int, np.integer))
    ):
        raise ValueError("declared v5 baseline index must be an integer")
    if explicit is None:
        explicit = declared
    elif declared is not None and int(declared) != int(explicit):
        raise ValueError("explicit v5 baseline disagrees with sample mapping")
    if isinstance(explicit, bool) or not isinstance(explicit, (int, np.integer)):
        raise ValueError("v5 requires an explicit or declared baseline index")
    baseline = int(explicit)
    if not 0 <= baseline < action_count:
        raise IndexError("v5 baseline index is outside actions")
    return baseline


def _summary(values: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(np.mean(values)),
        float(np.std(values)),
        float(np.max(values)),
        float(np.min(values)),
    )


def _rank_fraction(value: float, nonbaseline_values: np.ndarray) -> float:
    if nonbaseline_values.size <= 1:
        return 1.0
    return float(np.mean(nonbaseline_values < float(value)))


def _next_margin(
    action_index: int, values: np.ndarray, nonbaseline_indices: np.ndarray
) -> float:
    alternatives = nonbaseline_indices[nonbaseline_indices != action_index]
    if alternatives.size == 0:
        return 0.0
    return float(values[action_index] - np.max(values[alternatives]))


def _selection_scores(
    *,
    action_count: int,
    proposal_index: int,
    proposal_eligible: bool,
    baseline_index: int,
) -> np.ndarray:
    if (
        action_count < 2
        or not 0 <= baseline_index < action_count
        or not 0 <= proposal_index < action_count
        or proposal_index == baseline_index
    ):
        raise ValueError("v5 selection score mapping is invalid")
    scores = np.full(
        action_count, HU_M43_V5_INELIGIBLE_SCORE, dtype=np.float64
    )
    scores[baseline_index] = 0.0
    if proposal_eligible:
        scores[proposal_index] = HU_M43_V5_SELECTION_SCORE
    return scores


def _canonical_nonbaseline_argmax(
    values: Sequence[float] | np.ndarray,
    *,
    action_keys: Sequence[ActionKey],
    baseline_index: int,
) -> int:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if (
        array.size != len(action_keys)
        or not 0 <= baseline_index < array.size
        or not np.isfinite(array).all()
    ):
        raise ValueError("v5 nonbaseline action/value mapping is invalid")
    candidates = [
        index for index in range(array.size) if index != baseline_index
    ]
    best = max(float(array[index]) for index in candidates)
    tied = [index for index in candidates if float(array[index]) == best]
    return int(min(tied, key=lambda index: action_keys[index].sort_key()))


def _regression_prediction(
    estimator: Any, features: np.ndarray, *, label: str
) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError(f"{label} input is invalid")
    # LightGBM 4.6 with newer sklearn warns on every ndarray inference because
    # its wrapper synthesizes feature names even when fit from an ndarray.  The
    # frozen Booster has identical no-early-stopping semantics here and avoids
    # that noisy wrapper-only validation path.
    booster = getattr(estimator, "booster_", None)
    predictor = (
        booster
        if booster is not None and hasattr(booster, "predict")
        else estimator
    )
    values = np.asarray(predictor.predict(matrix), dtype=np.float64).reshape(-1)
    if values.shape != (matrix.shape[0],) or not np.isfinite(values).all():
        raise ValueError(f"{label} prediction is invalid")
    return values


def _positive_probability(estimator: Any, features: np.ndarray) -> float:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.shape != (1, HU_M43_V5_META_FEATURE_DIM):
        raise ValueError("v5 safety feature shape is invalid")
    if hasattr(estimator, "predict_proba"):
        raw = np.asarray(estimator.predict_proba(matrix), dtype=np.float64)
        classes = list(getattr(estimator, "classes_", ()))
        if raw.ndim != 2 or raw.shape[0] != 1:
            raise ValueError("v5 safety probability shape is invalid")
        if 1 in classes:
            probability = float(raw[0, classes.index(1)])
        elif classes == [0] and raw.shape[1] == 1:
            probability = 0.0
        else:
            raise ValueError("v5 safety estimator has no positive class")
    elif hasattr(estimator, "decision_function"):
        decision = float(
            np.asarray(estimator.decision_function(matrix), dtype=np.float64).reshape(-1)[
                0
            ]
        )
        probability = 1.0 / (1.0 + math.exp(-decision))
    elif hasattr(estimator, "predict_probability"):
        probability = float(estimator.predict_probability(matrix.reshape(-1)))
    else:
        raise TypeError("v5 safety estimator lacks a probability interface")
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("v5 safety probability is invalid")
    return probability


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_sha256(value: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("expected_sha256 must be a hexadecimal SHA-256 digest")
    return normalized


__all__ = [
    "HU_M43_V5_ACTION_SCORE_MODE",
    "HU_M43_V5_ARTIFACT_SCHEMA",
    "HU_M43_V5_FIT_ROW_SCOPE",
    "HU_M43_V5_FEATURE_NAMES",
    "HU_M43_V5_FEATURE_SCHEMA",
    "HU_M43_V5_META_FEATURE_DIM",
    "HU_M43_V5_MIN_POSITIVE_VOTES",
    "HU_M43_V5_MODEL_SCHEMA",
    "HU_M43_V5_PROPOSAL_SCHEMA",
    "HU_M43_V5_REQUIRED_FOLDS",
    "HU_M43_V5_SAFETY_FEATURE_SCHEMA",
    "HU_M43_V5_TRAINING_SCHEMA",
    "HuM43JointModelV5",
    "V5ActionPredictions",
    "V5BaseHeadPredictions",
    "V5MetaRankerConfig",
    "V5MetaRankerFitResult",
    "V5RuntimeDecision",
    "V5StackedFeatures",
    "build_v5_stacked_features",
    "build_v5_state_balanced_weights",
    "fit_v5_meta_ranker",
    "load_hu_m43_joint_model_v5",
]
