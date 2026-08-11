"""Opt-in M4.3 Attempt04 direct paired-risk model.

Attempt04 deliberately removes the absolute-zero fold vote that starved the
Attempt03 policy.  Five paired-action estimators are averaged, a candidate is
chosen from the model-predicted tail-safe pool, and a separate state-level
safety estimator decides whether that proposal may fire.  Teacher fields are
never accepted by runtime methods.
"""

from __future__ import annotations

import hashlib
import math
import os
import pickle
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .action_key import (
    ActionKey,
    action_key,
    action_key_from_payload,
    index_actions_by_key,
)
from .action_space import generate_turn_actions
from .hu_infoset import ActorObservation
from .hu_m4_joint_model import (
    PairedDeltaRiskFoldEstimator,
    build_paired_action_features_matrix,
)
from .hu_turn3_model import hu_policy_sample, sample_to_matrix


HU_M43_V6_MODEL_SCHEMA = "hu_m43_t1_joint_model_v6"
HU_M43_V6_ARTIFACT_SCHEMA = "hu_m43_t1_joint_model_v6_pickle_v1"
HU_M43_V6_PROPOSAL_SCHEMA = "tail_bounded_paired_delta_ensemble_v1"
HU_M43_V6_ACTION_SCORE_MODE = "tail_bounded_paired_delta_then_safety_v6"
HU_M43_V6_SAFETY_FEATURE_SCHEMA = "hu_m43_t1_v6_proposal_context_v1"
HU_M43_V6_REQUIRED_FOLDS = 5
HU_M43_V6_SAFETY_FEATURE_DIM = 19
HU_M43_V6_SELECTION_SCORE = 1.0
HU_M43_V6_INELIGIBLE_SCORE = -1.0

HU_M43_V6_SAFETY_FEATURE_NAMES = (
    "base_delta",
    "base_positive",
    "base_downside_p95",
    "base_downside_p99",
    "base_downside_max",
    "upper_downside_p95",
    "upper_downside_p99",
    "upper_downside_max",
    "delta_disagreement",
    "delta_next_margin",
    "delta_rank_fraction",
    "delta_nonbaseline_mean",
    "delta_nonbaseline_std",
    "delta_nonbaseline_max",
    "delta_nonbaseline_min",
    "nonbaseline_action_count",
    "tail_pool_action_count",
    "tail_pool_fraction",
    "risk_eligible",
)

if len(HU_M43_V6_SAFETY_FEATURE_NAMES) != HU_M43_V6_SAFETY_FEATURE_DIM:
    raise AssertionError("v6 safety feature schema dimension changed")


@dataclass(frozen=True)
class V6ActionPredictions:
    action_score: np.ndarray
    base_delta: np.ndarray
    base_positive: np.ndarray
    downside_p95: np.ndarray
    downside_p99: np.ndarray
    downside_max: np.ndarray
    upper_downside_p95: np.ndarray
    upper_downside_p99: np.ndarray
    upper_downside_max: np.ndarray
    delta_disagreement: np.ndarray
    risk_eligible_mask: np.ndarray
    proposal_index: int
    proposal_risk_eligible: bool
    safety_features: np.ndarray


@dataclass(frozen=True)
class V6RuntimeDecision:
    baseline_index: int
    proposal_index: int
    selected_index: int
    risk_eligible: bool
    safety_probability: float
    override_fired: bool


@dataclass(frozen=True)
class HuM43JointModelV6:
    """Versioned, opt-in v6 artifact; registration is intentionally external."""

    paired_fold_estimators: tuple[PairedDeltaRiskFoldEstimator, ...]
    safety_estimator: Any | None = None
    safety_threshold: float = 1.0
    safety_enabled: bool = False
    tail_cushions: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tail_limits: tuple[float, float, float] = (25.0, 40.0, 50.0)
    model_id: str = "hu-m43-t1-v6-unfrozen"
    manifest: Mapping[str, Any] = field(default_factory=dict)
    schema: str = HU_M43_V6_MODEL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != HU_M43_V6_MODEL_SCHEMA:
            raise ValueError(f"unsupported v6 model schema: {self.schema!r}")
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("v6 model_id must be non-empty")
        _validate_folds(self.paired_fold_estimators)
        threshold = float(self.safety_threshold)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("v6 safety_threshold must be in [0, 1]")
        if self.safety_enabled and self.safety_estimator is None:
            raise ValueError("enabled v6 safety requires an estimator")
        for label, values in (
            ("tail_cushions", self.tail_cushions),
            ("tail_limits", self.tail_limits),
        ):
            if len(values) != 3 or not all(math.isfinite(float(v)) for v in values):
                raise ValueError(f"v6 {label} must contain three finite values")
        if any(float(value) < 0.0 for value in self.tail_cushions):
            raise ValueError("v6 tail cushions must be non-negative")
        if any(float(value) <= 0.0 for value in self.tail_limits):
            raise ValueError("v6 tail limits must be positive")

    @property
    def action_score_mode(self) -> str:
        return HU_M43_V6_ACTION_SCORE_MODE

    @property
    def runtime_contract(self) -> dict[str, Any]:
        return {
            "proposal_schema": HU_M43_V6_PROPOSAL_SCHEMA,
            "proposal_tie_break": "base_delta_then_action_key",
            "fold_absolute_zero_vote_gate": False,
            "tail_bounds_before_safety": True,
            "safety_feature_schema": HU_M43_V6_SAFETY_FEATURE_SCHEMA,
            "runtime_teacher_inputs": False,
        }

    def predict_heads_sample(
        self,
        sample: Mapping[str, Any],
        *,
        baseline_index: int | None = None,
    ) -> V6ActionPredictions:
        action_count = _action_count(sample)
        baseline = _resolve_baseline_index(sample, baseline_index, action_count)
        runtime_sample = _runtime_policy_projection(sample)
        matrix, _unused = sample_to_matrix(runtime_sample)
        paired = build_paired_action_features_matrix(matrix, baseline_index=baseline)
        ordered = sorted(self.paired_fold_estimators, key=lambda fold: fold.fold_index)
        outputs = [fold.predict(paired) for fold in ordered]
        fold_delta = np.vstack([output[0] for output in outputs]).astype(np.float64)
        fold_centered = fold_delta - fold_delta[:, [baseline]]
        delta = np.mean(fold_centered, axis=0)
        positive = np.mean(np.vstack([output[1] for output in outputs]), axis=0)
        p95 = np.mean(np.vstack([output[2] for output in outputs]), axis=0)
        p99 = np.mean(np.vstack([output[3] for output in outputs]), axis=0)
        maximum = np.mean(np.vstack([output[4] for output in outputs]), axis=0)
        disagreement = np.std(fold_centered, axis=0)
        cushions = np.asarray(self.tail_cushions, dtype=np.float64)
        limits = np.asarray(self.tail_limits, dtype=np.float64)
        upper95 = p95 + cushions[0]
        upper99 = p99 + cushions[1]
        uppermax = maximum + cushions[2]
        risk_mask = (upper95 <= limits[0]) & (upper99 <= limits[1]) & (
            uppermax <= limits[2]
        )
        risk_mask[baseline] = False
        nonbaseline = np.asarray(
            [index for index in range(action_count) if index != baseline], dtype=np.int32
        )
        pool = nonbaseline[risk_mask[nonbaseline]]
        selection_pool = pool if pool.size else nonbaseline
        proposal = _canonical_delta_argmax(
            delta,
            candidate_indices=selection_pool,
            action_keys=_action_keys(runtime_sample),
        )
        risk_eligible = bool(pool.size and risk_mask[proposal])
        safety_features = _build_safety_features(
            proposal_index=proposal,
            baseline_index=baseline,
            delta=delta,
            positive=positive,
            p95=p95,
            p99=p99,
            maximum=maximum,
            upper95=upper95,
            upper99=upper99,
            uppermax=uppermax,
            disagreement=disagreement,
            risk_mask=risk_mask,
        )
        action_score = np.full(action_count, HU_M43_V6_INELIGIBLE_SCORE, dtype=np.float64)
        action_score[baseline] = 0.0
        if risk_eligible:
            action_score[proposal] = HU_M43_V6_SELECTION_SCORE
        for array in (
            action_score,
            delta,
            positive,
            p95,
            p99,
            maximum,
            upper95,
            upper99,
            uppermax,
            disagreement,
            safety_features,
        ):
            if not np.isfinite(array).all():
                raise ValueError("v6 prediction contains non-finite values")
        delta[baseline] = 0.0
        disagreement[baseline] = 0.0
        return V6ActionPredictions(
            action_score=action_score,
            base_delta=delta,
            base_positive=positive,
            downside_p95=p95,
            downside_p99=p99,
            downside_max=maximum,
            upper_downside_p95=upper95,
            upper_downside_p99=upper99,
            upper_downside_max=uppermax,
            delta_disagreement=disagreement,
            risk_eligible_mask=risk_mask,
            proposal_index=proposal,
            proposal_risk_eligible=risk_eligible,
            safety_features=safety_features,
        )

    def predict_sample_with_baseline(
        self, sample: Mapping[str, Any], *, baseline_index: int
    ) -> np.ndarray:
        return self.predict_heads_sample(
            sample, baseline_index=baseline_index
        ).action_score

    def predict_sample(self, sample: Mapping[str, Any]) -> np.ndarray:
        return self.predict_heads_sample(sample).action_score

    def predict_safety_probability(
        self,
        sample: Mapping[str, Any],
        *,
        candidate_index: int,
        baseline_index: int,
    ) -> float:
        if self.safety_estimator is None:
            raise RuntimeError("v6 artifact has no safety estimator")
        heads = self.predict_heads_sample(sample, baseline_index=baseline_index)
        if isinstance(candidate_index, (bool, np.bool_)) or not isinstance(
            candidate_index, (int, np.integer)
        ):
            raise TypeError("v6 candidate index must be an integer")
        candidate = int(candidate_index)
        if not 0 <= candidate < _action_count(sample):
            raise IndexError("v6 candidate index is outside actions")
        if candidate != heads.proposal_index or not heads.proposal_risk_eligible:
            return 0.0
        return _positive_probability(
            self.safety_estimator, heads.safety_features.reshape(1, -1)
        )

    def select_action_index(
        self, sample: Mapping[str, Any], *, baseline_index: int | None = None
    ) -> V6RuntimeDecision:
        baseline = _resolve_baseline_index(
            sample, baseline_index, _action_count(sample)
        )
        heads = self.predict_heads_sample(sample, baseline_index=baseline)
        probability = (
            self.predict_safety_probability(
                sample,
                candidate_index=heads.proposal_index,
                baseline_index=baseline,
            )
            if self.safety_estimator is not None and heads.proposal_risk_eligible
            else 0.0
        )
        fired = bool(
            heads.proposal_risk_eligible
            and self.safety_enabled
            and self.safety_estimator is not None
            and probability >= float(self.safety_threshold)
        )
        return V6RuntimeDecision(
            baseline_index=baseline,
            proposal_index=heads.proposal_index,
            selected_index=heads.proposal_index if fired else baseline,
            risk_eligible=heads.proposal_risk_eligible,
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
    ) -> "HuM43JointModelV6":
        return replace(
            self,
            safety_estimator=estimator,
            safety_threshold=float(threshold),
            safety_enabled=bool(enabled),
            manifest=self.manifest if manifest is None else dict(manifest),
        )

    def with_frozen_tail_cushions(
        self,
        cushions: Sequence[float],
        *,
        manifest: Mapping[str, Any] | None = None,
    ) -> "HuM43JointModelV6":
        """Freeze proposal bounds before any safety calibration is fitted."""

        if self.safety_estimator is not None or self.safety_enabled:
            raise RuntimeError("v6 tail cushions must freeze before safety fitting")
        return replace(
            self,
            tail_cushions=tuple(float(value) for value in cushions),
            manifest=self.manifest if manifest is None else dict(manifest),
        )

    def save(self, path: str | Path) -> str:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "artifact_schema": HU_M43_V6_ARTIFACT_SCHEMA,
            "model_schema": HU_M43_V6_MODEL_SCHEMA,
            "proposal_schema": HU_M43_V6_PROPOSAL_SCHEMA,
            "safety_feature_schema": HU_M43_V6_SAFETY_FEATURE_SCHEMA,
            "model": self,
        }
        encoded = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        temporary = destination.with_name(
            f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with temporary.open("xb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            # A hard-link publishes the fully-fsynced inode atomically and
            # fails if the immutable destination already exists.
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def load(
        cls, path: str | Path, *, expected_sha256: str | None = None
    ) -> "HuM43JointModelV6":
        encoded = Path(path).read_bytes()
        actual = hashlib.sha256(encoded).hexdigest()
        if expected_sha256 is not None and actual != _require_sha256(expected_sha256):
            raise ValueError("v6 artifact SHA-256 mismatch")
        payload = pickle.loads(encoded)
        if not isinstance(payload, dict):
            raise TypeError("v6 artifact envelope must be a mapping")
        if payload.get("artifact_schema") != HU_M43_V6_ARTIFACT_SCHEMA:
            raise ValueError("unsupported v6 artifact schema")
        if payload.get("model_schema") != HU_M43_V6_MODEL_SCHEMA:
            raise ValueError("unsupported v6 model schema")
        if payload.get("proposal_schema") != HU_M43_V6_PROPOSAL_SCHEMA:
            raise ValueError("unsupported v6 proposal schema")
        if payload.get("safety_feature_schema") != HU_M43_V6_SAFETY_FEATURE_SCHEMA:
            raise ValueError("unsupported v6 safety feature schema")
        if set(payload) != {
            "artifact_schema",
            "model_schema",
            "proposal_schema",
            "safety_feature_schema",
            "model",
        }:
            raise ValueError("v6 artifact envelope keys changed")
        model = payload.get("model")
        if not isinstance(model, cls):
            raise TypeError("v6 artifact model type mismatch")
        model.__post_init__()
        return model


def load_hu_m43_joint_model_v6(
    path: str | Path, *, expected_sha256: str | None = None
) -> HuM43JointModelV6:
    return HuM43JointModelV6.load(path, expected_sha256=expected_sha256)


def _build_safety_features(
    *,
    proposal_index: int,
    baseline_index: int,
    delta: np.ndarray,
    positive: np.ndarray,
    p95: np.ndarray,
    p99: np.ndarray,
    maximum: np.ndarray,
    upper95: np.ndarray,
    upper99: np.ndarray,
    uppermax: np.ndarray,
    disagreement: np.ndarray,
    risk_mask: np.ndarray,
) -> np.ndarray:
    nonbaseline = np.asarray(
        [index for index in range(delta.size) if index != baseline_index],
        dtype=np.int32,
    )
    values = delta[nonbaseline]
    proposal = int(proposal_index)
    other = values[nonbaseline != proposal]
    next_margin = float(delta[proposal] - np.max(other)) if other.size else 0.0
    rank_fraction = float(np.mean(values < delta[proposal])) if values.size > 1 else 1.0
    pool_count = int(np.sum(risk_mask[nonbaseline]))
    result = np.asarray(
        [
            delta[proposal],
            positive[proposal],
            p95[proposal],
            p99[proposal],
            maximum[proposal],
            upper95[proposal],
            upper99[proposal],
            uppermax[proposal],
            disagreement[proposal],
            next_margin,
            rank_fraction,
            float(np.mean(values)),
            float(np.std(values)),
            float(np.max(values)),
            float(np.min(values)),
            float(nonbaseline.size),
            float(pool_count),
            float(pool_count / nonbaseline.size),
            float(bool(risk_mask[proposal])),
        ],
        dtype=np.float32,
    )
    if result.shape != (HU_M43_V6_SAFETY_FEATURE_DIM,):
        raise AssertionError("v6 safety feature shape changed")
    return result


def _canonical_delta_argmax(
    values: np.ndarray,
    *,
    candidate_indices: np.ndarray,
    action_keys: tuple[ActionKey, ...],
) -> int:
    candidates = [int(index) for index in np.asarray(candidate_indices).reshape(-1)]
    if not candidates:
        raise ValueError("v6 proposal pool is empty")
    best = max(float(values[index]) for index in candidates)
    tied = [index for index in candidates if float(values[index]) == best]
    return min(tied, key=lambda index: action_keys[index].sort_key())


def _runtime_policy_projection(sample: Mapping[str, Any]) -> dict[str, Any]:
    raw_observation = sample.get("policy_observation")
    if not isinstance(raw_observation, Mapping):
        raise ValueError("v6 runtime requires policy_observation")
    observation = ActorObservation.from_dict(raw_observation)
    if observation.street != "T1":
        raise ValueError("v6 runtime accepts only T1 observations")
    canonical_observation = observation.to_dict()
    if dict(raw_observation) != canonical_observation:
        raise ValueError("v6 policy_observation is non-canonical or has extra fields")

    legal = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    legal_by_key = index_actions_by_key(legal)
    raw_actions = sample.get("actions")
    if isinstance(raw_actions, (str, bytes)) or not isinstance(raw_actions, Sequence):
        raise ValueError("v6 runtime requires an ordered legal action sequence")
    ordered_actions = []
    seen: set[ActionKey] = set()
    for index, raw_action in enumerate(raw_actions):
        if not isinstance(raw_action, Mapping):
            raise ValueError(f"v6 action {index} must be a mapping")
        key = action_key_from_payload(raw_action)
        declared = raw_action.get("action_key")
        if declared is not None and declared != key.to_token():
            raise ValueError(f"v6 action {index} declared ActionKey disagrees")
        if key in seen:
            raise ValueError("v6 action list contains a duplicate semantic action")
        try:
            regenerated = legal[legal_by_key[key]]
        except KeyError as error:
            raise ValueError("v6 action list contains an illegal action") from error
        seen.add(key)
        ordered_actions.append(regenerated)
    if seen != set(legal_by_key) or len(ordered_actions) != len(legal):
        raise ValueError("v6 action list is not the complete legal action set")

    rebuilt = hu_policy_sample(
        observation.hero_board,
        observation.dealt_cards,
        ordered_actions,
        opponent_board=observation.opponent_public_board,
        dead_cards=observation.legacy_dead_cards(),
        seat=observation.seat,
        to_act_order=observation.to_act_order,
    )
    for index, (raw_action, canonical_action) in enumerate(
        zip(raw_actions, rebuilt["actions"], strict=True)
    ):
        if raw_action.get("next_board") != canonical_action.get("next_board"):
            raise ValueError(f"v6 action {index} next_board disagrees with legality")
        if action_key(ordered_actions[index]) != action_key_from_payload(canonical_action):
            raise AssertionError("v6 regenerated action mapping changed")

    for key in ("board", "opponent_board", "dead_cards", "dealt", "seat", "to_act_order"):
        if sample.get(key) != rebuilt.get(key):
            raise ValueError(f"v6 sample {key} disagrees with policy_observation")
    rebuilt["policy_observation"] = canonical_observation
    if "baseline_action_row_index" in sample:
        rebuilt["baseline_action_row_index"] = sample["baseline_action_row_index"]
    if "baseline_action_key" in sample:
        rebuilt["baseline_action_key"] = sample["baseline_action_key"]
    return rebuilt


def _action_count(sample: Mapping[str, Any]) -> int:
    actions = sample.get("actions", ())
    if isinstance(actions, (str, bytes)) or len(actions) < 2:
        raise ValueError("v6 requires at least two legal actions")
    return len(actions)


def _action_keys(sample: Mapping[str, Any]) -> tuple[ActionKey, ...]:
    return tuple(action_key_from_payload(action) for action in sample["actions"])


def _resolve_baseline_index(
    sample: Mapping[str, Any], baseline_index: int | None, action_count: int
) -> int:
    declared = sample.get("baseline_action_row_index")
    if baseline_index is not None and (
        isinstance(baseline_index, (bool, np.bool_))
        or not isinstance(baseline_index, (int, np.integer))
    ):
        raise TypeError("v6 explicit baseline index must be an integer")
    if baseline_index is not None and declared is not None:
        if isinstance(declared, (bool, np.bool_)) or not isinstance(
            declared, (int, np.integer)
        ):
            raise TypeError("v6 declared baseline index must be an integer")
        if int(declared) != int(baseline_index):
            raise ValueError("v6 explicit baseline index disagrees with sample")
    value = declared if baseline_index is None else baseline_index
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("v6 baseline index must be an integer")
    baseline = int(value)
    if not 0 <= baseline < action_count:
        raise IndexError("v6 baseline index is outside actions")
    return baseline


def _positive_probability(estimator: Any, features: np.ndarray) -> float:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.shape != (1, HU_M43_V6_SAFETY_FEATURE_DIM):
        raise ValueError("v6 safety feature shape is invalid")
    if hasattr(estimator, "predict_proba"):
        raw = np.asarray(estimator.predict_proba(matrix), dtype=np.float64)
        if raw.shape != (1, 2):
            raise ValueError("v6 safety predict_proba shape is invalid")
        probability = float(raw[0, 1])
    elif hasattr(estimator, "predict_probability"):
        probability = float(estimator.predict_probability(matrix.reshape(-1)))
    else:
        raise TypeError("v6 safety estimator has no probability method")
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("v6 safety probability is invalid")
    return probability


def _validate_folds(folds: Sequence[PairedDeltaRiskFoldEstimator]) -> None:
    if len(folds) != HU_M43_V6_REQUIRED_FOLDS:
        raise ValueError("v6 requires exactly five paired fold estimators")
    indices = []
    for fold in folds:
        if not isinstance(fold, PairedDeltaRiskFoldEstimator):
            raise TypeError("v6 fold has incompatible type")
        indices.append(int(fold.fold_index))
    if sorted(indices) != list(range(HU_M43_V6_REQUIRED_FOLDS)):
        raise ValueError("v6 fold indices must be exactly 0..4")


def _require_sha256(value: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise ValueError("expected SHA-256 is invalid")
    return normalized


__all__ = [
    "HU_M43_V6_ACTION_SCORE_MODE",
    "HU_M43_V6_ARTIFACT_SCHEMA",
    "HU_M43_V6_MODEL_SCHEMA",
    "HU_M43_V6_PROPOSAL_SCHEMA",
    "HU_M43_V6_SAFETY_FEATURE_DIM",
    "HU_M43_V6_SAFETY_FEATURE_NAMES",
    "HU_M43_V6_SAFETY_FEATURE_SCHEMA",
    "HuM43JointModelV6",
    "V6ActionPredictions",
    "V6RuntimeDecision",
    "load_hu_m43_joint_model_v6",
]
