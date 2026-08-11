"""Opt-in M4 T1 second-seat selective-override policy.

The wrapper is deliberately a composition policy.  The frozen baseline is
asked first for every decision and remains the returned action unless a T1
second-seat candidate passes all action-value, legality, and safety checks.
Only :class:`~ofc_regular.hu_infoset.ActorObservation` crosses the policy
boundary; replay truth and the opponent's private discards have no API here.
"""

from __future__ import annotations

import json
import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .action_key import (
    ACTION_KEY_SCHEMA,
    action_key,
    canonical_argmax_index,
    index_actions_by_key,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from .action_space import Action, generate_turn_actions
from .hu_infoset import ActorObservation
from .hu_m43_attempt08_distilled_model import (
    HU_M43_ATTEMPT08_DISTILLED_MODEL_SCHEMA,
    is_bound_attempt08_distilled_model,
)
from .hu_m43_attempt10_distilled_model import (
    HU_M43_ATTEMPT10_DISTILLED_MODEL_SCHEMA,
    is_bound_attempt10_distilled_model,
)
from .hu_m43_attempt11_distilled_model import (
    HU_M43_ATTEMPT11_DISTILLED_MODEL_SCHEMA,
    is_bound_attempt11_distilled_model,
)
from .hu_m43_attempt12_distilled_model import (
    HU_M43_ATTEMPT12_DISTILLED_MODEL_SCHEMA,
    is_bound_attempt12_distilled_model,
)
from .hu_m43_attempt13_distilled_model import (
    HU_M43_ATTEMPT13_DISTILLED_MODEL_SCHEMA,
    is_bound_attempt13_distilled_model,
)
from .hu_m43_joint_model_loader import load_hu_m43_joint_action_model
from .hu_turn3_model import (
    hu_policy_sample,
    load_hu_action_value_model,
    sample_to_matrix,
)


HU_M4_T1_POLICY_SCHEMA = "hu_m4_t1_selective_override_policy_v1"
HU_M4_T1_DECISION_SCHEMA = "hu_m4_t1_selective_override_decision_v1"
HU_M4_T1_SAFETY_MODEL_SCHEMA = "hu_m4_t1_safety_model_v1"
HU_M4_T1_SAFETY_ARTIFACT_SCHEMA = "hu_m4_t1_safety_model_pickle_v1"


@dataclass(frozen=True)
class HuM4T1SafetyModel:
    """Versioned pickle artifact around a binary safety estimator.

    The estimator may expose ``predict_proba`` or ``decision_function``.  The
    positive-class probability means "safe to override".  ``feature_dim`` is
    checked at runtime so a stale selector fails closed instead of silently
    reinterpreting a newer feature layout.
    """

    estimator: Any
    feature_dim: int
    model_id: str = "hu-m4-t1-safety"
    schema: str = HU_M4_T1_SAFETY_MODEL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != HU_M4_T1_SAFETY_MODEL_SCHEMA:
            raise ValueError(f"unsupported M4 T1 safety schema: {self.schema!r}")
        if not isinstance(self.feature_dim, int) or isinstance(self.feature_dim, bool):
            raise TypeError("feature_dim must be an integer")
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("model_id must be a non-empty string")

    def predict_probability(self, features: np.ndarray) -> float:
        row = np.asarray(features, dtype=np.float32).reshape(1, -1)
        if row.shape[1] != self.feature_dim:
            raise ValueError(
                f"safety feature dimension mismatch: {row.shape[1]} != {self.feature_dim}"
            )
        if not np.isfinite(row).all():
            raise ValueError("safety features must be finite")
        if hasattr(self.estimator, "predict_proba"):
            probabilities = np.asarray(
                self.estimator.predict_proba(row), dtype=np.float64
            )
            if probabilities.ndim != 2 or probabilities.shape[0] != 1:
                raise ValueError("invalid predict_proba output shape")
            classes = list(getattr(self.estimator, "classes_", ()))
            if 1 in classes:
                probability = float(probabilities[0, classes.index(1)])
            elif classes == [0] and probabilities.shape[1] == 1:
                probability = 0.0
            else:
                raise ValueError("safety estimator has no positive class")
        elif hasattr(self.estimator, "decision_function"):
            decision = np.asarray(
                self.estimator.decision_function(row), dtype=np.float64
            ).reshape(-1)
            if decision.size != 1 or not np.isfinite(decision[0]):
                raise ValueError("invalid decision_function output")
            probability = _sigmoid(float(decision[0]))
        else:
            raise TypeError("safety estimator lacks predict_proba/decision_function")
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("safety probability must be finite and in [0, 1]")
        return probability

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": HU_M4_T1_SAFETY_ARTIFACT_SCHEMA,
            "model": self,
        }
        with output.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "HuM4T1SafetyModel":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict):
            raise TypeError("M4 T1 safety artifact must be a mapping")
        if payload.get("schema") != HU_M4_T1_SAFETY_ARTIFACT_SCHEMA:
            raise ValueError("unsupported M4 T1 safety artifact schema")
        model = payload.get("model")
        if not isinstance(model, cls):
            raise TypeError(f"expected {cls.__name__}, got {type(model).__name__}")
        # Re-run version/dimension validation even for an old pickle whose
        # __post_init__ was not invoked while unpickling.
        model.__post_init__()
        return model


def load_hu_m4_t1_safety_model(path: str | Path) -> HuM4T1SafetyModel:
    return HuM4T1SafetyModel.load(path)


def build_hu_m4_t1_safety_features(
    sample: dict[str, Any],
    *,
    candidate_index: int,
    baseline_index: int,
    predicted_values: Sequence[float] | np.ndarray,
    seat: str,
) -> np.ndarray:
    """Encode candidate/baseline/delta plus scores and seat one-hot.

    Layout ``v1`` is::

        candidate action features
        baseline action features
        candidate - baseline feature delta
        candidate score, baseline score, score delta, candidate-vs-next margin
        seat_is_first, seat_is_second
    """

    matrix, _targets = sample_to_matrix(sample)
    values = np.asarray(predicted_values, dtype=np.float64)
    if values.ndim != 1 or values.shape[0] != matrix.shape[0]:
        raise ValueError("predicted value/action length mismatch")
    if not np.isfinite(values).all() or not np.isfinite(matrix).all():
        raise ValueError("safety feature inputs must be finite")
    if not 0 <= candidate_index < matrix.shape[0]:
        raise IndexError("candidate index is outside the legal action list")
    if not 0 <= baseline_index < matrix.shape[0]:
        raise IndexError("baseline index is outside the legal action list")
    if seat not in {"first", "second"}:
        raise ValueError(f"invalid seat: {seat!r}")

    candidate = matrix[candidate_index].astype(np.float32, copy=False)
    baseline = matrix[baseline_index].astype(np.float32, copy=False)
    candidate_score = float(values[candidate_index])
    baseline_score = float(values[baseline_index])
    other_scores = np.delete(values, candidate_index)
    margin = (
        candidate_score - float(np.max(other_scores))
        if other_scores.size
        else 0.0
    )
    scalars = np.asarray(
        [
            candidate_score,
            baseline_score,
            candidate_score - baseline_score,
            margin,
            float(seat == "first"),
            float(seat == "second"),
        ],
        dtype=np.float32,
    )
    features = np.concatenate([candidate, baseline, candidate - baseline, scalars])
    if not np.isfinite(features).all():
        raise ValueError("constructed safety features must be finite")
    return features.astype(np.float32, copy=False)


class HuM4T1SelectiveOverridePolicy:
    """Composition wrapper that can override only second-seat T1 decisions."""

    def __init__(
        self,
        baseline_policy: object,
        *,
        action_value_model: object | None,
        safety_model: object | None,
        safety_probability_threshold: float,
        enabled: bool = True,
        allowed_seats: Sequence[str] = ("second",),
        decision_log: list[dict[str, Any]] | None = None,
        decision_log_path: str | Path | None = None,
        policy_id: str = "hu-m4-t1-second-selective-v1",
        model_load_failures: Sequence[str] = (),
        runtime_binding_verified: bool = False,
    ) -> None:
        if not hasattr(baseline_policy, "choose_action_observation"):
            raise TypeError("baseline_policy must implement choose_action_observation")
        threshold = float(safety_probability_threshold)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("safety_probability_threshold must be in [0, 1]")
        normalized_seats = tuple(str(seat) for seat in allowed_seats)
        if len(set(normalized_seats)) != len(normalized_seats):
            raise ValueError("allowed_seats must not contain duplicates")
        # M4 promotion is explicitly scoped to the second seat.  Rejecting a
        # first-seat opt-in prevents a config typo from broadening that scope.
        if any(seat != "second" for seat in normalized_seats):
            raise ValueError("M4 T1 runtime may allow only the second seat")
        self.baseline_policy = baseline_policy
        self.action_value_model = action_value_model
        self.safety_model = safety_model
        self.safety_probability_threshold = threshold
        self.enabled = bool(enabled)
        self.allowed_seats = normalized_seats
        self.decision_log = decision_log
        self.decision_log_path = Path(decision_log_path) if decision_log_path else None
        self.policy_id = str(policy_id)
        self.model_load_failures = tuple(str(error) for error in model_load_failures)
        self.runtime_binding_verified = bool(runtime_binding_verified)

    @classmethod
    def from_model_paths(
        cls,
        baseline_policy: object,
        *,
        action_value_model_path: str | Path,
        safety_model_path: str | Path,
        safety_probability_threshold: float,
        expected_joint_artifact_sha256: str | None = None,
        freeze_manifest_path: str | Path | None = None,
        training_manifest_path: str | Path | None = None,
        runtime_source_manifest_path: str | Path | None = None,
        runtime_source_root: str | Path | None = None,
        runtime_dependency_root: str | Path | None = None,
        **kwargs: Any,
    ) -> "HuM4T1SelectiveOverridePolicy":
        failures: list[str] = []
        binding_requested = any(
            value is not None
            for value in (
                expected_joint_artifact_sha256,
                freeze_manifest_path,
                training_manifest_path,
                runtime_source_manifest_path,
                runtime_source_root,
                runtime_dependency_root,
            )
        )
        binding_verified = False
        if binding_requested:
            if Path(action_value_model_path).resolve() != Path(
                safety_model_path
            ).resolve():
                action_value_model = None
                safety_model = None
                failures.append("frozen_joint_artifact_path_mismatch")
            else:
                try:
                    joint_model = load_hu_m43_joint_action_model(
                        action_value_model_path,
                        expected_sha256=expected_joint_artifact_sha256,
                        freeze_manifest=freeze_manifest_path,
                        training_manifest_path=training_manifest_path,
                        runtime_source_manifest_path=runtime_source_manifest_path,
                        runtime_source_root=runtime_source_root,
                        runtime_dependency_root=runtime_dependency_root,
                    )
                    action_value_model = joint_model
                    safety_model = joint_model
                    binding_verified = True
                except Exception as exc:
                    action_value_model = None
                    safety_model = None
                    failures.append(
                        "frozen_joint_artifact_binding_failed:"
                        f"{type(exc).__name__}"
                    )
        else:
            try:
                action_value_model = load_hu_action_value_model(action_value_model_path)
            except Exception as legacy_exc:  # A bad optional artifact must not take down play.
                try:
                    action_value_model = load_hu_m43_joint_action_model(
                        action_value_model_path
                    )
                except Exception as joint_exc:
                    action_value_model = None
                    failures.append(
                        "action_value_model_load_failed:"
                        f"{type(legacy_exc).__name__}/{type(joint_exc).__name__}"
                    )
            try:
                safety_model = load_hu_m4_t1_safety_model(safety_model_path)
            except Exception as legacy_exc:
                try:
                    safety_model = load_hu_m43_joint_action_model(safety_model_path)
                except Exception as joint_exc:
                    safety_model = None
                    failures.append(
                        "safety_model_load_failed:"
                        f"{type(legacy_exc).__name__}/{type(joint_exc).__name__}"
                    )
        artifact_threshold = getattr(safety_model, "safety_threshold", None)
        if artifact_threshold is not None and not math.isclose(
            float(artifact_threshold),
            float(safety_probability_threshold),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            safety_model = None
            failures.append("safety_threshold_mismatch")
        action_is_joint = hasattr(action_value_model, "predict_heads_sample")
        safety_is_joint = hasattr(safety_model, "predict_heads_sample")
        if action_is_joint or safety_is_joint:
            same_joint_artifact = (
                action_is_joint
                and safety_is_joint
                and Path(action_value_model_path).resolve()
                == Path(safety_model_path).resolve()
            )
            if not same_joint_artifact:
                safety_model = None
                failures.append("joint_action_safety_artifact_mismatch")
        return cls(
            baseline_policy,
            action_value_model=action_value_model,
            safety_model=safety_model,
            safety_probability_threshold=safety_probability_threshold,
            model_load_failures=failures,
            runtime_binding_verified=binding_verified,
            **kwargs,
        )

    def choose_action_observation(
        self,
        observation: ActorObservation,
        *,
        hand_id: str | int | None = None,
        game_id: str | int | None = None,
        decision_seed: int | None = None,
    ) -> Action:
        if not isinstance(observation, ActorObservation):
            raise TypeError("M4 policy requires an ActorObservation")
        started_at = time.perf_counter()
        baseline_action = self.baseline_policy.choose_action_observation(
            observation,
            hand_id=hand_id,
            game_id=game_id,
            decision_seed=decision_seed,
        )

        final_action = baseline_action
        candidate_action: Action | None = None
        actions: list[Action] = []
        baseline_index: int | None = None
        candidate_index: int | None = None
        final_index: int | None = None
        candidate_score: float | None = None
        baseline_score: float | None = None
        predicted_delta: float | None = None
        candidate_margin: float | None = None
        safety_probability: float | None = None
        override_fired = False
        nonfire_reason = ""

        if observation.street != "T1":
            nonfire_reason = "street_not_t1"
        elif observation.seat == "first":
            nonfire_reason = "first_seat_delegated"
        elif observation.seat not in self.allowed_seats:
            nonfire_reason = "seat_not_allowed"
        elif not self.enabled:
            nonfire_reason = "override_disabled"
        elif self.model_load_failures:
            nonfire_reason = "model_load_failed"
        elif self.action_value_model is None:
            nonfire_reason = "action_value_model_unavailable"
        elif self.safety_model is None:
            nonfire_reason = "safety_model_unavailable"
        elif (
            (
                getattr(self.action_value_model, "schema", None)
                == HU_M43_ATTEMPT08_DISTILLED_MODEL_SCHEMA
                and not (
                    self.runtime_binding_verified
                    and is_bound_attempt08_distilled_model(self.action_value_model)
                    and self.safety_model is self.action_value_model
                )
            )
            or (
                getattr(self.action_value_model, "schema", None)
                == HU_M43_ATTEMPT10_DISTILLED_MODEL_SCHEMA
                and not (
                    self.runtime_binding_verified
                    and is_bound_attempt10_distilled_model(self.action_value_model)
                    and self.safety_model is self.action_value_model
                )
            )
            or (
                getattr(self.action_value_model, "schema", None)
                == HU_M43_ATTEMPT11_DISTILLED_MODEL_SCHEMA
                and not (
                    self.runtime_binding_verified
                    and is_bound_attempt11_distilled_model(self.action_value_model)
                    and self.safety_model is self.action_value_model
                )
            )
            or (
                getattr(self.action_value_model, "schema", None)
                == HU_M43_ATTEMPT12_DISTILLED_MODEL_SCHEMA
                and not (
                    self.runtime_binding_verified
                    and is_bound_attempt12_distilled_model(self.action_value_model)
                    and self.safety_model is self.action_value_model
                )
            )
            or (
                getattr(self.action_value_model, "schema", None)
                == HU_M43_ATTEMPT13_DISTILLED_MODEL_SCHEMA
                and not (
                    self.runtime_binding_verified
                    and is_bound_attempt13_distilled_model(self.action_value_model)
                    and self.safety_model is self.action_value_model
                )
            )
        ):
            nonfire_reason = "runtime_binding_unverified"
        elif getattr(self.safety_model, "safety_enabled", True) is not True:
            nonfire_reason = "artifact_safety_disabled"
        else:
            try:
                actions = generate_turn_actions(
                    observation.hero_board, observation.dealt_cards
                )
                if not actions:
                    raise ValueError("no legal T1 actions")
                baseline_key = action_key(baseline_action)
                baseline_index = index_actions_by_key(actions).get(baseline_key)
                if baseline_index is None:
                    raise LookupError("baseline action is not legal at observation")

                sample = hu_policy_sample(
                    observation.hero_board,
                    observation.dealt_cards,
                    actions,
                    opponent_board=observation.opponent_public_board,
                    dead_cards=observation.legacy_dead_cards(),
                    seat=observation.seat,
                    to_act_order=observation.to_act_order,
                )
                # v6 rebuilds its complete legal sample from this typed public
                # information-set boundary. Older model schemas ignore it.
                sample["policy_observation"] = observation.to_dict()
                if hasattr(
                    self.action_value_model, "predict_sample_with_baseline"
                ):
                    predicted = self.action_value_model.predict_sample_with_baseline(
                        sample, baseline_index=baseline_index
                    )
                else:
                    predicted = self.action_value_model.predict_sample(sample)
                raw_values = np.asarray(predicted, dtype=np.float64)
                if raw_values.ndim == 2 and raw_values.shape[1] == 1:
                    raw_values = raw_values[:, 0]
                if raw_values.ndim != 1 or raw_values.shape[0] != len(actions):
                    raise ValueError("action-value prediction shape mismatch")
                if not np.isfinite(raw_values).all():
                    raise ValueError("action-value predictions are non-finite")

                candidate_index = canonical_argmax_index(raw_values, actions)
                candidate_action = actions[candidate_index]
                candidate_score = float(raw_values[candidate_index])
                baseline_score = float(raw_values[baseline_index])
                predicted_delta = candidate_score - baseline_score
                other = np.delete(raw_values, candidate_index)
                candidate_margin = (
                    candidate_score - float(np.max(other)) if other.size else 0.0
                )
                if candidate_index == baseline_index:
                    nonfire_reason = "same_as_baseline"
                else:
                    if hasattr(self.safety_model, "predict_safety_probability"):
                        safety_probability = float(
                            self.safety_model.predict_safety_probability(
                                sample,
                                candidate_index=candidate_index,
                                baseline_index=baseline_index,
                            )
                        )
                    else:
                        safety_features = build_hu_m4_t1_safety_features(
                            sample,
                            candidate_index=candidate_index,
                            baseline_index=baseline_index,
                            predicted_values=raw_values,
                            seat=observation.seat,
                        )
                        safety_probability = _predict_safety_probability(
                            self.safety_model, safety_features
                        )
                    if (
                        not math.isfinite(safety_probability)
                        or not 0.0 <= safety_probability <= 1.0
                    ):
                        raise ValueError(
                            "safety probability must be finite and in [0, 1]"
                        )
                    if safety_probability < self.safety_probability_threshold:
                        nonfire_reason = "below_safety_probability_threshold"
                    else:
                        final_action = candidate_action
                        final_index = candidate_index
                        override_fired = True
            except Exception as exc:
                # Model/feature/index/legality failures are all fail-closed.
                final_action = baseline_action
                override_fired = False
                nonfire_reason = _failure_reason(exc)

        if actions and final_index is None and baseline_index is not None:
            final_index = baseline_index
        if not nonfire_reason and not override_fired:
            nonfire_reason = "fallback_to_baseline"

        record = self._decision_record(
            observation=observation,
            actions=actions,
            baseline_action=baseline_action,
            candidate_action=candidate_action,
            final_action=final_action,
            baseline_index=baseline_index,
            candidate_index=candidate_index,
            final_index=final_index,
            candidate_score=candidate_score,
            baseline_score=baseline_score,
            predicted_delta=predicted_delta,
            candidate_margin=candidate_margin,
            safety_probability=safety_probability,
            override_fired=override_fired,
            nonfire_reason=nonfire_reason,
            hand_id=hand_id,
            game_id=game_id,
            decision_seed=decision_seed,
            latency_ms=(time.perf_counter() - started_at) * 1000.0,
        )
        self._log_decision(record)
        return final_action

    def _decision_record(
        self,
        *,
        observation: ActorObservation,
        actions: Sequence[Action],
        baseline_action: Any,
        candidate_action: Action | None,
        final_action: Any,
        baseline_index: int | None,
        candidate_index: int | None,
        final_index: int | None,
        candidate_score: float | None,
        baseline_score: float | None,
        predicted_delta: float | None,
        candidate_margin: float | None,
        safety_probability: float | None,
        override_fired: bool,
        nonfire_reason: str,
        hand_id: str | int | None,
        game_id: str | int | None,
        decision_seed: int | None,
        latency_ms: float,
    ) -> dict[str, Any]:
        return {
            "schema": HU_M4_T1_DECISION_SCHEMA,
            "policy_schema": HU_M4_T1_POLICY_SCHEMA,
            "policy_id": self.policy_id,
            "observation_fingerprint": observation.fingerprint(),
            # This is the only card-bearing log field.  It contains public
            # boards, this actor's deal/discards, and no opponent private cards.
            "policy_observation": observation.to_dict(),
            "visibility_model": "actor_observation_v1",
            "discard_visibility": "own_private_only",
            "hand_id": hand_id,
            "game_id": game_id,
            "decision_seed": decision_seed,
            "street": observation.street,
            "seat": observation.seat,
            "enabled": self.enabled,
            "allowed_seats": list(self.allowed_seats),
            "action_key_schema": ACTION_KEY_SCHEMA,
            "legal_action_count": len(actions),
            "legal_action_set_digest": legal_action_set_digest(actions) if actions else None,
            "legal_action_order_digest": (
                ordered_action_mapping_digest(actions) if actions else None
            ),
            "baseline_action_index": baseline_index,
            "candidate_action_index": candidate_index,
            "final_action_index": final_index,
            "baseline_action_key": _safe_action_key(baseline_action),
            "candidate_action_key": _safe_action_key(candidate_action),
            "final_action_key": _safe_action_key(final_action),
            "candidate_score": candidate_score,
            "baseline_score": baseline_score,
            "predicted_delta": predicted_delta,
            "candidate_margin": candidate_margin,
            "safety_probability": safety_probability,
            "safety_probability_threshold": self.safety_probability_threshold,
            "override_fired": override_fired,
            "nonfire_reason": nonfire_reason,
            "model_load_failures": list(self.model_load_failures),
            "runtime_binding_verified": self.runtime_binding_verified,
            "runtime_latency_ms": latency_ms,
        }

    def _log_decision(self, record: dict[str, Any]) -> None:
        if self.decision_log is not None:
            self.decision_log.append(record)
        if self.decision_log_path is None:
            return
        self.decision_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.decision_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def _predict_safety_probability(model: object, features: np.ndarray) -> float:
    if hasattr(model, "predict_probability"):
        probability = float(model.predict_probability(features))
    elif hasattr(model, "predict_proba"):
        output = np.asarray(
            model.predict_proba(features.reshape(1, -1)), dtype=np.float64
        )
        if output.ndim != 2 or output.shape[0] != 1 or output.shape[1] < 1:
            raise ValueError("invalid safety predict_proba output shape")
        classes = list(getattr(model, "classes_", ()))
        if 1 in classes:
            probability = float(output[0, classes.index(1)])
        elif classes == [0] and output.shape[1] == 1:
            probability = 0.0
        else:
            raise ValueError("safety estimator has no positive class")
    elif hasattr(model, "decision_function"):
        output = np.asarray(
            model.decision_function(features.reshape(1, -1)), dtype=np.float64
        ).reshape(-1)
        if output.size != 1:
            raise ValueError("invalid safety decision_function output shape")
        probability = _sigmoid(float(output[0]))
    else:
        raise TypeError("safety model lacks a probability interface")
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("safety probability must be finite and in [0, 1]")
    return probability


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _failure_reason(error: Exception) -> str:
    if isinstance(error, LookupError):
        return "illegal_baseline_action"
    message = str(error).lower()
    if "non-finite" in message or "finite" in message:
        return "nonfinite_prediction"
    if "action-value" in message or "predict" in message:
        return "action_value_model_failed"
    if "safety" in message or "feature" in message:
        return "safety_model_failed"
    if "legal" in message or "index" in message or "shape" in message:
        return "illegal_candidate"
    return "model_or_feature_failure"


def _safe_action_key(action: Any) -> str | None:
    if action is None:
        return None
    try:
        return action_key(action).to_token()
    except Exception:
        return None


__all__ = [
    "HU_M4_T1_DECISION_SCHEMA",
    "HU_M4_T1_POLICY_SCHEMA",
    "HU_M4_T1_SAFETY_ARTIFACT_SCHEMA",
    "HU_M4_T1_SAFETY_MODEL_SCHEMA",
    "HuM4T1SafetyModel",
    "HuM4T1SelectiveOverridePolicy",
    "build_hu_m4_t1_safety_features",
    "load_hu_m4_t1_safety_model",
]
