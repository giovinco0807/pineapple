"""Fail-closed distilled runtime selector for M4.3 Attempt11.

The expensive Attempt11 search remains a development teacher.  This module
distils that teacher into a five-fold public-information-set model while
retaining the frozen Attempt05 Lambda model solely as a variable, at-most
twelve candidate generator. Runtime never consumes teacher values, teacher lower bounds,
opponent-private discards, or an opponent-profile feature.

The model returns a deliberately simple action score vector: the exact
baseline scores zero, the single eligible proposal scores one, and every other
action scores minus one.  That lets the existing M4 wrapper preserve its
baseline-first, ActionKey-stable, fail-closed behavior.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import os
import pickle
import sys
import uuid
import warnings
from dataclasses import dataclass, field, fields as dataclass_fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .action_key import ActionKey, action_key, action_key_from_payload, index_actions_by_key
from .action_space import Action, generate_turn_actions
from .hu_infoset import ActorObservation
from .hu_m43_attempt05_model import (
    HuM43Attempt05Model,
    _runtime_policy_projection,
)
from .hu_m43_attempt11_teacher import (
    ATTEMPT11_CANDIDATE_MAX,
    ATTEMPT11_FROZEN_MODEL_ID,
    ATTEMPT11_FROZEN_MODEL_SHA256,
    Attempt11RankScores,
    FrozenAttempt11LambdaRanker,
)
from .hu_m43_attempt11_distilled_runtime import (
    ATTEMPT11_BOUND_EXECUTION_MODULES,
    FrozenExecutionModulesAttestation,
)
from .hu_m4_joint_model import build_paired_action_features_matrix
from .hu_turn3_model import HU_FEATURE_DIM, sample_to_matrix


HU_M43_ATTEMPT11_DISTILLED_MODEL_SCHEMA = (
    "hu_m43_attempt11_t1_second_distilled_selector_v1"
)
HU_M43_ATTEMPT11_DISTILLED_ARTIFACT_SCHEMA = (
    "hu_m43_attempt11_t1_second_distilled_pickle_v1"
)
HU_M43_ATTEMPT11_DISTILLED_FEATURE_SCHEMA = (
    "hu_m43_attempt11_lambda_variable_top12_public_infoset_features_v1"
)
HU_M43_ATTEMPT11_DISTILLED_HEAD_SCHEMA = (
    "hu_m43_attempt11_policy_delta_safe_tail_heads_v1"
)
HU_M43_ATTEMPT11_DISTILLED_ACTION_SCORE_MODE = (
    "attempt11_lambda_variable_top12_distilled_safe_selector_v1"
)
HU_M43_ATTEMPT11_DISTILLED_CONFORMAL_SCHEMA = (
    "hu_m43_attempt11_identity_oof_upper_residual_conformal_v1"
)
HU_M43_ATTEMPT11_DISTILLED_RUNTIME_AUTHORIZATION = "T1-second-only"
HU_M43_ATTEMPT11_DISTILLED_EXTRA_FEATURES = 10
HU_M43_ATTEMPT11_DISTILLED_FEATURE_DIM = (
    4 * HU_FEATURE_DIM + HU_M43_ATTEMPT11_DISTILLED_EXTRA_FEATURES
)
HU_M43_ATTEMPT11_DISTILLED_REQUIRED_FOLDS = 5
HU_M43_ATTEMPT11_DISTILLED_TAIL_LIMITS = (25.0, 40.0, 50.0)
@dataclass(frozen=True)
class Attempt11DistilledFoldOutput:
    rank_score: np.ndarray
    predicted_delta: np.ndarray
    safe_probability: np.ndarray
    downside_p95: np.ndarray
    downside_p99: np.ndarray
    downside_max: np.ndarray


@dataclass(frozen=True)
class Attempt11DistilledFoldPredictor:
    """One identity-grouped fold with separate policy, EV, and safety heads."""

    ranker: Any
    delta_head: Any
    safe_head: Any
    tail_p95_head: Any
    tail_p99_head: Any
    tail_max_head: Any
    fold_index: int
    feature_dim: int = HU_M43_ATTEMPT11_DISTILLED_FEATURE_DIM

    def __post_init__(self) -> None:
        if self.fold_index < 0:
            raise ValueError("Attempt11 distilled fold_index must be non-negative")
        if self.feature_dim != HU_M43_ATTEMPT11_DISTILLED_FEATURE_DIM:
            raise ValueError("Attempt11 distilled feature dimension changed")
        for name in (
            "ranker",
            "delta_head",
            "safe_head",
            "tail_p95_head",
            "tail_p99_head",
            "tail_max_head",
        ):
            if getattr(self, name) is None:
                raise ValueError(f"Attempt11 distilled {name} must not be None")

    def predict(self, features: np.ndarray) -> Attempt11DistilledFoldOutput:
        matrix = np.asarray(features, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.feature_dim:
            raise ValueError("Attempt11 distilled fold feature shape changed")
        rank = _regression(self.ranker, matrix, "rank")
        delta = _regression(self.delta_head, matrix, "delta")
        safe = _probability(self.safe_head, matrix)
        tails = np.vstack(
            (
                np.maximum(0.0, _regression(self.tail_p95_head, matrix, "p95")),
                np.maximum(0.0, _regression(self.tail_p99_head, matrix, "p99")),
                np.maximum(0.0, _regression(self.tail_max_head, matrix, "max")),
            )
        )
        tails = np.maximum.accumulate(tails, axis=0)
        return Attempt11DistilledFoldOutput(
            rank_score=rank,
            predicted_delta=delta,
            safe_probability=safe,
            downside_p95=tails[0],
            downside_p99=tails[1],
            downside_max=tails[2],
        )


@dataclass(frozen=True)
class Attempt11DistilledFeatures:
    runtime_sample: Mapping[str, Any]
    features: np.ndarray
    action_keys: tuple[ActionKey, ...]
    candidate_indices: tuple[int, ...]
    baseline_index: int


@dataclass(frozen=True)
class Attempt11DistilledHeadPredictions:
    action_score: np.ndarray
    rank_score: np.ndarray
    predicted_delta: np.ndarray
    safe_probability: np.ndarray
    downside_p95: np.ndarray
    downside_p99: np.ndarray
    downside_max: np.ndarray
    upper_downside_p95: np.ndarray
    upper_downside_p99: np.ndarray
    upper_downside_max: np.ndarray
    rank_disagreement: np.ndarray
    candidate_mask: np.ndarray
    gate_eligible_mask: np.ndarray
    fold_vote_count: np.ndarray
    selected_index: int
    baseline_index: int


@dataclass(frozen=True)
class HuM43Attempt11DistilledModel:
    """Hash-described, opt-in Attempt11 selector; fitting cannot activate it."""

    candidate_generator: HuM43Attempt05Model
    fold_predictors: tuple[Attempt11DistilledFoldPredictor, ...]
    conformal_cushions: tuple[float, float, float] = (0.0, 0.0, 0.0)
    conformal_quantile: float = 0.95
    safety_threshold: float = 0.5
    tail_limits: tuple[float, float, float] = HU_M43_ATTEMPT11_DISTILLED_TAIL_LIMITS
    minimum_fold_votes: int = 4
    safety_enabled: bool = False
    winner_frozen: bool = False
    model_id: str = "hu-m43-attempt11-distilled-unfrozen"
    source_candidate_sha256: str = ATTEMPT11_FROZEN_MODEL_SHA256
    manifest: Mapping[str, Any] = field(default_factory=dict)
    schema: str = HU_M43_ATTEMPT11_DISTILLED_MODEL_SCHEMA
    artifact_schema: str = HU_M43_ATTEMPT11_DISTILLED_ARTIFACT_SCHEMA
    feature_schema: str = HU_M43_ATTEMPT11_DISTILLED_FEATURE_SCHEMA
    head_schema: str = HU_M43_ATTEMPT11_DISTILLED_HEAD_SCHEMA
    action_score_mode: str = HU_M43_ATTEMPT11_DISTILLED_ACTION_SCORE_MODE
    _last_prediction_cache: Any = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.schema != HU_M43_ATTEMPT11_DISTILLED_MODEL_SCHEMA:
            raise ValueError("unsupported Attempt11 distilled model schema")
        if self.artifact_schema != HU_M43_ATTEMPT11_DISTILLED_ARTIFACT_SCHEMA:
            raise ValueError("Attempt11 distilled artifact schema changed")
        if self.feature_schema != HU_M43_ATTEMPT11_DISTILLED_FEATURE_SCHEMA:
            raise ValueError("Attempt11 distilled feature schema changed")
        if self.head_schema != HU_M43_ATTEMPT11_DISTILLED_HEAD_SCHEMA:
            raise ValueError("Attempt11 distilled head schema changed")
        if self.action_score_mode != HU_M43_ATTEMPT11_DISTILLED_ACTION_SCORE_MODE:
            raise ValueError("Attempt11 distilled action score mode changed")
        if len(self.fold_predictors) != HU_M43_ATTEMPT11_DISTILLED_REQUIRED_FOLDS:
            raise ValueError("Attempt11 distilled runtime requires exactly five folds")
        if sorted(int(item.fold_index) for item in self.fold_predictors) != list(
            range(HU_M43_ATTEMPT11_DISTILLED_REQUIRED_FOLDS)
        ):
            raise ValueError("Attempt11 distilled fold indices must be exactly 0..4")
        normalized_sha = _require_sha256(self.source_candidate_sha256)
        if normalized_sha != ATTEMPT11_FROZEN_MODEL_SHA256:
            raise ValueError("Attempt11 distilled source candidate SHA changed")
        object.__setattr__(self, "source_candidate_sha256", normalized_sha)
        # Reuse the teacher's strict candidate-only validation.  In particular,
        # the embedded Lambda model cannot itself be runtime enabled.
        FrozenAttempt11LambdaRanker(
            model=self.candidate_generator,
            artifact_sha256=normalized_sha,
        )
        if self.candidate_generator.model_id != ATTEMPT11_FROZEN_MODEL_ID:
            raise ValueError("Attempt11 distilled source candidate model_id changed")
        if self.safety_enabled and not self.winner_frozen:
            raise ValueError("Attempt11 distilled runtime cannot enable an unfrozen winner")
        if not self.model_id:
            raise ValueError("Attempt11 distilled model_id must be non-empty")
        if not 0.0 < float(self.conformal_quantile) < 1.0:
            raise ValueError("Attempt11 distilled conformal quantile must be in (0,1)")
        if not 0.0 <= float(self.safety_threshold) <= 1.0:
            raise ValueError("Attempt11 distilled safety threshold must be in [0,1]")
        if not 1 <= int(self.minimum_fold_votes) <= len(self.fold_predictors):
            raise ValueError("Attempt11 distilled minimum fold votes is invalid")
        if len(self.conformal_cushions) != 3 or any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in self.conformal_cushions
        ):
            raise ValueError("Attempt11 distilled conformal cushions are invalid")
        if len(self.tail_limits) != 3 or any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in self.tail_limits
        ):
            raise ValueError("Attempt11 distilled tail limits are invalid")

    @property
    def runtime_contract(self) -> dict[str, Any]:
        return {
            "authorization": HU_M43_ATTEMPT11_DISTILLED_RUNTIME_AUTHORIZATION,
            "feature_schema": self.feature_schema,
            "head_schema": self.head_schema,
            "conformal_schema": HU_M43_ATTEMPT11_DISTILLED_CONFORMAL_SCHEMA,
            "candidate_source_sha256": self.source_candidate_sha256,
            "candidate_nonbaseline_max": ATTEMPT11_CANDIDATE_MAX,
            "candidate_count_is_variable": True,
            "candidate_padding": False,
            "candidate_duplicates": False,
            "runtime_teacher_ev": False,
            "runtime_teacher_lcb": False,
            "opponent_private_discard_input": False,
            "opponent_profile_runtime_feature": False,
            "baseline_fallback": True,
            "runtime_binding_verified": False,
            "current_profile_mutated": False,
        }

    @property
    def runtime_binding_verified(self) -> bool:
        """Read-only diagnostic; raw artifacts are never runtime-authorized."""

        return False

    def build_features(
        self, sample: Mapping[str, Any], *, baseline_index: int | None = None
    ) -> Attempt11DistilledFeatures:
        return build_attempt11_distilled_features(
            sample,
            candidate_generator=self.candidate_generator,
            source_candidate_sha256=self.source_candidate_sha256,
            baseline_index=baseline_index,
        )

    def predict_heads_sample(
        self, sample: Mapping[str, Any], *, baseline_index: int | None = None
    ) -> Attempt11DistilledHeadPredictions:
        return self._predict_heads_sample(sample, baseline_index=baseline_index)

    def _predict_heads_sample(
        self,
        sample: Mapping[str, Any],
        *,
        baseline_index: int | None = None,
        _runtime_attestation: FrozenExecutionModulesAttestation | None = None,
    ) -> Attempt11DistilledHeadPredictions:
        built = self.build_features(sample, baseline_index=baseline_index)
        action_count = built.features.shape[0]
        outputs = [
            predictor.predict(built.features)
            for predictor in sorted(
                self.fold_predictors, key=lambda item: item.fold_index
            )
        ]
        for output in outputs:
            _validate_fold_output(output, action_count)

        rank_folds = np.vstack([item.rank_score for item in outputs])
        delta_folds = np.vstack([item.predicted_delta for item in outputs])
        safe_folds = np.vstack([item.safe_probability for item in outputs])
        tail_folds = np.stack(
            [
                np.vstack(
                    (item.downside_p95, item.downside_p99, item.downside_max)
                )
                for item in outputs
            ],
            axis=0,
        )
        rank = np.mean(rank_folds, axis=0)
        disagreement = np.std(rank_folds, axis=0)
        delta = np.mean(delta_folds, axis=0)
        safe = np.mean(safe_folds, axis=0)
        tails = np.mean(tail_folds, axis=0)
        tails = np.maximum.accumulate(np.maximum(tails, 0.0), axis=0)
        upper = tails + np.asarray(self.conformal_cushions, dtype=np.float64).reshape(3, 1)

        candidate_mask = np.zeros(action_count, dtype=bool)
        candidate_mask[list(built.candidate_indices)] = True
        limits = np.asarray(self.tail_limits, dtype=np.float64).reshape(3, 1)
        mean_eligible = (
            candidate_mask
            & (safe >= float(self.safety_threshold))
            & (delta > 0.0)
            & np.all(upper <= limits, axis=0)
        )

        vote_count = np.zeros(action_count, dtype=np.int16)
        for fold_index, output in enumerate(outputs):
            fold_tail = np.vstack(
                (output.downside_p95, output.downside_p99, output.downside_max)
            )
            fold_upper = fold_tail + np.asarray(
                self.conformal_cushions, dtype=np.float64
            ).reshape(3, 1)
            fold_eligible = (
                candidate_mask
                & (output.safe_probability >= float(self.safety_threshold))
                & (output.predicted_delta > 0.0)
                & np.all(fold_upper <= limits, axis=0)
            )
            eligible_indices = np.flatnonzero(fold_eligible)
            if eligible_indices.size:
                winner = _canonical_argmax(
                    output.rank_score,
                    eligible_indices,
                    built.action_keys,
                )
                vote_count[winner] += 1

        gate_eligible = mean_eligible & (
            vote_count >= int(self.minimum_fold_votes)
        )
        # Offline/raw artifacts may expose diagnostic heads, but can never
        # emit an override-capable action score.  Authorization is carried by
        # the loader-issued wrapper, not by a dataclass field that replace()
        # or pickle state could forge.
        runtime_authorized = bool(
            self.safety_enabled
            and self.winner_frozen
            and type(_runtime_attestation) is FrozenExecutionModulesAttestation
            and _runtime_attestation.covers(ATTEMPT11_BOUND_EXECUTION_MODULES)
        )
        final_candidates = (
            np.flatnonzero(gate_eligible)
            if runtime_authorized
            else np.asarray([], dtype=np.int64)
        )
        selected = built.baseline_index
        if final_candidates.size:
            selected = _canonical_argmax(rank, final_candidates, built.action_keys)
        action_score = np.full(action_count, -1.0, dtype=np.float64)
        action_score[built.baseline_index] = 0.0
        if selected != built.baseline_index:
            action_score[selected] = 1.0

        return Attempt11DistilledHeadPredictions(
            action_score=action_score,
            rank_score=rank,
            predicted_delta=delta,
            safe_probability=safe,
            downside_p95=tails[0],
            downside_p99=tails[1],
            downside_max=tails[2],
            upper_downside_p95=upper[0],
            upper_downside_p99=upper[1],
            upper_downside_max=upper[2],
            rank_disagreement=disagreement,
            candidate_mask=candidate_mask,
            gate_eligible_mask=gate_eligible,
            fold_vote_count=vote_count,
            selected_index=int(selected),
            baseline_index=built.baseline_index,
        )

    def predict_sample_with_baseline(
        self, sample: Mapping[str, Any], *, baseline_index: int
    ) -> np.ndarray:
        heads = self.predict_heads_sample(sample, baseline_index=baseline_index)
        object.__setattr__(
            self,
            "_last_prediction_cache",
            (_prediction_cache_key(sample, baseline_index), heads),
        )
        return heads.action_score.copy()

    def predict_sample(self, sample: Mapping[str, Any]) -> np.ndarray:
        baseline = _resolve_baseline(sample, None, len(sample.get("actions", ())))
        return self.predict_sample_with_baseline(sample, baseline_index=baseline)

    def predict_safety_probability(
        self,
        sample: Mapping[str, Any],
        *,
        candidate_index: int,
        baseline_index: int,
    ) -> float:
        cache_key = _prediction_cache_key(sample, baseline_index)
        cached = self._last_prediction_cache
        heads = (
            cached[1]
            if isinstance(cached, tuple) and len(cached) == 2 and cached[0] == cache_key
            else self.predict_heads_sample(sample, baseline_index=baseline_index)
        )
        if (
            isinstance(candidate_index, bool)
            or not isinstance(candidate_index, (int, np.integer))
            or not 0 <= int(candidate_index) < len(heads.action_score)
        ):
            raise IndexError("Attempt11 distilled candidate index is invalid")
        candidate = int(candidate_index)
        if candidate != heads.selected_index or candidate == heads.baseline_index:
            return 0.0
        return float(heads.safe_probability[candidate])

    def __getstate__(self) -> dict[str, Any]:
        return {
            item.name: getattr(self, item.name)
            for item in dataclass_fields(self)
            if item.name != "_last_prediction_cache"
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        if "runtime_binding_verified" in state:
            state = {
                name: value
                for name, value in state.items()
                if name != "runtime_binding_verified"
            }
        for name, value in state.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_last_prediction_cache", None)
        self.__post_init__()

    def save(self, path: str | Path) -> str:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        envelope = {
            "artifact_schema": HU_M43_ATTEMPT11_DISTILLED_ARTIFACT_SCHEMA,
            "model_schema": HU_M43_ATTEMPT11_DISTILLED_MODEL_SCHEMA,
            "model": self,
        }
        encoded = pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL)
        digest = hashlib.sha256(encoded).hexdigest()
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return digest

    @classmethod
    def load(
        cls, path: str | Path, *, expected_sha256: str | None = None
    ) -> "HuM43Attempt11DistilledModel":
        encoded = Path(path).read_bytes()
        digest = hashlib.sha256(encoded).hexdigest()
        if expected_sha256 is not None and digest != _require_sha256(expected_sha256):
            raise ValueError("Attempt11 distilled artifact SHA-256 mismatch")
        envelope = pickle.loads(encoded)
        if not isinstance(envelope, dict) or set(envelope) != {
            "artifact_schema",
            "model_schema",
            "model",
        }:
            raise ValueError("Attempt11 distilled artifact envelope is invalid")
        if (
            envelope["artifact_schema"]
            != HU_M43_ATTEMPT11_DISTILLED_ARTIFACT_SCHEMA
            or envelope["model_schema"]
            != HU_M43_ATTEMPT11_DISTILLED_MODEL_SCHEMA
        ):
            raise ValueError("Attempt11 distilled artifact schema mismatch")
        model = envelope["model"]
        if not isinstance(model, cls):
            raise TypeError("Attempt11 distilled artifact payload type mismatch")
        model.__post_init__()
        return model


class BoundHuM43Attempt11DistilledModel:
    """Non-serializable runtime view issued only after complete verification."""

    __slots__ = ("__model", "__attestation")

    def __init__(
        self,
        model: HuM43Attempt11DistilledModel,
        attestation: FrozenExecutionModulesAttestation,
    ) -> None:
        if (
            not _called_by_verified_attempt11_loader(model, attestation)
            or type(attestation) is not FrozenExecutionModulesAttestation
            or not attestation.covers(ATTEMPT11_BOUND_EXECUTION_MODULES)
        ):
            raise TypeError("Attempt11 bound runtime wrappers are loader-issued")
        if model.safety_enabled is not True or model.winner_frozen is not True:
            raise ValueError("Attempt11 bound runtime requires a frozen safe model")
        object.__setattr__(self, "_BoundHuM43Attempt11DistilledModel__model", model)
        object.__setattr__(
            self,
            "_BoundHuM43Attempt11DistilledModel__attestation",
            attestation,
        )

    def __require_capability(self) -> None:
        try:
            attestation = self.__attestation
        except AttributeError as error:
            raise RuntimeError("Attempt11 runtime binding capability is absent") from error
        if (
            type(attestation) is not FrozenExecutionModulesAttestation
            or not attestation.covers(ATTEMPT11_BOUND_EXECUTION_MODULES)
        ):
            raise RuntimeError("Attempt11 runtime binding capability is invalid")

    @property
    def runtime_binding_verified(self) -> bool:
        self.__require_capability()
        return True

    @property
    def runtime_contract(self) -> dict[str, Any]:
        self.__require_capability()
        contract = dict(self.__model.runtime_contract)
        contract["runtime_binding_verified"] = True
        return contract

    def predict_heads_sample(
        self, sample: Mapping[str, Any], *, baseline_index: int | None = None
    ) -> Attempt11DistilledHeadPredictions:
        self.__require_capability()
        return self.__model._predict_heads_sample(
            sample,
            baseline_index=baseline_index,
            _runtime_attestation=self.__attestation,
        )

    def predict_sample_with_baseline(
        self, sample: Mapping[str, Any], *, baseline_index: int
    ) -> np.ndarray:
        heads = self.predict_heads_sample(sample, baseline_index=baseline_index)
        object.__setattr__(
            self.__model,
            "_last_prediction_cache",
            (_prediction_cache_key(sample, baseline_index), heads),
        )
        return heads.action_score.copy()

    def predict_sample(self, sample: Mapping[str, Any]) -> np.ndarray:
        baseline = _resolve_baseline(sample, None, len(sample.get("actions", ())))
        return self.predict_sample_with_baseline(sample, baseline_index=baseline)

    def predict_safety_probability(
        self,
        sample: Mapping[str, Any],
        *,
        candidate_index: int,
        baseline_index: int,
    ) -> float:
        cache_key = _prediction_cache_key(sample, baseline_index)
        cached = self.__model._last_prediction_cache
        heads = (
            cached[1]
            if isinstance(cached, tuple) and len(cached) == 2 and cached[0] == cache_key
            else self.predict_heads_sample(sample, baseline_index=baseline_index)
        )
        if (
            isinstance(candidate_index, bool)
            or not isinstance(candidate_index, (int, np.integer))
            or not 0 <= int(candidate_index) < len(heads.action_score)
        ):
            raise IndexError("Attempt11 distilled candidate index is invalid")
        candidate = int(candidate_index)
        if candidate != heads.selected_index or candidate == heads.baseline_index:
            return 0.0
        return float(heads.safe_probability[candidate])

    def __getattr__(self, name: str) -> Any:
        self.__require_capability()
        return getattr(self.__model, name)

    def __reduce__(self) -> Any:
        raise TypeError("Attempt11 bound runtime wrappers cannot be serialized")


def is_bound_attempt11_distilled_model(value: object) -> bool:
    """Return True only for a wrapper carrying the loader-issued capability."""

    if not isinstance(value, BoundHuM43Attempt11DistilledModel):
        return False
    try:
        return value.runtime_binding_verified is True
    except (AttributeError, RuntimeError):
        return False


def _called_by_verified_attempt11_loader(
    model: HuM43Attempt11DistilledModel,
    attestation: FrozenExecutionModulesAttestation,
) -> bool:
    """Reject ordinary constructor calls; only the attested loader may issue."""

    frame = inspect.currentframe()
    try:
        init_frame = None if frame is None else frame.f_back
        caller = None if init_frame is None else init_frame.f_back
        acceptance = sys.modules.get(
            "ofc_regular.validate_hu_m43_attempt11_acceptance"
        )
        loader = (
            None
            if acceptance is None
            else getattr(acceptance, "load_bound_attempt11_distilled_model", None)
        )
        return bool(
            caller is not None
            and loader is not None
            and caller.f_globals.get("__name__")
            == "ofc_regular.validate_hu_m43_attempt11_acceptance"
            and caller.f_code is getattr(loader, "__code__", None)
            and caller.f_locals.get("model") is model
            and caller.f_locals.get("execution_attestation") is attestation
        )
    finally:
        del frame


def build_attempt11_distilled_features(
    sample: Mapping[str, Any],
    *,
    candidate_generator: HuM43Attempt05Model,
    source_candidate_sha256: str,
    baseline_index: int | None = None,
) -> Attempt11DistilledFeatures:
    """Rebuild legal actions and append frozen Lambda proposal diagnostics."""

    raw_actions = sample.get("actions")
    if isinstance(raw_actions, (str, bytes)) or not isinstance(raw_actions, Sequence):
        raise ValueError("Attempt11 distilled runtime requires ordered actions")
    baseline = _resolve_baseline(sample, baseline_index, len(raw_actions))
    projected_input = dict(sample)
    projected_input["baseline_action_row_index"] = baseline
    projected_input["baseline_action_key"] = action_key_from_payload(
        raw_actions[baseline]
    ).to_token()
    runtime_sample, observation = _runtime_policy_projection(projected_input)
    if observation.street != "T1" or observation.seat != "second":
        raise ValueError("Attempt11 distilled runtime authorizes only T1-second")

    legal = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    by_key = index_actions_by_key(legal)
    ordered_actions: list[Action] = []
    for payload in runtime_sample["actions"]:
        key = action_key_from_payload(payload)
        ordered_actions.append(legal[by_key[key]])
    keys = tuple(action_key(item) for item in ordered_actions)

    frozen_ranker = FrozenAttempt11LambdaRanker(
        model=candidate_generator,
        artifact_sha256=_require_sha256(source_candidate_sha256),
    )
    matrix, _unused = sample_to_matrix(dict(runtime_sample))
    paired = build_paired_action_features_matrix(matrix, baseline_index=baseline)
    if paired.shape != (len(ordered_actions), 4 * HU_FEATURE_DIM):
        raise ValueError("Attempt11 distilled paired feature shape changed")
    scores = _score_frozen_candidate_generator(
        frozen_ranker,
        paired=paired,
        observation=observation,
        ordered_actions=ordered_actions,
        baseline_index=baseline,
    )
    nonbaseline = [index for index in range(len(ordered_actions)) if index != baseline]
    nonbaseline.sort(
        key=lambda index: (-float(scores.rank_mean[index]), keys[index].sort_key())
    )
    candidates = tuple(nonbaseline[:ATTEMPT11_CANDIDATE_MAX])
    candidate_rank = {index: rank + 1 for rank, index in enumerate(candidates)}

    rank_values = np.asarray(scores.rank_mean, dtype=np.float64)
    best_other = np.empty(len(ordered_actions), dtype=np.float64)
    for index in range(len(ordered_actions)):
        alternatives = np.delete(rank_values, index)
        best_other[index] = float(np.max(alternatives)) if alternatives.size else 0.0
    p95 = np.asarray(scores.raw_downside_p95, dtype=np.float64)
    p99 = np.asarray(scores.raw_downside_p99, dtype=np.float64)
    maximum = np.asarray(scores.raw_downside_max, dtype=np.float64)
    normalized_risk = np.maximum.reduce((p95 / 22.0, p99 / 36.0, maximum / 45.0))
    extra = np.column_stack(
        (
            rank_values,
            np.asarray(scores.rank_disagreement, dtype=np.float64),
            rank_values - rank_values[baseline],
            rank_values - best_other,
            p95 / 22.0,
            p99 / 36.0,
            maximum / 45.0,
            normalized_risk,
            np.asarray(
                [
                    candidate_rank.get(index, ATTEMPT11_CANDIDATE_MAX + 1)
                    / ATTEMPT11_CANDIDATE_MAX
                    for index in range(len(ordered_actions))
                ],
                dtype=np.float64,
            ),
            np.asarray(
                [float(index == baseline) for index in range(len(ordered_actions))],
                dtype=np.float64,
            ),
        )
    )
    features = np.concatenate((paired, extra.astype(np.float32)), axis=1).astype(
        np.float32, copy=False
    )
    if (
        features.shape
        != (len(ordered_actions), HU_M43_ATTEMPT11_DISTILLED_FEATURE_DIM)
        or not np.isfinite(features).all()
    ):
        raise ValueError("Attempt11 distilled feature matrix is invalid")
    return Attempt11DistilledFeatures(
        runtime_sample=runtime_sample,
        features=features,
        action_keys=keys,
        candidate_indices=candidates,
        baseline_index=baseline,
    )


def _score_frozen_candidate_generator(
    ranker: FrozenAttempt11LambdaRanker,
    *,
    paired: np.ndarray,
    observation: ActorObservation,
    ordered_actions: Sequence[Action],
    baseline_index: int,
) -> Attempt11RankScores:
    """Score the frozen Lambda folds once over a shared paired feature matrix.

    The legacy fold API reconstructs the same 4x feature matrix independently
    for every fold and also evaluates an unused gain head.  Production
    LambdaRank artifacts expose their estimators, so this equivalent path is
    substantially faster.  Protocol-only test doubles retain the strict legacy
    fallback.
    """

    predictors = sorted(
        ranker.model.fold_predictors, key=lambda item: item.fold_index
    )
    fast_fields = (
        "ranker",
        "tail_p95_head",
        "tail_p99_head",
        "tail_max_head",
    )
    if not all(all(hasattr(predictor, name) for name in fast_fields) for predictor in predictors):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names.*",
                category=UserWarning,
            )
            return ranker.score_actions(
                observation, ordered_actions, baseline_index=baseline_index
            )
    if [int(item.fold_index) for item in predictors] != list(range(5)) or any(
        item.family != "lambda_rank" for item in predictors
    ):
        raise ValueError("Attempt11 distilled candidate fold identities changed")
    rank_folds: list[np.ndarray] = []
    tail_folds: list[np.ndarray] = []
    for predictor in predictors:
        rank_values = _regression(predictor.ranker, paired, "source rank")
        fold_tails = np.vstack(
            (
                np.maximum(
                    0.0,
                    _regression(predictor.tail_p95_head, paired, "source p95"),
                ),
                np.maximum(
                    0.0,
                    _regression(predictor.tail_p99_head, paired, "source p99"),
                ),
                np.maximum(
                    0.0,
                    _regression(predictor.tail_max_head, paired, "source max"),
                ),
            )
        )
        rank_folds.append(rank_values)
        tail_folds.append(np.maximum.accumulate(fold_tails, axis=0))
    rank_matrix = np.vstack(rank_folds)
    raw_tails = np.mean(np.stack(tail_folds, axis=0), axis=0)
    result = Attempt11RankScores(
        rank_mean=tuple(float(value) for value in np.mean(rank_matrix, axis=0)),
        rank_disagreement=tuple(float(value) for value in np.std(rank_matrix, axis=0)),
        raw_downside_p95=tuple(float(value) for value in raw_tails[0]),
        raw_downside_p99=tuple(float(value) for value in raw_tails[1]),
        raw_downside_max=tuple(float(value) for value in raw_tails[2]),
        fold_count=len(predictors),
    )
    result.validate(len(ordered_actions))
    return result


def _resolve_baseline(
    sample: Mapping[str, Any], baseline_index: int | None, action_count: int
) -> int:
    declared = sample.get("baseline_action_row_index")
    value = declared if baseline_index is None else baseline_index
    if baseline_index is not None and declared is not None and int(declared) != int(
        baseline_index
    ):
        raise ValueError("Attempt11 distilled baseline index disagrees with sample")
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("Attempt11 distilled baseline index must be an integer")
    result = int(value)
    if not 0 <= result < action_count:
        raise IndexError("Attempt11 distilled baseline index is outside actions")
    declared_key = sample.get("baseline_action_key")
    if declared_key is not None:
        if not isinstance(declared_key, str):
            raise TypeError("Attempt11 distilled baseline ActionKey must be a string")
        if action_key_from_payload(sample["actions"][result]).to_token() != declared_key:
            raise ValueError("Attempt11 distilled baseline ActionKey disagrees with index")
    return result


def _prediction_cache_key(
    sample: Mapping[str, Any], baseline_index: int
) -> tuple[str, tuple[str, ...], int]:
    raw_actions = sample.get("actions")
    if isinstance(raw_actions, (str, bytes)) or not isinstance(raw_actions, Sequence):
        raise ValueError("Attempt11 distilled cache requires ordered actions")
    baseline = _resolve_baseline(sample, baseline_index, len(raw_actions))
    projected = dict(sample)
    projected["baseline_action_row_index"] = baseline
    projected["baseline_action_key"] = action_key_from_payload(
        raw_actions[baseline]
    ).to_token()
    runtime_sample, observation = _runtime_policy_projection(projected)
    return (
        observation.fingerprint(),
        tuple(
            action_key_from_payload(action).to_token()
            for action in runtime_sample["actions"]
        ),
        baseline,
    )


def _canonical_argmax(
    values: np.ndarray, indices: Sequence[int] | np.ndarray, keys: tuple[ActionKey, ...]
) -> int:
    candidates = [int(index) for index in indices]
    if not candidates:
        raise ValueError("Attempt11 distilled argmax requires candidates")
    best = max(float(values[index]) for index in candidates)
    return min(
        (index for index in candidates if float(values[index]) == best),
        key=lambda index: keys[index].sort_key(),
    )


def _validate_fold_output(
    output: Attempt11DistilledFoldOutput, action_count: int
) -> None:
    for name in (
        "rank_score",
        "predicted_delta",
        "safe_probability",
        "downside_p95",
        "downside_p99",
        "downside_max",
    ):
        values = np.asarray(getattr(output, name), dtype=np.float64)
        if values.shape != (action_count,) or not np.isfinite(values).all():
            raise ValueError(f"Attempt11 distilled fold {name} output is invalid")
    safe = np.asarray(output.safe_probability, dtype=np.float64)
    if np.any((safe < 0.0) | (safe > 1.0)):
        raise ValueError("Attempt11 distilled safe probability is outside [0,1]")
    if np.any(output.downside_p95 < 0.0) or np.any(
        output.downside_p95 > output.downside_p99
    ) or np.any(output.downside_p99 > output.downside_max):
        raise ValueError("Attempt11 distilled downside heads are not monotone")


def _regression(estimator: Any, features: np.ndarray, label: str) -> np.ndarray:
    predictor = getattr(estimator, "booster_", estimator)
    result = np.asarray(predictor.predict(features), dtype=np.float64).reshape(-1)
    if result.shape != (features.shape[0],) or not np.isfinite(result).all():
        raise ValueError(f"Attempt11 distilled {label} prediction is invalid")
    return result


def _probability(estimator: Any, features: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "booster_"):
        result = np.asarray(
            estimator.booster_.predict(features), dtype=np.float64
        ).reshape(-1)
    elif hasattr(estimator, "predict_proba"):
        raw = np.asarray(estimator.predict_proba(features), dtype=np.float64)
        if raw.shape == (features.shape[0], 2):
            result = raw[:, 1]
        elif raw.shape == (features.shape[0], 1):
            classes = np.asarray(getattr(estimator, "classes_", ()), dtype=np.int8)
            if classes.shape != (1,):
                raise ValueError("Attempt11 distilled safe classes are invalid")
            result = np.full(features.shape[0], float(classes[0] == 1))
        else:
            raise ValueError("Attempt11 distilled safe predict_proba shape is invalid")
    else:
        result = _regression(estimator, features, "safe")
    if not np.isfinite(result).all() or np.any((result < 0.0) | (result > 1.0)):
        raise ValueError("Attempt11 distilled safe probability is invalid")
    return result


def _require_sha256(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("Attempt11 distilled SHA-256 must be a string")
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("Attempt11 distilled SHA-256 is invalid")
    return normalized


__all__ = [
    "Attempt11DistilledFeatures",
    "Attempt11DistilledFoldOutput",
    "Attempt11DistilledFoldPredictor",
    "Attempt11DistilledHeadPredictions",
    "BoundHuM43Attempt11DistilledModel",
    "HU_M43_ATTEMPT11_DISTILLED_ACTION_SCORE_MODE",
    "HU_M43_ATTEMPT11_DISTILLED_ARTIFACT_SCHEMA",
    "HU_M43_ATTEMPT11_DISTILLED_FEATURE_DIM",
    "HU_M43_ATTEMPT11_DISTILLED_FEATURE_SCHEMA",
    "HU_M43_ATTEMPT11_DISTILLED_HEAD_SCHEMA",
    "HU_M43_ATTEMPT11_DISTILLED_MODEL_SCHEMA",
    "HuM43Attempt11DistilledModel",
    "build_attempt11_distilled_features",
    "is_bound_attempt11_distilled_model",
]
