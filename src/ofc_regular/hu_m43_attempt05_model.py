"""Opt-in M4.3 Attempt05 architecture-neutral T1-second runtime model.

Attempt05 compares exactly two pre-registered model families:

``lambda_rank``
    A LightGBM LambdaRank action ranker with separate gain and distributional
    downside heads over paired candidate-versus-baseline features.

``deepsets``
    A compact permutation-invariant card/action encoder with listwise, gain,
    and distributional downside heads.

The common runtime wrapper deliberately has no teacher-value input.  It
reconstructs the complete legal action set from a canonical ActorObservation,
maintains positional mapping with ActionKey, authorizes only T1-second, and
falls back to the declared baseline unless a separately frozen runtime gate is
enabled.  Merely fitting or loading an Attempt05 artifact cannot activate it.
"""

from __future__ import annotations

import hashlib
import math
import os
import pickle
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .action_key import ActionKey, action_key, action_key_from_payload, index_actions_by_key
from .action_space import generate_turn_actions
from .cards import ALL_CARDS
from .hu_infoset import ActorObservation
from .hu_m4_joint_model import build_paired_action_features_matrix
from .hu_turn3_model import HU_FEATURE_DIM, hu_policy_sample, sample_to_matrix


HU_M43_ATTEMPT05_MODEL_SCHEMA = "hu_m43_attempt05_t1_second_model_v1"
HU_M43_ATTEMPT05_ARTIFACT_SCHEMA = "hu_m43_attempt05_t1_second_pickle_v1"
HU_M43_ATTEMPT05_HEAD_SCHEMA = "hu_m43_attempt05_rank_gain_distributional_tail_v1"
HU_M43_ATTEMPT05_CONFORMAL_SCHEMA = "hu_m43_attempt05_oof_upper_residual_conformal_v1"
HU_M43_ATTEMPT05_RUNTIME_AUTHORIZATION = "T1-second-only"
HU_M43_ATTEMPT05_FAMILIES = frozenset({"lambda_rank", "deepsets"})
HU_M43_ATTEMPT05_REQUIRED_FOLDS = 5
HU_M43_ATTEMPT05_TAIL_NAMES = ("p95", "p99", "max")
HU_M43_ATTEMPT05_TAIL_LIMITS = (25.0, 40.0, 50.0)
HU_M43_ATTEMPT05_TAIL_SCALES = (25.0, 40.0, 50.0)

DEEPSETS_ROLE_NAMES = (
    "hero_top",
    "hero_middle",
    "hero_bottom",
    "opponent_top",
    "opponent_middle",
    "opponent_bottom",
    "dealt",
    "hero_discard",
    "action_top",
    "action_middle",
    "action_bottom",
    "action_discard",
    "baseline_top",
    "baseline_middle",
    "baseline_bottom",
    "baseline_discard",
)
DEEPSETS_CARD_TOKEN_DIM = 13 + 4 + len(DEEPSETS_ROLE_NAMES)
DEEPSETS_CONTEXT_DIM = 18
_RANK_INDEX = {rank: index for index, rank in enumerate("23456789TJQKA")}
_SUIT_INDEX = {suit: index for index, suit in enumerate("hdcs")}
_ROLE_INDEX = {role: index for index, role in enumerate(DEEPSETS_ROLE_NAMES)}
_STREETS = ("T0", "T1", "T2", "T3", "T4", "FL")


@dataclass(frozen=True)
class Attempt05FoldOutput:
    rank_score: np.ndarray
    gain_probability: np.ndarray
    downside_p95: np.ndarray
    downside_p99: np.ndarray
    downside_max: np.ndarray


class Attempt05FoldPredictor(Protocol):
    family: str
    fold_index: int

    def predict(
        self, runtime_sample: Mapping[str, Any], *, baseline_index: int
    ) -> Attempt05FoldOutput: ...


@dataclass(frozen=True)
class LambdaRankFoldPredictor:
    """One tree-family identity fold; provenance is absent by construction."""

    ranker: Any
    gain_head: Any
    tail_p95_head: Any
    tail_p99_head: Any
    tail_max_head: Any
    fold_index: int
    paired_feature_dim: int = 4 * HU_FEATURE_DIM
    family: str = "lambda_rank"

    def __post_init__(self) -> None:
        if self.family != "lambda_rank":
            raise ValueError("Attempt05 tree fold family mismatch")
        if self.fold_index < 0:
            raise ValueError("Attempt05 fold_index must be non-negative")
        if self.paired_feature_dim != 4 * HU_FEATURE_DIM:
            raise ValueError("Attempt05 paired feature dimension changed")
        for name in (
            "ranker",
            "gain_head",
            "tail_p95_head",
            "tail_p99_head",
            "tail_max_head",
        ):
            if getattr(self, name) is None:
                raise ValueError(f"Attempt05 {name} must not be None")

    def predict(
        self, runtime_sample: Mapping[str, Any], *, baseline_index: int
    ) -> Attempt05FoldOutput:
        matrix, _unused = sample_to_matrix(dict(runtime_sample))
        paired = build_paired_action_features_matrix(
            matrix, baseline_index=baseline_index
        )
        if paired.shape[1] != self.paired_feature_dim:
            raise ValueError("Attempt05 paired runtime feature shape changed")
        rank = _regression(self.ranker, paired, "rank")
        gain = _probability(self.gain_head, paired)
        tails = np.vstack(
            (
                np.maximum(0.0, _regression(self.tail_p95_head, paired, "p95")),
                np.maximum(0.0, _regression(self.tail_p99_head, paired, "p99")),
                np.maximum(0.0, _regression(self.tail_max_head, paired, "max")),
            )
        )
        tails = np.maximum.accumulate(tails, axis=0)
        return Attempt05FoldOutput(rank, gain, tails[0], tails[1], tails[2])


@dataclass(frozen=True)
class DeepSetsFoldPredictor:
    """One compact DeepSets fold stored as a CPU state dictionary."""

    state_dict: Mapping[str, Any]
    fold_index: int
    token_hidden_dim: int = 32
    trunk_hidden_dim: int = 64
    dropout: float = 0.05
    family: str = "deepsets"
    _net_cache: Any = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.family != "deepsets":
            raise ValueError("Attempt05 DeepSets fold family mismatch")
        if self.fold_index < 0:
            raise ValueError("Attempt05 fold_index must be non-negative")
        if not self.state_dict:
            raise ValueError("Attempt05 DeepSets state_dict is empty")
        if self.token_hidden_dim <= 0 or self.trunk_hidden_dim <= 0:
            raise ValueError("Attempt05 DeepSets dimensions must be positive")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("Attempt05 DeepSets dropout must be in [0,1)")

    def predict(
        self, runtime_sample: Mapping[str, Any], *, baseline_index: int
    ) -> Attempt05FoldOutput:
        torch = _import_torch()
        state_tokens, action_tokens, context = encode_deepsets_runtime_sample(
            runtime_sample, baseline_index=baseline_index
        )
        net = self._network(torch)
        net.eval()
        inference_guard = getattr(torch, "inference_mode", torch.no_grad)
        with inference_guard():
            rank, gain_logit, normalized_tails = net(
                torch.from_numpy(state_tokens),
                torch.from_numpy(action_tokens),
                torch.from_numpy(context),
            )
            rank_array = rank.detach().cpu().numpy().astype(np.float64)
            gain = torch.sigmoid(gain_logit).detach().cpu().numpy().astype(np.float64)
            tails = (
                normalized_tails.detach().cpu().numpy().astype(np.float64)
                * np.asarray(HU_M43_ATTEMPT05_TAIL_SCALES)
            ).T
        tails = np.maximum.accumulate(np.maximum(tails, 0.0), axis=0)
        return Attempt05FoldOutput(
            rank_array, gain, tails[0], tails[1], tails[2]
        )

    def _network(self, torch: Any) -> Any:
        cached = self._net_cache
        if cached is not None:
            return cached
        net = build_deepsets_network(
            torch,
            token_hidden_dim=self.token_hidden_dim,
            trunk_hidden_dim=self.trunk_hidden_dim,
            dropout=self.dropout,
        )
        net.load_state_dict(dict(self.state_dict), strict=True)
        net.to("cpu")
        object.__setattr__(self, "_net_cache", net)
        return net

    def __getstate__(self) -> dict[str, Any]:
        # The lazily built network is a local implementation class and must
        # never enter the portable artifact.  CPU tensors in state_dict are the
        # sole serialized neural parameters.
        return {
            "state_dict": self.state_dict,
            "fold_index": self.fold_index,
            "token_hidden_dim": self.token_hidden_dim,
            "trunk_hidden_dim": self.trunk_hidden_dim,
            "dropout": self.dropout,
            "family": self.family,
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        for name in (
            "state_dict",
            "fold_index",
            "token_hidden_dim",
            "trunk_hidden_dim",
            "dropout",
            "family",
        ):
            object.__setattr__(self, name, state[name])
        object.__setattr__(self, "_net_cache", None)
        self.__post_init__()


@dataclass(frozen=True)
class Attempt05HeadPredictions:
    action_score: np.ndarray
    rank_score: np.ndarray
    gain_probability: np.ndarray
    downside_p95: np.ndarray
    downside_p99: np.ndarray
    downside_max: np.ndarray
    upper_downside_p95: np.ndarray
    upper_downside_p99: np.ndarray
    upper_downside_max: np.ndarray
    rank_disagreement: np.ndarray
    gate_eligible_mask: np.ndarray
    proposal_index: int
    proposal_gate_eligible: bool


@dataclass(frozen=True)
class Attempt05CandidateEvaluation:
    baseline_index: int
    candidate_index: int
    candidate_action_key: str
    gain_probability: float
    upper_downside_p95: float
    upper_downside_p99: float
    upper_downside_max: float
    rank_disagreement: float
    gate_eligible: bool


@dataclass(frozen=True)
class Attempt05RuntimeDecision:
    baseline_index: int
    proposal_index: int
    selected_index: int
    override_fired: bool
    authorized: bool
    reason: str


@dataclass(frozen=True)
class HuM43Attempt05Model:
    """Common fail-closed ensemble wrapper for the two Attempt05 families."""

    family: str
    fold_predictors: tuple[Attempt05FoldPredictor, ...]
    conformal_cushions: tuple[float, float, float] = (0.0, 0.0, 0.0)
    conformal_quantile: float = 0.95
    gain_threshold: float = 1.0
    uncertainty_max: float = 0.0
    tail_limits: tuple[float, float, float] = HU_M43_ATTEMPT05_TAIL_LIMITS
    runtime_enabled: bool = False
    winner_frozen: bool = False
    model_id: str = "hu-m43-attempt05-unfrozen"
    manifest: Mapping[str, Any] = field(default_factory=dict)
    schema: str = HU_M43_ATTEMPT05_MODEL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != HU_M43_ATTEMPT05_MODEL_SCHEMA:
            raise ValueError("unsupported Attempt05 model schema")
        if self.family not in HU_M43_ATTEMPT05_FAMILIES:
            raise ValueError("unsupported Attempt05 model family")
        if len(self.fold_predictors) != HU_M43_ATTEMPT05_REQUIRED_FOLDS:
            raise ValueError("Attempt05 runtime requires exactly five identity folds")
        indices = []
        for predictor in self.fold_predictors:
            if predictor.family != self.family:
                raise ValueError("Attempt05 artifact mixes model families")
            indices.append(int(predictor.fold_index))
        if sorted(indices) != list(range(HU_M43_ATTEMPT05_REQUIRED_FOLDS)):
            raise ValueError("Attempt05 fold indices must be exactly 0..4")
        if self.runtime_enabled and not self.winner_frozen:
            raise ValueError("Attempt05 runtime cannot enable an unfrozen winner")
        if not self.model_id:
            raise ValueError("Attempt05 model_id must be non-empty")
        if not 0.0 < float(self.conformal_quantile) < 1.0:
            raise ValueError("Attempt05 conformal quantile must be in (0,1)")
        for name, values, positive in (
            ("conformal_cushions", self.conformal_cushions, False),
            ("tail_limits", self.tail_limits, True),
        ):
            if len(values) != 3 or not all(math.isfinite(float(v)) for v in values):
                raise ValueError(f"Attempt05 {name} must have three finite values")
            if positive and any(float(v) <= 0.0 for v in values):
                raise ValueError("Attempt05 tail limits must be positive")
            if not positive and any(float(v) < 0.0 for v in values):
                raise ValueError("Attempt05 conformal cushions must be non-negative")
        if not 0.0 <= float(self.gain_threshold) <= 1.0:
            raise ValueError("Attempt05 gain threshold must be in [0,1]")
        if not math.isfinite(float(self.uncertainty_max)) or self.uncertainty_max < 0.0:
            raise ValueError("Attempt05 uncertainty_max must be finite and non-negative")

    @property
    def runtime_contract(self) -> dict[str, Any]:
        return {
            "authorization": HU_M43_ATTEMPT05_RUNTIME_AUTHORIZATION,
            "head_schema": HU_M43_ATTEMPT05_HEAD_SCHEMA,
            "conformal_schema": HU_M43_ATTEMPT05_CONFORMAL_SCHEMA,
            "runtime_teacher_ev": False,
            "runtime_teacher_lcb": False,
            "profile_runtime_feature": False,
            "baseline_fallback": True,
            "current_profile_mutated": False,
        }

    def predict_heads_sample(
        self, sample: Mapping[str, Any], *, baseline_index: int | None = None
    ) -> Attempt05HeadPredictions:
        runtime_sample, observation = _runtime_policy_projection(sample)
        if observation.street != "T1" or observation.seat != "second":
            raise ValueError("Attempt05 heads authorize only T1-second")
        action_count = len(runtime_sample["actions"])
        baseline = _resolve_baseline_index(sample, baseline_index, action_count)
        outputs = [
            predictor.predict(runtime_sample, baseline_index=baseline)
            for predictor in sorted(
                self.fold_predictors, key=lambda item: item.fold_index
            )
        ]
        for output in outputs:
            _validate_fold_output(output, action_count)
        rank_folds = np.vstack([output.rank_score for output in outputs])
        rank = np.mean(rank_folds, axis=0)
        disagreement = np.std(rank_folds, axis=0)
        gain = np.mean(np.vstack([output.gain_probability for output in outputs]), axis=0)
        tails = np.vstack(
            [
                np.mean(np.vstack([getattr(output, f"downside_{name}") for output in outputs]), axis=0)
                for name in HU_M43_ATTEMPT05_TAIL_NAMES
            ]
        )
        tails = np.maximum.accumulate(np.maximum(tails, 0.0), axis=0)
        upper = tails + np.asarray(self.conformal_cushions).reshape(3, 1)
        nonbaseline = np.asarray(
            [index for index in range(action_count) if index != baseline], dtype=np.int32
        )
        proposal = _canonical_argmax(
            rank,
            nonbaseline,
            tuple(action_key_from_payload(action) for action in runtime_sample["actions"]),
        )
        gate_mask = (
            (gain >= float(self.gain_threshold))
            & (disagreement <= float(self.uncertainty_max))
            & np.all(
                upper <= np.asarray(self.tail_limits).reshape(3, 1), axis=0
            )
        )
        gate_mask[baseline] = False
        gate = bool(gate_mask[proposal])
        action_score = np.full(action_count, -1.0, dtype=np.float64)
        action_score[baseline] = 0.0
        if gate and self.runtime_enabled and self.winner_frozen:
            action_score[proposal] = 1.0
        rank[baseline] = 0.0
        disagreement[baseline] = 0.0
        for array in (rank, gain, tails, upper, disagreement, action_score):
            if not np.isfinite(array).all():
                raise ValueError("Attempt05 prediction contains non-finite values")
        return Attempt05HeadPredictions(
            action_score=action_score,
            rank_score=rank,
            gain_probability=gain,
            downside_p95=tails[0],
            downside_p99=tails[1],
            downside_max=tails[2],
            upper_downside_p95=upper[0],
            upper_downside_p99=upper[1],
            upper_downside_max=upper[2],
            rank_disagreement=disagreement,
            gate_eligible_mask=gate_mask,
            proposal_index=proposal,
            proposal_gate_eligible=gate,
        )

    def predict_sample(self, sample: Mapping[str, Any]) -> np.ndarray:
        return self.predict_heads_sample(sample).action_score

    def select_action_index(
        self, sample: Mapping[str, Any], *, baseline_index: int | None = None
    ) -> Attempt05RuntimeDecision:
        action_count = _action_count(sample)
        baseline = _resolve_baseline_index(sample, baseline_index, action_count)
        raw_observation = sample.get("policy_observation")
        try:
            observation = ActorObservation.from_dict(raw_observation) if isinstance(raw_observation, Mapping) else None
        except (TypeError, ValueError):
            observation = None
        if observation is None or observation.street != "T1" or observation.seat != "second":
            return Attempt05RuntimeDecision(
                baseline, baseline, baseline, False, False, "unauthorized_or_invalid_observation"
            )
        try:
            heads = self.predict_heads_sample(sample, baseline_index=baseline)
        except (TypeError, ValueError, IndexError):
            return Attempt05RuntimeDecision(
                baseline, baseline, baseline, False, True, "fail_closed_prediction_error"
            )
        fired = bool(
            self.runtime_enabled
            and self.winner_frozen
            and heads.proposal_gate_eligible
            and heads.action_score[heads.proposal_index] > heads.action_score[baseline]
        )
        return Attempt05RuntimeDecision(
            baseline_index=baseline,
            proposal_index=heads.proposal_index,
            selected_index=heads.proposal_index if fired else baseline,
            override_fired=fired,
            authorized=True,
            reason="override" if fired else "baseline_fallback",
        )

    def evaluate_candidate(
        self,
        sample: Mapping[str, Any],
        *,
        candidate_index: int,
        baseline_index: int | None = None,
        candidate_action_key: str | ActionKey | None = None,
    ) -> Attempt05CandidateEvaluation:
        """Evaluate an external search winner with the frozen per-action heads.

        This is the integration boundary for the Rust top-K search.  The
        learned rank argmax is not substituted for the search-selected action.
        """

        action_count = _action_count(sample)
        baseline = _resolve_baseline_index(sample, baseline_index, action_count)
        candidate = _resolve_candidate_index(
            sample,
            candidate_index=candidate_index,
            baseline_index=baseline,
            candidate_action_key=candidate_action_key,
        )
        heads = self.predict_heads_sample(sample, baseline_index=baseline)
        key = action_key_from_payload(sample["actions"][candidate]).to_token()
        return Attempt05CandidateEvaluation(
            baseline_index=baseline,
            candidate_index=candidate,
            candidate_action_key=key,
            gain_probability=float(heads.gain_probability[candidate]),
            upper_downside_p95=float(heads.upper_downside_p95[candidate]),
            upper_downside_p99=float(heads.upper_downside_p99[candidate]),
            upper_downside_max=float(heads.upper_downside_max[candidate]),
            rank_disagreement=float(heads.rank_disagreement[candidate]),
            gate_eligible=bool(heads.gate_eligible_mask[candidate]),
        )

    def select_external_candidate_index(
        self,
        sample: Mapping[str, Any],
        *,
        candidate_index: int,
        baseline_index: int | None = None,
        candidate_action_key: str | ActionKey | None = None,
    ) -> Attempt05RuntimeDecision:
        """Gate a search-selected action; fail closed to the explicit baseline."""

        action_count = _action_count(sample)
        baseline = _resolve_baseline_index(sample, baseline_index, action_count)
        raw_observation = sample.get("policy_observation")
        try:
            observation = (
                ActorObservation.from_dict(raw_observation)
                if isinstance(raw_observation, Mapping)
                else None
            )
        except (TypeError, ValueError):
            observation = None
        if observation is None or observation.street != "T1" or observation.seat != "second":
            return Attempt05RuntimeDecision(
                baseline, baseline, baseline, False, False, "unauthorized_or_invalid_observation"
            )
        try:
            evaluation = self.evaluate_candidate(
                sample,
                candidate_index=candidate_index,
                baseline_index=baseline,
                candidate_action_key=candidate_action_key,
            )
        except (TypeError, ValueError, IndexError, KeyError):
            return Attempt05RuntimeDecision(
                baseline, baseline, baseline, False, True, "fail_closed_candidate_error"
            )
        fired = bool(
            self.runtime_enabled and self.winner_frozen and evaluation.gate_eligible
        )
        return Attempt05RuntimeDecision(
            baseline_index=baseline,
            proposal_index=evaluation.candidate_index,
            selected_index=evaluation.candidate_index if fired else baseline,
            override_fired=fired,
            authorized=True,
            reason="external_candidate_override" if fired else "baseline_fallback",
        )

    def save(self, path: str | Path) -> str:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        envelope = {
            "schema": HU_M43_ATTEMPT05_ARTIFACT_SCHEMA,
            "model_schema": HU_M43_ATTEMPT05_MODEL_SCHEMA,
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
    ) -> "HuM43Attempt05Model":
        encoded = Path(path).read_bytes()
        digest = hashlib.sha256(encoded).hexdigest()
        if expected_sha256 is not None and digest != _require_sha256(expected_sha256):
            raise ValueError("Attempt05 artifact SHA-256 mismatch")
        envelope = pickle.loads(encoded)
        if not isinstance(envelope, dict) or set(envelope) != {
            "schema", "model_schema", "model"
        }:
            raise ValueError("Attempt05 artifact envelope is invalid")
        if envelope["schema"] != HU_M43_ATTEMPT05_ARTIFACT_SCHEMA or envelope[
            "model_schema"
        ] != HU_M43_ATTEMPT05_MODEL_SCHEMA:
            raise ValueError("Attempt05 artifact schema mismatch")
        model = envelope["model"]
        if not isinstance(model, cls):
            raise TypeError("Attempt05 artifact payload type mismatch")
        model.__post_init__()
        return model


def encode_deepsets_runtime_sample(
    runtime_sample: Mapping[str, Any], *, baseline_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode a safe policy sample as permutation-invariant card/action sets."""

    observation = ActorObservation.from_dict(runtime_sample["policy_observation"])
    state_tokens: list[np.ndarray] = []
    for prefix, board in (
        ("hero", observation.hero_board),
        ("opponent", observation.opponent_public_board),
    ):
        for row in ("top", "middle", "bottom"):
            for card in getattr(board, row):
                state_tokens.append(_card_token(card, f"{prefix}_{row}"))
    state_tokens.extend(_card_token(card, "dealt") for card in observation.dealt_cards)
    state_tokens.extend(
        _card_token(card, "hero_discard")
        for card in observation.hero_private_discards
    )
    actions = runtime_sample.get("actions", ())
    if (
        isinstance(baseline_index, (bool, np.bool_))
        or not isinstance(baseline_index, (int, np.integer))
        or not 0 <= int(baseline_index) < len(actions)
    ):
        raise ValueError("Attempt05 DeepSets baseline index is invalid")
    baseline_action = actions[int(baseline_index)]
    state_tokens.extend(
        _card_token(str(card), f"baseline_{row}")
        for card, row in baseline_action.get("placements", ())
    )
    state_tokens.extend(
        _card_token(str(card), "baseline_discard")
        for card in baseline_action.get("discards", ())
    )
    if not state_tokens:
        raise ValueError("Attempt05 DeepSets state has no card tokens")

    action_blocks = []
    for action in actions:
        tokens = [
            _card_token(str(card), f"action_{row}")
            for card, row in action.get("placements", ())
        ]
        tokens.extend(
            _card_token(str(card), "action_discard")
            for card in action.get("discards", ())
        )
        if len(tokens) != len(observation.dealt_cards):
            raise ValueError("Attempt05 DeepSets action does not account for every dealt card")
        # Sum pooling makes this token order irrelevant; equal length permits a
        # dense state batch and catches malformed action geometry.
        action_blocks.append(np.vstack(tokens))
    if not action_blocks:
        raise ValueError("Attempt05 DeepSets sample has no actions")
    context = _deepsets_context(observation)
    contexts = np.repeat(context.reshape(1, -1), len(action_blocks), axis=0)
    return (
        np.vstack(state_tokens).astype(np.float32, copy=False),
        np.stack(action_blocks).astype(np.float32, copy=False),
        contexts.astype(np.float32, copy=False),
    )


def build_deepsets_network(
    torch: Any, *, token_hidden_dim: int, trunk_hidden_dim: int, dropout: float
) -> Any:
    nn = torch.nn

    class _Attempt05DeepSets(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.state_phi = nn.Sequential(
                nn.Linear(DEEPSETS_CARD_TOKEN_DIM, token_hidden_dim),
                nn.ReLU(),
                nn.Linear(token_hidden_dim, token_hidden_dim),
                nn.ReLU(),
            )
            self.action_phi = nn.Sequential(
                nn.Linear(DEEPSETS_CARD_TOKEN_DIM, token_hidden_dim),
                nn.ReLU(),
                nn.Linear(token_hidden_dim, token_hidden_dim),
                nn.ReLU(),
            )
            self.trunk = nn.Sequential(
                nn.Linear(2 * token_hidden_dim + DEEPSETS_CONTEXT_DIM, trunk_hidden_dim),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(trunk_hidden_dim, trunk_hidden_dim),
                nn.ReLU(),
            )
            self.rank_head = nn.Linear(trunk_hidden_dim, 1)
            self.gain_head = nn.Linear(trunk_hidden_dim, 1)
            self.tail_head = nn.Linear(trunk_hidden_dim, 3)

        def forward(self, state_tokens: Any, action_tokens: Any, context: Any):
            state_pool = self.state_phi(state_tokens).mean(dim=0, keepdim=True)
            state_pool = state_pool.expand(action_tokens.shape[0], -1)
            action_pool = self.action_phi(action_tokens).mean(dim=1)
            hidden = self.trunk(torch.cat((state_pool, action_pool, context), dim=1))
            rank = self.rank_head(hidden).squeeze(-1)
            gain = self.gain_head(hidden).squeeze(-1)
            tails = torch.nn.functional.softplus(self.tail_head(hidden))
            return rank, gain, tails

    return _Attempt05DeepSets()


def _runtime_policy_projection(
    sample: Mapping[str, Any]
) -> tuple[dict[str, Any], ActorObservation]:
    raw_observation = sample.get("policy_observation")
    if not isinstance(raw_observation, Mapping):
        raise ValueError("Attempt05 runtime requires policy_observation")
    observation = ActorObservation.from_dict(raw_observation)
    canonical_observation = observation.to_dict()
    if dict(raw_observation) != canonical_observation:
        raise ValueError("Attempt05 policy_observation is non-canonical or has extra fields")
    legal = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    legal_by_key = index_actions_by_key(legal)
    raw_actions = sample.get("actions")
    if isinstance(raw_actions, (str, bytes)) or not isinstance(raw_actions, Sequence):
        raise ValueError("Attempt05 runtime requires an ordered legal action sequence")
    ordered = []
    seen: set[ActionKey] = set()
    for index, raw_action in enumerate(raw_actions):
        if not isinstance(raw_action, Mapping):
            raise ValueError(f"Attempt05 action {index} must be a mapping")
        key = action_key_from_payload(raw_action)
        if raw_action.get("action_key") not in (None, key.to_token()):
            raise ValueError("Attempt05 declared ActionKey disagrees")
        if key in seen or key not in legal_by_key:
            raise ValueError("Attempt05 action set is duplicate or illegal")
        seen.add(key)
        ordered.append(legal[legal_by_key[key]])
    if seen != set(legal_by_key) or len(ordered) != len(legal):
        raise ValueError("Attempt05 action list is not the complete legal action set")
    rebuilt = hu_policy_sample(
        observation.hero_board,
        observation.dealt_cards,
        ordered,
        opponent_board=observation.opponent_public_board,
        dead_cards=observation.legacy_dead_cards(),
        seat=observation.seat,
        to_act_order=observation.to_act_order,
    )
    for index, (raw_action, canonical_action) in enumerate(
        zip(raw_actions, rebuilt["actions"], strict=True)
    ):
        if raw_action.get("next_board") != canonical_action.get("next_board"):
            raise ValueError(f"Attempt05 action {index} next_board disagrees")
        if action_key(ordered[index]) != action_key_from_payload(canonical_action):
            raise AssertionError("Attempt05 regenerated ActionKey mapping changed")
    for key in ("board", "opponent_board", "dead_cards", "dealt", "seat", "to_act_order"):
        if sample.get(key) != rebuilt.get(key):
            raise ValueError(f"Attempt05 sample {key} disagrees with observation")
    rebuilt["policy_observation"] = canonical_observation
    for key in ("baseline_action_row_index", "baseline_action_key"):
        if key in sample:
            rebuilt[key] = sample[key]
    return rebuilt, observation


def _card_token(card: str, role: str) -> np.ndarray:
    if card not in ALL_CARDS or role not in _ROLE_INDEX:
        raise ValueError("Attempt05 card token is invalid")
    result = np.zeros(DEEPSETS_CARD_TOKEN_DIM, dtype=np.float32)
    result[_RANK_INDEX[card[0]]] = 1.0
    result[13 + _SUIT_INDEX[card[1]]] = 1.0
    result[17 + _ROLE_INDEX[role]] = 1.0
    return result


def _deepsets_context(observation: ActorObservation) -> np.ndarray:
    result = np.zeros(DEEPSETS_CONTEXT_DIM, dtype=np.float32)
    result[0 if observation.seat == "first" else 1] = 1.0
    result[2 + (0 if observation.to_act_order == "first" else 1)] = 1.0
    result[4 + _STREETS.index(observation.street)] = 1.0
    cursor = 10
    for board in (observation.hero_board, observation.opponent_public_board):
        result[cursor : cursor + 3] = (
            len(board.top) / 3.0,
            len(board.middle) / 5.0,
            len(board.bottom) / 5.0,
        )
        cursor += 3
    result[16] = len(observation.hero_private_discards) / 4.0
    result[17] = len(observation.dealt_cards) / 5.0
    return result


def _validate_fold_output(output: Attempt05FoldOutput, action_count: int) -> None:
    for name in (
        "rank_score",
        "gain_probability",
        "downside_p95",
        "downside_p99",
        "downside_max",
    ):
        value = np.asarray(getattr(output, name), dtype=np.float64)
        if value.shape != (action_count,) or not np.isfinite(value).all():
            raise ValueError(f"Attempt05 fold {name} output is invalid")
    gain = np.asarray(output.gain_probability)
    if np.any((gain < 0.0) | (gain > 1.0)):
        raise ValueError("Attempt05 fold gain probability is outside [0,1]")


def _regression(estimator: Any, features: np.ndarray, label: str) -> np.ndarray:
    result = np.asarray(estimator.predict(features), dtype=np.float64).reshape(-1)
    if result.shape != (features.shape[0],) or not np.isfinite(result).all():
        raise ValueError(f"Attempt05 {label} prediction is invalid")
    return result


def _probability(estimator: Any, features: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        raw = np.asarray(estimator.predict_proba(features), dtype=np.float64)
        if raw.shape == (features.shape[0], 2):
            result = raw[:, 1]
        elif raw.shape == (features.shape[0], 1):
            classes = np.asarray(getattr(estimator, "classes_", ()), dtype=np.int8)
            result = np.full(features.shape[0], float(classes[0] == 1))
        else:
            raise ValueError("Attempt05 gain predict_proba shape is invalid")
    else:
        result = _regression(estimator, features, "gain")
    if not np.isfinite(result).all() or np.any((result < 0.0) | (result > 1.0)):
        raise ValueError("Attempt05 gain probability is invalid")
    return result


def _canonical_argmax(
    values: np.ndarray, indices: np.ndarray, keys: tuple[ActionKey, ...]
) -> int:
    candidates = [int(index) for index in indices]
    best = max(float(values[index]) for index in candidates)
    return min(
        (index for index in candidates if float(values[index]) == best),
        key=lambda index: keys[index].sort_key(),
    )


def _action_count(sample: Mapping[str, Any]) -> int:
    actions = sample.get("actions", ())
    if isinstance(actions, (str, bytes)) or not isinstance(actions, Sequence) or len(actions) < 2:
        raise ValueError("Attempt05 requires at least two actions")
    return len(actions)


def _resolve_baseline_index(
    sample: Mapping[str, Any], baseline_index: int | None, action_count: int
) -> int:
    declared = sample.get("baseline_action_row_index")
    if declared is not None and (
        isinstance(declared, (bool, np.bool_))
        or not isinstance(declared, (int, np.integer))
    ):
        raise TypeError("Attempt05 declared baseline index must be an integer")
    value = declared if baseline_index is None else baseline_index
    if baseline_index is not None and declared is not None and int(declared) != int(baseline_index):
        raise ValueError("Attempt05 baseline index disagrees with sample")
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("Attempt05 baseline index must be an integer")
    result = int(value)
    if not 0 <= result < action_count:
        raise IndexError("Attempt05 baseline index is outside actions")
    baseline_key = sample.get("baseline_action_key")
    if not isinstance(baseline_key, str) or not baseline_key:
        raise ValueError("Attempt05 baseline ActionKey is required")
    if action_key_from_payload(sample["actions"][result]).to_token() != baseline_key:
        raise ValueError("Attempt05 baseline ActionKey disagrees with index")
    return result


def _resolve_candidate_index(
    sample: Mapping[str, Any],
    *,
    candidate_index: int,
    baseline_index: int,
    candidate_action_key: str | ActionKey | None,
) -> int:
    if isinstance(candidate_index, (bool, np.bool_)) or not isinstance(
        candidate_index, (int, np.integer)
    ):
        raise TypeError("Attempt05 candidate index must be an integer")
    candidate = int(candidate_index)
    if not 0 <= candidate < _action_count(sample):
        raise IndexError("Attempt05 candidate index is outside actions")
    if candidate == baseline_index:
        raise ValueError("Attempt05 external candidate must be nonbaseline")
    actual = action_key_from_payload(sample["actions"][candidate])
    if candidate_action_key is None:
        raise ValueError("Attempt05 external candidate ActionKey is required")
    expected = (
        ActionKey.from_token(candidate_action_key)
        if isinstance(candidate_action_key, str)
        else candidate_action_key
    )
    if not isinstance(expected, ActionKey) or expected != actual:
        raise ValueError("Attempt05 candidate ActionKey disagrees with index")
    return candidate


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("Attempt05 DeepSets requires PyTorch") from error
    return torch


def _require_sha256(value: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise ValueError("Attempt05 expected SHA-256 is invalid")
    return normalized


__all__ = [
    "Attempt05FoldOutput",
    "Attempt05HeadPredictions",
    "Attempt05CandidateEvaluation",
    "Attempt05RuntimeDecision",
    "DEEPSETS_CARD_TOKEN_DIM",
    "DEEPSETS_CONTEXT_DIM",
    "DeepSetsFoldPredictor",
    "HU_M43_ATTEMPT05_CONFORMAL_SCHEMA",
    "HU_M43_ATTEMPT05_HEAD_SCHEMA",
    "HU_M43_ATTEMPT05_MODEL_SCHEMA",
    "HuM43Attempt05Model",
    "LambdaRankFoldPredictor",
    "build_deepsets_network",
    "encode_deepsets_runtime_sample",
]
