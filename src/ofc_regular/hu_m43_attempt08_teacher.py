"""Attempt08 T1-second safe-promotion search teacher.

The frozen Attempt05 LambdaRank ensemble remains a candidate generator only.
It fixes eight non-baseline proposals before any hidden-card particle namespace
is opened and exposes the *raw* (non-conformal) fold-mean downside heads used
for the deterministic risk-reserve slot.

Four independent common-random-future (CRF) phases have separate jobs:

* R128 scores top-eight plus the explicit baseline and fixes an R order;
* K4 is R top-three plus one raw-risk reserve from R ranks four through eight;
* V256 traverses K4 in the already-frozen R order and promotes the first safe
  action;
* X512 may only cancel that locked V action on a catastrophe-tail breach; and
* A256 opens only after a non-baseline final output is frozen and is
  diagnostic-only.

If V does not fire, X is never opened.  Every phase owns a fresh child selector
and releases both its hidden-particle batch and selector cache before the next
phase.  The returned payload therefore retains only value vectors and opaque
belief/RNG digests, never hidden particles or opponent-private discards.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .action_key import (
    ACTION_KEY_SCHEMA,
    action_key,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from .action_space import Action, generate_turn_actions
from .hu_belief import sample_hidden_card_particles
from .hu_infoset import ActorObservation
from .hu_m43_attempt05_model import HuM43Attempt05Model
from .hu_m43_attempt06_teacher import (
    ATTEMPT06_FROZEN_MODEL_ID,
    ATTEMPT06_FROZEN_MODEL_SHA256,
    ATTEMPT06_T2_POLICY_ID,
    _require_sha256,
    _require_stage9f_p2_policies,
)
from .hu_m4_t1_teacher import (
    M4T1TeacherConfig,
    _ActionScores,
    _ChildSelector,
    _paired_delta_summary,
    _score_actions,
    _score_actions_batched,
)
from .hu_m4_teacher_contract import (
    T1_SECOND_LIVE_SCHEDULE,
    require_disjoint_root_rng_keys,
    require_t1_second_root,
)
from .hu_turn3_model import hu_policy_sample


ATTEMPT08_TEACHER_SCHEMA = (
    "hu_m43_attempt08_t1_second_r128_k4_v256_x512_a256_v1"
)
ATTEMPT08_SOLVER_ID = (
    "attempt05_lambda_top8_r128_top3_risk1_v256_x512_a256_m3_mc1_v1"
)
ATTEMPT08_FROZEN_MODEL_SHA256 = ATTEMPT06_FROZEN_MODEL_SHA256
ATTEMPT08_FROZEN_MODEL_ID = ATTEMPT06_FROZEN_MODEL_ID
ATTEMPT08_T2_POLICY_ID = ATTEMPT06_T2_POLICY_ID
ATTEMPT08_TOP_K = 8
ATTEMPT08_RERANK_SAMPLES = 128
ATTEMPT08_RERANK_TOP_K = 3
ATTEMPT08_K4 = 4
ATTEMPT08_VETO_SAMPLES = 256
ATTEMPT08_STRESS_SAMPLES = 512
ATTEMPT08_ASSESSMENT_SAMPLES = 256
ATTEMPT08_MIN_MEAN = 0.0
ATTEMPT08_MIN_P05 = -22.0
ATTEMPT08_MIN_P01 = -36.0
ATTEMPT08_MIN_VALUE = -45.0
ATTEMPT08_RNG_DOMAINS = (
    "hand_external",
    "rerank_r128",
    "veto_v256",
    "stress_x512",
    "assessment_a256",
    "child_policy",
)


@dataclass(frozen=True)
class Attempt08RankScores:
    """Candidate-only Lambda outputs required by Attempt08.

    Downside fields are raw means across the five Lambda folds.  They do not
    include Attempt05 conformal cushions and are never a runtime gate.
    """

    rank_mean: tuple[float, ...]
    rank_disagreement: tuple[float, ...]
    raw_downside_p95: tuple[float, ...]
    raw_downside_p99: tuple[float, ...]
    raw_downside_max: tuple[float, ...]
    fold_count: int = 5

    def validate(self, action_count: int) -> None:
        vectors = (
            self.rank_mean,
            self.rank_disagreement,
            self.raw_downside_p95,
            self.raw_downside_p99,
            self.raw_downside_max,
        )
        if any(len(values) != action_count for values in vectors):
            raise ValueError("Attempt08 rank output length disagrees with legal actions")
        if self.fold_count != 5:
            raise ValueError("Attempt08 requires exactly five Lambda folds")
        flat = tuple(value for values in vectors for value in values)
        if not all(math.isfinite(float(value)) for value in flat):
            raise ValueError("Attempt08 rank output contains non-finite values")
        if any(float(value) < 0.0 for value in self.rank_disagreement):
            raise ValueError("Attempt08 rank disagreement must be non-negative")
        for p95, p99, maximum in zip(
            self.raw_downside_p95,
            self.raw_downside_p99,
            self.raw_downside_max,
            strict=True,
        ):
            if float(p95) < 0.0 or float(p95) > float(p99) or float(p99) > float(maximum):
                raise ValueError(
                    "Attempt08 raw downside predictions must satisfy 0 <= p95 <= p99 <= max"
                )


class Attempt08Ranker(Protocol):
    artifact_sha256: str
    model_id: str

    def score_actions(
        self,
        observation: ActorObservation,
        actions: Sequence[Action],
        *,
        baseline_index: int,
    ) -> Attempt08RankScores: ...


@dataclass(frozen=True)
class FrozenAttempt08LambdaRanker:
    """Hash-bound, runtime-disabled Lambda folds used only for proposals."""

    model: HuM43Attempt05Model
    artifact_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_sha256",
            _require_sha256(self.artifact_sha256, name="artifact_sha256"),
        )
        if self.artifact_sha256 != ATTEMPT08_FROZEN_MODEL_SHA256:
            raise ValueError("Attempt08 requires the frozen candidate-only Lambda artifact")
        if self.model.family != "lambda_rank":
            raise ValueError("Attempt08 candidate generator must be LambdaRank")
        if self.model.model_id != ATTEMPT08_FROZEN_MODEL_ID:
            raise ValueError("Attempt08 candidate generator model_id changed")
        if self.model.runtime_enabled or self.model.winner_frozen:
            raise ValueError(
                "Attempt08 Lambda artifact must remain candidate-only and runtime-disabled"
            )

    @property
    def model_id(self) -> str:
        return self.model.model_id

    @classmethod
    def load(
        cls, path: str | Path, *, expected_sha256: str
    ) -> "FrozenAttempt08LambdaRanker":
        expected = _require_sha256(expected_sha256, name="expected_sha256")
        if expected != ATTEMPT08_FROZEN_MODEL_SHA256:
            raise ValueError("Attempt08 frozen Lambda artifact hash changed")
        model = HuM43Attempt05Model.load(path, expected_sha256=expected)
        return cls(model=model, artifact_sha256=expected)

    def score_actions(
        self,
        observation: ActorObservation,
        actions: Sequence[Action],
        *,
        baseline_index: int,
    ) -> Attempt08RankScores:
        require_t1_second_root(observation)
        if not 0 <= baseline_index < len(actions):
            raise ValueError("Attempt08 baseline index is invalid")
        sample = hu_policy_sample(
            observation.hero_board,
            observation.dealt_cards,
            actions,
            opponent_board=observation.opponent_public_board,
            dead_cards=observation.legacy_dead_cards(),
            seat=observation.seat,
            to_act_order=observation.to_act_order,
        )
        sample["policy_observation"] = observation.to_dict()
        sample["baseline_action_row_index"] = baseline_index
        sample["baseline_action_key"] = action_key(actions[baseline_index]).to_token()

        rank_folds: list[np.ndarray] = []
        tail_folds: dict[str, list[np.ndarray]] = {
            "p95": [],
            "p99": [],
            "max": [],
        }
        predictors = sorted(
            self.model.fold_predictors, key=lambda item: item.fold_index
        )
        if [int(item.fold_index) for item in predictors] != list(range(5)):
            raise ValueError("Attempt08 Lambda fold identities changed")
        for predictor in predictors:
            if predictor.family != "lambda_rank":
                raise ValueError("Attempt08 candidate generator mixes fold families")
            output = predictor.predict(sample, baseline_index=baseline_index)
            rank = np.asarray(output.rank_score, dtype=np.float64)
            tails = {
                "p95": np.asarray(output.downside_p95, dtype=np.float64),
                "p99": np.asarray(output.downside_p99, dtype=np.float64),
                "max": np.asarray(output.downside_max, dtype=np.float64),
            }
            if rank.shape != (len(actions),) or not np.isfinite(rank).all():
                raise ValueError("Attempt08 Lambda rank fold output is invalid")
            for name, values in tails.items():
                if (
                    values.shape != (len(actions),)
                    or not np.isfinite(values).all()
                    or np.any(values < 0.0)
                ):
                    raise ValueError(
                        f"Attempt08 Lambda raw downside {name} fold output is invalid"
                    )
                tail_folds[name].append(values)
            if np.any(tails["p95"] > tails["p99"]) or np.any(
                tails["p99"] > tails["max"]
            ):
                raise ValueError(
                    "Attempt08 Lambda raw downside fold heads are not monotone"
                )
            rank_folds.append(rank)

        rank_matrix = np.vstack(rank_folds)
        raw_tails = {
            name: np.mean(np.vstack(values), axis=0)
            for name, values in tail_folds.items()
        }
        # Keep the fold means raw.  The underlying predictors already enforce
        # non-negative monotone heads; validation rejects any contract drift
        # instead of silently clipping or applying conformal cushions here.
        raw_tail_matrix = np.vstack(
            (raw_tails["p95"], raw_tails["p99"], raw_tails["max"])
        )
        result = Attempt08RankScores(
            rank_mean=tuple(float(value) for value in np.mean(rank_matrix, axis=0)),
            rank_disagreement=tuple(
                float(value) for value in np.std(rank_matrix, axis=0)
            ),
            raw_downside_p95=tuple(float(value) for value in raw_tail_matrix[0]),
            raw_downside_p99=tuple(float(value) for value in raw_tail_matrix[1]),
            raw_downside_max=tuple(float(value) for value in raw_tail_matrix[2]),
            fold_count=len(predictors),
        )
        result.validate(len(actions))
        return result


@dataclass(frozen=True)
class Attempt08TeacherConfig:
    """Hard-locked Attempt08 configuration for one canonical root."""

    frozen_model_sha256: str
    hand_seed: int
    rerank_seed: int
    veto_seed: int
    stress_seed: int
    assessment_seed: int
    child_policy_seed: int
    run_id: str
    candidate_top_k: int = ATTEMPT08_TOP_K
    rerank_samples: int = ATTEMPT08_RERANK_SAMPLES
    rerank_top_k: int = ATTEMPT08_RERANK_TOP_K
    k4_size: int = ATTEMPT08_K4
    veto_samples: int = ATTEMPT08_VETO_SAMPLES
    stress_samples: int = ATTEMPT08_STRESS_SAMPLES
    assessment_samples: int = ATTEMPT08_ASSESSMENT_SAMPLES
    min_mean: float = ATTEMPT08_MIN_MEAN
    min_p05: float = ATTEMPT08_MIN_P05
    min_p01: float = ATTEMPT08_MIN_P01
    min_value: float = ATTEMPT08_MIN_VALUE
    t2_policy_id: str = ATTEMPT08_T2_POLICY_ID
    t3_candidate_samples: int = 1
    t3_evaluation_samples: int = 1
    t3_downstream_samples: int = 1
    t4_candidate_samples: int = 1
    t4_evaluation_samples: int = 1
    batch_child_selectors: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "frozen_model_sha256",
            _require_sha256(self.frozen_model_sha256, name="frozen_model_sha256"),
        )
        if self.frozen_model_sha256 != ATTEMPT08_FROZEN_MODEL_SHA256:
            raise ValueError("Attempt08 requires the frozen candidate-only Lambda artifact")
        fixed_ints = {
            "candidate_top_k": ATTEMPT08_TOP_K,
            "rerank_samples": ATTEMPT08_RERANK_SAMPLES,
            "rerank_top_k": ATTEMPT08_RERANK_TOP_K,
            "k4_size": ATTEMPT08_K4,
            "veto_samples": ATTEMPT08_VETO_SAMPLES,
            "stress_samples": ATTEMPT08_STRESS_SAMPLES,
            "assessment_samples": ATTEMPT08_ASSESSMENT_SAMPLES,
            "t3_candidate_samples": 1,
            "t3_evaluation_samples": 1,
            "t3_downstream_samples": 1,
            "t4_candidate_samples": 1,
            "t4_evaluation_samples": 1,
        }
        for name, expected in fixed_ints.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value != expected:
                raise ValueError(f"Attempt08 {name} is fixed at {expected}")
        fixed_floats = {
            "min_mean": ATTEMPT08_MIN_MEAN,
            "min_p05": ATTEMPT08_MIN_P05,
            "min_p01": ATTEMPT08_MIN_P01,
            "min_value": ATTEMPT08_MIN_VALUE,
        }
        for name, expected in fixed_floats.items():
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) != expected
            ):
                raise ValueError(f"Attempt08 {name} is fixed at {expected}")
        seed_names = (
            "hand_seed",
            "rerank_seed",
            "veto_seed",
            "stress_seed",
            "assessment_seed",
            "child_policy_seed",
        )
        for name in seed_names:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"Attempt08 {name} must be an integer")
        if len({getattr(self, name) for name in seed_names}) != len(seed_names):
            raise ValueError("Attempt08 hand/R/V/X/A/child seeds must all be distinct")
        if self.t2_policy_id != ATTEMPT08_T2_POLICY_ID:
            raise ValueError("Attempt08 T2 policy is fixed at stage9f_p2")
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("Attempt08 run_id must not be empty")
        if not isinstance(self.batch_child_selectors, bool):
            raise TypeError("batch_child_selectors must be a bool")

    def m4_config(self) -> M4T1TeacherConfig:
        return M4T1TeacherConfig(
            candidate_samples=self.rerank_samples,
            evaluation_samples=self.veto_samples,
            candidate_seed=self.rerank_seed,
            evaluation_seed=self.veto_seed,
            run_id=self.run_id,
            t2_policy_id=self.t2_policy_id,
            child_policy_seed=self.child_policy_seed,
            t3_candidate_samples=self.t3_candidate_samples,
            t3_evaluation_samples=self.t3_evaluation_samples,
            t3_downstream_samples=self.t3_downstream_samples,
            t4_candidate_samples=self.t4_candidate_samples,
            t4_evaluation_samples=self.t4_evaluation_samples,
            batch_child_selectors=self.batch_child_selectors,
        )


@dataclass(frozen=True)
class _PhaseResult:
    scores: tuple[_ActionScores, ...]
    belief_digest: str
    rng_keys: tuple[str, ...]
    child_information_set_count: int


def _resolve_baseline(actions: Sequence[Action], token: str) -> tuple[int, str]:
    matches = [
        index
        for index, action in enumerate(actions)
        if action_key(action).to_token() == token
    ]
    if len(matches) != 1:
        raise ValueError("Attempt08 explicit baseline ActionKey is not uniquely legal")
    return matches[0], token


def _raw_digest(values: Sequence[float]) -> str:
    return hashlib.sha256(
        json.dumps(
            [float(value) for value in values],
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _validate_scores(
    scores: Sequence[_ActionScores],
    *,
    action_count: int,
    sample_count: int,
    phase: str,
) -> tuple[_ActionScores, ...]:
    checked = tuple(scores)
    if len(checked) != action_count:
        raise ValueError(f"Attempt08 {phase} scorer returned the wrong action count")
    for score in checked:
        if len(score.values) != sample_count:
            raise ValueError(f"Attempt08 {phase} scorer returned the wrong sample count")
        if not all(math.isfinite(float(value)) for value in score.values):
            raise ValueError(f"Attempt08 {phase} scorer emitted a non-finite value")
    return checked


def _paired_row(candidate: _ActionScores, baseline: _ActionScores) -> dict[str, Any]:
    deltas = [
        float(candidate_value - baseline_value)
        for candidate_value, baseline_value in zip(
            candidate.values, baseline.values, strict=True
        )
    ]
    if not deltas or not all(math.isfinite(value) for value in deltas):
        raise ValueError("Attempt08 paired deltas must be finite and non-empty")
    return {
        "paired_delta_vs_baseline": _paired_delta_summary(candidate, baseline),
        "raw_paired_deltas_vs_baseline": deltas,
        "raw_paired_deltas_sha256": _raw_digest(deltas),
        # The underlying marginal action-value vector is deliberately not
        # retained.  Its digest is opaque provenance, not independently
        # recomputable evidence; all decision-relevant paired values below are
        # retained and independently revalidated.
        "opaque_action_values_sha256": _raw_digest(candidate.values),
    }


def _phase_actions(
    actions: Sequence[Action],
    scores: Sequence[_ActionScores],
    *,
    baseline_position: int,
) -> list[dict[str, Any]]:
    if len(actions) != len(scores) or not 0 <= baseline_position < len(actions):
        raise ValueError("Attempt08 phase action mapping is invalid")
    baseline = scores[baseline_position]
    return [
        {
            "phase_position": position,
            "action_key": action_key(action).to_token(),
            "mean": score.mean,
            "standard_error": score.standard_error,
            "is_explicit_baseline": position == baseline_position,
            **_paired_row(score, baseline),
        }
        for position, (action, score) in enumerate(zip(actions, scores, strict=True))
    ]


def _mapping_payload(actions: Sequence[Action]) -> dict[str, Any]:
    return {
        "action_count": len(actions),
        "action_keys": [action_key(action).to_token() for action in actions],
        "action_set_digest": legal_action_set_digest(actions),
        "action_order_digest": ordered_action_mapping_digest(actions),
    }


def _veto_checks(summary: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "mean_gt_0": float(summary["mean"]) > ATTEMPT08_MIN_MEAN,
        "p05_ge_neg22": float(summary["p05"]) >= ATTEMPT08_MIN_P05,
        "p01_ge_neg36": float(summary["p01"]) >= ATTEMPT08_MIN_P01,
        "min_ge_neg45": float(summary["min"]) >= ATTEMPT08_MIN_VALUE,
    }


def _risk_components(rank_scores: Attempt08RankScores, index: int) -> dict[str, float]:
    return {
        "p95_over_22": float(rank_scores.raw_downside_p95[index]) / 22.0,
        "p99_over_36": float(rank_scores.raw_downside_p99[index]) / 36.0,
        "max_over_45": float(rank_scores.raw_downside_max[index]) / 45.0,
    }


def _score_phase(
    observation: ActorObservation,
    actions: Sequence[Action],
    *,
    phase: str,
    seed: int,
    sample_count: int,
    config: Attempt08TeacherConfig,
    t2_policies: Mapping[str, object],
    prior_rng_keys: Sequence[Sequence[str]],
    library: Any | None,
) -> _PhaseResult:
    """Score one phase without retaining its particles or child cache."""

    batch = sample_hidden_card_particles(
        observation,
        base_seed=seed,
        run_id=f"{config.run_id}:{phase}",
        sample_count=sample_count,
    )
    batch.validate_against(observation)
    keys = tuple(particle.rng_key_digest for particle in batch.particles)
    if len(keys) != sample_count or len(set(keys)) != sample_count:
        raise ValueError(f"Attempt08 {phase} particle RNG keys are invalid")
    for existing in prior_rng_keys:
        require_disjoint_root_rng_keys(existing, keys)
    belief_digest = batch.digest()
    selector = _ChildSelector(
        t2_policies=t2_policies,
        config=config.m4_config(),
        library=library,
    )
    scorer = _score_actions_batched if config.batch_child_selectors else _score_actions
    scores = _validate_scores(
        scorer(observation, actions, batch, selector),
        action_count=len(actions),
        sample_count=sample_count,
        phase=phase,
    )
    child_count = len(selector.cache)
    # No hidden world, particle batch, or child observation may survive into
    # the next phase.  This also bounds peak RSS on c4-standard-4 workers.
    selector.cache.clear()
    del selector
    del batch
    gc.collect()
    return _PhaseResult(
        scores=scores,
        belief_digest=belief_digest,
        rng_keys=keys,
        child_information_set_count=child_count,
    )


def evaluate_attempt08_t1_second(
    observation: ActorObservation,
    *,
    baseline_action_key: str,
    ranker: Attempt08Ranker,
    t2_policies: Mapping[str, object],
    config: Attempt08TeacherConfig,
    library: Any | None = None,
) -> dict[str, Any]:
    """Evaluate the frozen Attempt08 single arm without activating a policy."""

    require_t1_second_root(observation)
    if ranker.artifact_sha256 != config.frozen_model_sha256:
        raise ValueError("Attempt08 ranker artifact hash disagrees with config")
    _require_stage9f_p2_policies(t2_policies)

    legal_actions = tuple(
        generate_turn_actions(observation.hero_board, observation.dealt_cards)
    )
    if not legal_actions:
        raise ValueError("Attempt08 root has no legal actions")
    baseline_index, baseline_token = _resolve_baseline(
        legal_actions, baseline_action_key
    )

    # Candidate generation is complete before the first particle namespace.
    rank_scores = ranker.score_actions(
        observation, legal_actions, baseline_index=baseline_index
    )
    rank_scores.validate(len(legal_actions))
    nonbaseline_indices = [
        index for index in range(len(legal_actions)) if index != baseline_index
    ]
    if len(nonbaseline_indices) < config.candidate_top_k:
        raise ValueError("Attempt08 root has fewer than eight nonbaseline actions")
    nonbaseline_indices.sort(
        key=lambda index: (
            -float(rank_scores.rank_mean[index]),
            action_key(legal_actions[index]).sort_key(),
        )
    )
    top_indices = tuple(nonbaseline_indices[: config.candidate_top_k])
    proposal_indices = (*top_indices, baseline_index)
    proposal_actions = tuple(legal_actions[index] for index in proposal_indices)
    if len({action_key(action) for action in proposal_actions}) != 9:
        raise AssertionError("Attempt08 proposal set must be top8 plus baseline once")
    proposal_baseline_position = len(proposal_actions) - 1

    belief_digests: dict[str, str] = {}
    rng_key_digests: dict[str, list[str]] = {}
    child_counts: dict[str, int] = {}
    prior_rng_keys: list[tuple[str, ...]] = []

    def remember(phase: str, result: _PhaseResult) -> None:
        belief_digests[phase] = result.belief_digest
        rng_key_digests[phase] = list(result.rng_keys)
        child_counts[phase] = result.child_information_set_count
        prior_rng_keys.append(result.rng_keys)

    rerank_result = _score_phase(
        observation,
        proposal_actions,
        phase="rerank_r128",
        seed=config.rerank_seed,
        sample_count=config.rerank_samples,
        config=config,
        t2_policies=t2_policies,
        prior_rng_keys=prior_rng_keys,
        library=library,
    )
    remember("rerank_r128", rerank_result)
    rerank_rows = _phase_actions(
        proposal_actions,
        rerank_result.scores,
        baseline_position=proposal_baseline_position,
    )
    rerank_order = sorted(
        range(config.candidate_top_k),
        key=lambda position: (
            -float(rerank_rows[position]["paired_delta_vs_baseline"]["mean"]),
            action_key(proposal_actions[position]).sort_key(),
        ),
    )
    top3_positions = tuple(rerank_order[: config.rerank_top_k])
    reserve_pool = tuple(rerank_order[config.rerank_top_k :])
    if len(reserve_pool) != 5:
        raise AssertionError("Attempt08 reserve pool must be R ranks four through eight")

    reserve_risk: dict[int, tuple[float, dict[str, float]]] = {}
    for position in reserve_pool:
        original_index = proposal_indices[position]
        components = _risk_components(rank_scores, original_index)
        reserve_risk[position] = (max(components.values()), components)
    reserve_position = min(
        reserve_pool,
        key=lambda position: (
            reserve_risk[position][0],
            action_key(proposal_actions[position]).sort_key(),
        ),
    )
    k4_positions_unordered = {*top3_positions, reserve_position}
    if len(k4_positions_unordered) != config.k4_size:
        raise AssertionError("Attempt08 K4 membership must contain four unique actions")
    # V traversal is the original R order, not risk order or membership order.
    k4_positions = tuple(
        position for position in rerank_order if position in k4_positions_unordered
    )
    k4_actions = tuple(proposal_actions[position] for position in k4_positions)
    veto_actions = (*k4_actions, legal_actions[baseline_index])
    veto_baseline_position = len(veto_actions) - 1

    veto_result = _score_phase(
        observation,
        veto_actions,
        phase="veto_v256",
        seed=config.veto_seed,
        sample_count=config.veto_samples,
        config=config,
        t2_policies=t2_policies,
        prior_rng_keys=prior_rng_keys,
        library=library,
    )
    remember("veto_v256", veto_result)
    veto_rows = _phase_actions(
        veto_actions,
        veto_result.scores,
        baseline_position=veto_baseline_position,
    )
    veto_checks = [
        _veto_checks(row["paired_delta_vs_baseline"])
        for row in veto_rows[: config.k4_size]
    ]
    passing_positions = [
        position for position, checks in enumerate(veto_checks) if all(checks.values())
    ]
    veto_selected_position = passing_positions[0] if passing_positions else None
    veto_selected_action = (
        veto_actions[veto_selected_position]
        if veto_selected_position is not None
        else None
    )
    veto_selected_token = (
        action_key(veto_selected_action).to_token()
        if veto_selected_action is not None
        else None
    )

    stress_opened = veto_selected_action is not None
    stress_actions: tuple[Action, ...] = ()
    stress_rows: list[dict[str, Any]] = []
    stress_summary: dict[str, Any] | None = None
    stress_raw: list[float] = []
    stress_raw_sha256: str | None = None
    stress_pass: bool | None = None
    stress_cancelled = False
    if stress_opened:
        stress_actions = (veto_selected_action, legal_actions[baseline_index])
        stress_result = _score_phase(
            observation,
            stress_actions,
            phase="stress_x512",
            seed=config.stress_seed,
            sample_count=config.stress_samples,
            config=config,
            t2_policies=t2_policies,
            prior_rng_keys=prior_rng_keys,
            library=library,
        )
        remember("stress_x512", stress_result)
        stress_rows = _phase_actions(
            stress_actions, stress_result.scores, baseline_position=1
        )
        stress_summary = dict(stress_rows[0]["paired_delta_vs_baseline"])
        stress_raw = list(stress_rows[0]["raw_paired_deltas_vs_baseline"])
        stress_raw_sha256 = str(stress_rows[0]["raw_paired_deltas_sha256"])
        stress_pass = float(stress_summary["min"]) >= ATTEMPT08_MIN_VALUE
        stress_cancelled = not stress_pass

    final_action = (
        veto_selected_action
        if veto_selected_action is not None and stress_pass is True
        else legal_actions[baseline_index]
    )
    final_token = action_key(final_action).to_token()
    override_fired = final_token != baseline_token
    if veto_selected_action is None:
        fallback_reason = "no_v256_candidate_passed"
    elif stress_cancelled:
        fallback_reason = "x512_catastrophe_cancel"
    else:
        fallback_reason = None

    # Final output is immutable before A can open.  A is fire-only: a baseline
    # final output retains no assessment action, RNG namespace, or raw vector.
    assessment_opened = override_fired
    assessment_actions: tuple[Action, ...] = ()
    assessment_rows: list[dict[str, Any]] = []
    assessment_summary: dict[str, Any] | None = None
    assessment_raw: list[float] = []
    assessment_raw_sha256: str | None = None
    assessment_best_token: str | None = None
    if assessment_opened:
        assessment_actions = (final_action, legal_actions[baseline_index])
        assessment_result = _score_phase(
            observation,
            assessment_actions,
            phase="assessment_a256",
            seed=config.assessment_seed,
            sample_count=config.assessment_samples,
            config=config,
            t2_policies=t2_policies,
            prior_rng_keys=prior_rng_keys,
            library=library,
        )
        remember("assessment_a256", assessment_result)
        assessment_rows = _phase_actions(
            assessment_actions,
            assessment_result.scores,
            baseline_position=1,
        )
        assessment_selected_row = assessment_rows[0]
        assessment_summary = dict(
            assessment_selected_row["paired_delta_vs_baseline"]
        )
        assessment_raw = list(
            assessment_selected_row["raw_paired_deltas_vs_baseline"]
        )
        assessment_raw_sha256 = str(
            assessment_selected_row["raw_paired_deltas_sha256"]
        )
        assessment_means = tuple(score.mean for score in assessment_result.scores)
        best_mean = max(assessment_means)
        assessment_best_position = min(
            [
                position
                for position, value in enumerate(assessment_means)
                if value == best_mean
            ],
            key=lambda position: action_key(assessment_actions[position]).sort_key(),
        )
        assessment_best_token = action_key(
            assessment_actions[assessment_best_position]
        ).to_token()

    legal_rows = []
    top_set = set(top_indices)
    for index, action in enumerate(legal_actions):
        components = _risk_components(rank_scores, index)
        legal_rows.append(
            {
                "original_legal_index": index,
                "action_key": action_key(action).to_token(),
                "legal": True,
                "illegal_action_masked": False,
                "model_rank_mean": float(rank_scores.rank_mean[index]),
                "model_rank_disagreement": float(
                    rank_scores.rank_disagreement[index]
                ),
                "raw_predicted_downside_p95": float(
                    rank_scores.raw_downside_p95[index]
                ),
                "raw_predicted_downside_p99": float(
                    rank_scores.raw_downside_p99[index]
                ),
                "raw_predicted_downside_max": float(
                    rank_scores.raw_downside_max[index]
                ),
                "normalized_raw_risk_score": max(components.values()),
                "in_learned_top8": index in top_set,
                "is_explicit_baseline": index == baseline_index,
            }
        )

    payload: dict[str, Any] = {
        "status": "ok",
        "schema": ATTEMPT08_TEACHER_SCHEMA,
        "solver_id": ATTEMPT08_SOLVER_ID,
        "street": "T1",
        "seat": "second",
        "to_act_order": "second",
        "observation_fingerprint": observation.fingerprint(),
        "policy_observation": observation.to_dict(),
        "action_key_schema": ACTION_KEY_SCHEMA,
        "frozen_candidate_generator": {
            "family": "lambda_rank",
            "model_id": ranker.model_id,
            "artifact_sha256": ranker.artifact_sha256,
            "purpose": "candidate_generation_and_raw_risk_reserve_only",
            "raw_downside_source": "mean_of_five_candidate_only_lambda_folds",
            "fold_count": rank_scores.fold_count,
            "conformal_cushions_applied": False,
            "runtime_authorized": False,
            "profile_runtime_feature": False,
        },
        "legal_action_mapping": _mapping_payload(legal_actions),
        "legal_action_mask": [True for _ in legal_actions],
        "illegal_action_mask": [False for _ in legal_actions],
        "legal_actions": legal_rows,
        "baseline_action_key": baseline_token,
        "baseline_original_legal_index": baseline_index,
        "learned_top8_original_legal_indices": list(top_indices),
        "learned_top8_action_keys": [
            action_key(legal_actions[index]).to_token() for index in top_indices
        ],
        "proposal_mapping": _mapping_payload(proposal_actions),
        "rerank": {
            **_mapping_payload(proposal_actions),
            "sample_count": ATTEMPT08_RERANK_SAMPLES,
            "prefix_semantics": "full_R128_only",
            "common_random_futures": True,
            "nonbaseline_order_rule": "paired_mean_desc_then_ActionKey",
            "ordered_nonbaseline_proposal_positions": list(rerank_order),
            "ordered_nonbaseline_action_keys": [
                action_key(proposal_actions[position]).to_token()
                for position in rerank_order
            ],
            "actions": rerank_rows,
        },
        "k4": {
            **_mapping_payload(k4_actions),
            "selection_rule": (
                "R_top3_plus_ranks4_8_min_max_raw_p95_over22_"
                "raw_p99_over36_raw_max_over45_then_ActionKey"
            ),
            "top3_rerank_positions": list(top3_positions),
            "top3_action_keys": [
                action_key(proposal_actions[position]).to_token()
                for position in top3_positions
            ],
            "risk_reserve_rerank_position": reserve_position,
            "risk_reserve_r_rank": rerank_order.index(reserve_position) + 1,
            "risk_reserve_action_key": action_key(
                proposal_actions[reserve_position]
            ).to_token(),
            "risk_reserve_raw_components": reserve_risk[reserve_position][1],
            "risk_reserve_normalized_score": reserve_risk[reserve_position][0],
            "veto_traversal_rule": "original_R_order",
            "veto_traversal_rerank_positions": list(k4_positions),
        },
        "veto": {
            **_mapping_payload(veto_actions),
            "opened": True,
            "sample_count": ATTEMPT08_VETO_SAMPLES,
            "prefix_semantics": "full_V256_only",
            "common_random_futures": True,
            "scope": "frozen_K4_in_R_order_plus_explicit_baseline",
            "thresholds": {
                "mean_strictly_greater_than": ATTEMPT08_MIN_MEAN,
                "p05_at_least": ATTEMPT08_MIN_P05,
                "p01_at_least": ATTEMPT08_MIN_P01,
                "min_at_least": ATTEMPT08_MIN_VALUE,
            },
            "traversal_action_keys": [
                action_key(action).to_token() for action in k4_actions
            ],
            "checks_by_traversal_position": veto_checks,
            "first_passing_traversal_position": veto_selected_position,
            "first_passing_action_key": veto_selected_token,
            "selection_rule": "first_safe_in_frozen_R_order_else_baseline",
            "actions": veto_rows,
        },
        "stress": {
            **_mapping_payload(stress_actions),
            "opened": stress_opened,
            "sample_count": ATTEMPT08_STRESS_SAMPLES if stress_opened else 0,
            "prefix_semantics": "full_X512_only_if_V_fires",
            "common_random_futures": stress_opened,
            "scope": (
                "locked_V_candidate_plus_explicit_baseline"
                if stress_opened
                else "not_opened_because_V_did_not_fire"
            ),
            "locked_candidate_action_key": veto_selected_token,
            "min_at_least": ATTEMPT08_MIN_VALUE,
            "cancel_only": True,
            "may_promote_or_rerank": False,
            "pass": stress_pass,
            "cancelled": stress_cancelled,
            "paired_delta_vs_baseline": stress_summary,
            "raw_paired_deltas_vs_baseline": stress_raw,
            "raw_paired_deltas_sha256": stress_raw_sha256,
            "actions": stress_rows,
        },
        "decision": {
            "veto_selected_action_key": veto_selected_token,
            "veto_override_fired": veto_selected_token is not None,
            "stress_opened": stress_opened,
            "stress_cancelled": stress_cancelled,
            "final_selected_action_key": final_token,
            "override_fired": override_fired,
            "exact_baseline_fallback": not override_fired,
            "fallback_reason": fallback_reason,
            "second_candidate_promotion_after_stress_cancel_allowed": False,
            "frozen_before_assessment_namespace_open": True,
        },
        "assessment": {
            **_mapping_payload(assessment_actions),
            "opened": assessment_opened,
            "sample_count": (
                ATTEMPT08_ASSESSMENT_SAMPLES if assessment_opened else 0
            ),
            "prefix_semantics": "full_A256_only_if_final_output_is_nonbaseline",
            "common_random_futures": assessment_opened,
            "scope": (
                "locked_nonbaseline_final_plus_explicit_baseline"
                if assessment_opened
                else "not_opened_because_final_output_is_baseline"
            ),
            "execution_condition": "final_output_is_nonbaseline_after_X512",
            "skipped_when_final_output_is_baseline": True,
            "diagnostics_only": True,
            "can_rerank_or_gate": False,
            "decision_frozen_before_namespace_open": True,
            "locked_final_action_key": final_token,
            "sample_best_action_key": assessment_best_token,
            "paired_delta_vs_baseline": assessment_summary,
            "raw_paired_deltas_vs_baseline": assessment_raw,
            "raw_paired_deltas_sha256": assessment_raw_sha256,
            "actions": assessment_rows,
        },
        "belief_digests": belief_digests,
        "rng_key_digests": rng_key_digests,
        "phase_child_information_set_counts": child_counts,
        "sample_independence": (
            "pairwise_disjoint_R128_V256_optional_X512_optional_fire_A256_particle_rng_keys"
        ),
        "seed_domain_provenance": {
            "domain_order": list(ATTEMPT08_RNG_DOMAINS),
            "hand_external": config.hand_seed,
            "rerank_r128": config.rerank_seed,
            "veto_v256": config.veto_seed,
            "stress_x512": config.stress_seed,
            "assessment_a256": config.assessment_seed,
            "child_policy": config.child_policy_seed,
            "all_six_base_seeds_pairwise_distinct": True,
            "hand_sampled_inside_teacher": False,
        },
        "root_selection_lock": (
            "top8_before_R128_then_K4_before_V256_then_V_choice_before_"
            "optional_X512_then_final_before_optional_fire_diagnostic_A256"
        ),
        "search_config": {
            "learned_nonbaseline_top_k": config.candidate_top_k,
            "baseline_added_exactly_once": True,
            "rerank_samples": config.rerank_samples,
            "rerank_top_k": config.rerank_top_k,
            "risk_reserve_count": 1,
            "k4_size": config.k4_size,
            "veto_samples": config.veto_samples,
            "stress_samples": config.stress_samples,
            "assessment_samples": config.assessment_samples,
            "hand_seed": config.hand_seed,
            "rerank_seed": config.rerank_seed,
            "veto_seed": config.veto_seed,
            "stress_seed": config.stress_seed,
            "assessment_seed": config.assessment_seed,
            "child_policy_seed": config.child_policy_seed,
            "run_id": config.run_id,
            "batch_child_selectors": config.batch_child_selectors,
            "candidate_and_rerank_tie_break": "ActionKey",
            "raw_risk_reserve_tie_break": "ActionKey",
        },
        "continuation_policy": {
            "t2_policy_id": ATTEMPT08_T2_POLICY_ID,
            "t2_resolution": "explicit_profile_never_current",
            "t2_runtime_status": "p2_fixed",
            "t3_selector": "m3_rust_evaluate_t3",
            "t3_candidate_samples": 1,
            "t3_evaluation_samples": 1,
            "t3_downstream_samples": 1,
            "hypothetical_t4_selector": "m3_rust_evaluate_t4",
            "hypothetical_t4_candidate_samples": 1,
            "hypothetical_t4_evaluation_samples": 1,
            "real_live_t4_exact_unchanged": True,
            "fresh_selector_per_phase": True,
            "child_cache_released_after_each_phase": True,
        },
        "live_schedule": [
            {
                "seat": step.seat,
                "street": step.street,
                "draw_offset": step.draw_offset,
            }
            for step in T1_SECOND_LIVE_SCHEDULE
        ],
        "memory_retention": {
            "retained_particle_batches": 0,
            "retained_hidden_particle_payload": False,
            "retained_child_selector_caches": 0,
            "output_contains_only_value_vectors_and_opaque_digests": True,
        },
        "teacher_value_status": "diagnostic_not_match_EV",
        "runtime_gate_allowed": False,
        "profile_activation_allowed": False,
        "current_profile_resolved": False,
        "development_only": True,
    }
    validate_attempt08_teacher_output(
        observation,
        baseline_action_key=baseline_token,
        payload=payload,
        config=config,
    )
    return payload


def _payload_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Attempt08 {label} must be a mapping")
    return value


def _payload_sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise ValueError(f"Attempt08 {label} must be a sequence")
    return value


def _payload_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Attempt08 {label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Attempt08 {label} must be finite")
    return result


def _require_exact_keys(
    value: Mapping[str, Any], expected: Sequence[str] | set[str] | frozenset[str], label: str
) -> None:
    expected_set = set(expected)
    if set(value) != expected_set:
        missing = sorted(expected_set - set(value))
        extra = sorted(set(value) - expected_set)
        raise ValueError(
            f"Attempt08 {label} fields changed; missing={missing}, extra={extra}"
        )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_payload_mapping(
    value: Mapping[str, Any],
    actions: Sequence[Action],
    *,
    label: str,
    exact: bool = True,
) -> None:
    expected = _mapping_payload(actions)
    if exact:
        _require_exact_keys(value, set(expected), f"{label} mapping")
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"Attempt08 {label} {key} changed")


def _validate_phase_payload(
    value: Mapping[str, Any],
    actions: Sequence[Action],
    *,
    baseline_position: int,
    sample_count: int,
    label: str,
) -> list[dict[str, Any]]:
    """Recompute a phase's paired vectors, digests, and summaries."""

    _validate_payload_mapping(value, actions, label=label, exact=False)
    if value.get("sample_count") != sample_count:
        raise ValueError(f"Attempt08 {label} sample count changed")
    raw_rows = _payload_sequence(value.get("actions"), f"{label}.actions")
    if len(raw_rows) != len(actions) or not 0 <= baseline_position < len(actions):
        raise ValueError(f"Attempt08 {label} action rows changed")
    normalized: list[dict[str, Any]] = []
    baseline_mean: float | None = None
    for position, (action, raw_row) in enumerate(
        zip(actions, raw_rows, strict=True)
    ):
        row = _payload_mapping(raw_row, f"{label}.actions[{position}]")
        _require_exact_keys(
            row,
            {
                "phase_position",
                "action_key",
                "mean",
                "standard_error",
                "is_explicit_baseline",
                "paired_delta_vs_baseline",
                "raw_paired_deltas_vs_baseline",
                "raw_paired_deltas_sha256",
                "opaque_action_values_sha256",
            },
            f"{label}.actions[{position}]",
        )
        token = action_key(action).to_token()
        if (
            row.get("phase_position") != position
            or row.get("action_key") != token
            or row.get("is_explicit_baseline") is not (
                position == baseline_position
            )
        ):
            raise ValueError(f"Attempt08 {label} action mapping changed")
        mean = _payload_finite(row.get("mean"), f"{label}[{position}].mean")
        standard_error = _payload_finite(
            row.get("standard_error"), f"{label}[{position}].standard_error"
        )
        if standard_error < 0.0:
            raise ValueError(f"Attempt08 {label} standard error is negative")
        raw_values = _payload_sequence(
            row.get("raw_paired_deltas_vs_baseline"),
            f"{label}[{position}].raw_paired_deltas",
        )
        if len(raw_values) != sample_count:
            raise ValueError(f"Attempt08 {label} raw paired count changed")
        raw = [
            _payload_finite(value, f"{label}[{position}].raw[{index}]")
            for index, value in enumerate(raw_values)
        ]
        if row.get("raw_paired_deltas_sha256") != _raw_digest(raw):
            raise ValueError(f"Attempt08 {label} raw paired digest changed")
        if not _is_sha256(row.get("opaque_action_values_sha256")):
            raise ValueError(f"Attempt08 {label} opaque action-value digest changed")
        zero = _ActionScores(tuple(0.0 for _ in raw))
        expected_summary = _paired_delta_summary(_ActionScores(tuple(raw)), zero)
        summary = _payload_mapping(
            row.get("paired_delta_vs_baseline"),
            f"{label}[{position}].paired_delta_vs_baseline",
        )
        _require_exact_keys(
            summary,
            set(expected_summary),
            f"{label}[{position}].paired_delta_vs_baseline",
        )
        if summary != expected_summary:
            raise ValueError(f"Attempt08 {label} paired summary changed")
        if position == baseline_position:
            baseline_mean = mean
            if any(value != 0.0 for value in raw):
                raise ValueError(f"Attempt08 {label} baseline does not cancel exactly")
        normalized.append(
            {
                "action_key": token,
                "mean": mean,
                "raw": raw,
                "summary": expected_summary,
            }
        )
    if baseline_mean is None:  # pragma: no cover - guarded by position bounds
        raise AssertionError("Attempt08 phase baseline row disappeared")
    for position, row in enumerate(normalized):
        expected_delta_mean = float(row["mean"]) - baseline_mean
        if not math.isclose(
            expected_delta_mean,
            float(row["summary"]["mean"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Attempt08 {label} mean/paired mean changed at {position}")
    return normalized


def validate_attempt08_teacher_output(
    observation: ActorObservation,
    *,
    baseline_action_key: str,
    payload: Mapping[str, Any],
    config: Attempt08TeacherConfig,
) -> dict[str, Any]:
    """Validate one in-memory teacher row and return normalized selectors.

    The runner and one-shot selector can use this function instead of copying
    the teacher's mapping, RNG, and conditional-X invariants.
    """

    require_t1_second_root(observation)
    _require_exact_keys(
        payload,
        {
            "status",
            "schema",
            "solver_id",
            "street",
            "seat",
            "to_act_order",
            "observation_fingerprint",
            "policy_observation",
            "action_key_schema",
            "frozen_candidate_generator",
            "legal_action_mapping",
            "legal_action_mask",
            "illegal_action_mask",
            "legal_actions",
            "baseline_action_key",
            "baseline_original_legal_index",
            "learned_top8_original_legal_indices",
            "learned_top8_action_keys",
            "proposal_mapping",
            "rerank",
            "k4",
            "veto",
            "stress",
            "decision",
            "assessment",
            "belief_digests",
            "rng_key_digests",
            "phase_child_information_set_counts",
            "sample_independence",
            "seed_domain_provenance",
            "root_selection_lock",
            "search_config",
            "continuation_policy",
            "live_schedule",
            "memory_retention",
            "teacher_value_status",
            "runtime_gate_allowed",
            "profile_activation_allowed",
            "current_profile_resolved",
            "development_only",
        },
        "teacher top-level",
    )
    if (
        payload.get("schema") != ATTEMPT08_TEACHER_SCHEMA
        or payload.get("solver_id") != ATTEMPT08_SOLVER_ID
        or payload.get("status") != "ok"
        or payload.get("street") != "T1"
        or payload.get("seat") != "second"
        or payload.get("to_act_order") != "second"
        or payload.get("action_key_schema") != ACTION_KEY_SCHEMA
    ):
        raise ValueError("Attempt08 teacher identity changed")
    if payload.get("observation_fingerprint") != observation.fingerprint():
        raise ValueError("Attempt08 teacher observation fingerprint changed")
    if payload.get("policy_observation") != observation.to_dict():
        raise ValueError("Attempt08 teacher policy observation changed")
    if payload.get("baseline_action_key") != baseline_action_key:
        raise ValueError("Attempt08 teacher baseline ActionKey changed")

    legal_actions = tuple(
        generate_turn_actions(observation.hero_board, observation.dealt_cards)
    )
    baseline_index, _baseline_token = _resolve_baseline(
        legal_actions, baseline_action_key
    )
    legal_mapping = _payload_mapping(
        payload.get("legal_action_mapping"), "legal_action_mapping"
    )
    _validate_payload_mapping(legal_mapping, legal_actions, label="legal")
    if payload.get("legal_action_mask") != [True] * len(legal_actions):
        raise ValueError("Attempt08 legal action mask changed")
    if payload.get("illegal_action_mask") != [False] * len(legal_actions):
        raise ValueError("Attempt08 illegal action mask changed")
    if payload.get("baseline_original_legal_index") != baseline_index:
        raise ValueError("Attempt08 baseline legal index changed")

    generator = _payload_mapping(
        payload.get("frozen_candidate_generator"), "frozen_candidate_generator"
    )
    _require_exact_keys(
        generator,
        {
            "family",
            "model_id",
            "artifact_sha256",
            "purpose",
            "raw_downside_source",
            "fold_count",
            "conformal_cushions_applied",
            "runtime_authorized",
            "profile_runtime_feature",
        },
        "frozen_candidate_generator",
    )
    if (
        generator.get("family") != "lambda_rank"
        or generator.get("model_id") != ATTEMPT08_FROZEN_MODEL_ID
        or generator.get("artifact_sha256") != config.frozen_model_sha256
        or generator.get("purpose")
        != "candidate_generation_and_raw_risk_reserve_only"
        or generator.get("raw_downside_source")
        != "mean_of_five_candidate_only_lambda_folds"
        or generator.get("fold_count") != 5
        or generator.get("conformal_cushions_applied") is not False
        or generator.get("runtime_authorized") is not False
        or generator.get("profile_runtime_feature") is not False
    ):
        raise ValueError("Attempt08 frozen candidate-generator contract changed")

    raw_legal_rows = _payload_sequence(payload.get("legal_actions"), "legal_actions")
    if len(raw_legal_rows) != len(legal_actions):
        raise ValueError("Attempt08 legal action rows changed")
    legal_rank_mean: list[float] = []
    legal_tails: list[tuple[float, float, float]] = []
    for index, (action, raw_row) in enumerate(
        zip(legal_actions, raw_legal_rows, strict=True)
    ):
        row = _payload_mapping(raw_row, f"legal_actions[{index}]")
        _require_exact_keys(
            row,
            {
                "original_legal_index",
                "action_key",
                "legal",
                "illegal_action_masked",
                "model_rank_mean",
                "model_rank_disagreement",
                "raw_predicted_downside_p95",
                "raw_predicted_downside_p99",
                "raw_predicted_downside_max",
                "normalized_raw_risk_score",
                "in_learned_top8",
                "is_explicit_baseline",
            },
            f"legal_actions[{index}]",
        )
        if (
            row.get("original_legal_index") != index
            or row.get("action_key") != action_key(action).to_token()
            or row.get("legal") is not True
            or row.get("illegal_action_masked") is not False
            or row.get("is_explicit_baseline") is not (index == baseline_index)
        ):
            raise ValueError("Attempt08 legal action row mapping changed")
        rank_mean = _payload_finite(
            row.get("model_rank_mean"), f"legal_actions[{index}].model_rank_mean"
        )
        disagreement = _payload_finite(
            row.get("model_rank_disagreement"),
            f"legal_actions[{index}].model_rank_disagreement",
        )
        p95 = _payload_finite(
            row.get("raw_predicted_downside_p95"),
            f"legal_actions[{index}].raw_p95",
        )
        p99 = _payload_finite(
            row.get("raw_predicted_downside_p99"),
            f"legal_actions[{index}].raw_p99",
        )
        maximum = _payload_finite(
            row.get("raw_predicted_downside_max"),
            f"legal_actions[{index}].raw_max",
        )
        if disagreement < 0.0 or p95 < 0.0 or p95 > p99 or p99 > maximum:
            raise ValueError("Attempt08 legal raw model heads changed")
        expected_risk = max(p95 / 22.0, p99 / 36.0, maximum / 45.0)
        if not math.isclose(
            _payload_finite(
                row.get("normalized_raw_risk_score"),
                f"legal_actions[{index}].normalized_risk",
            ),
            expected_risk,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("Attempt08 normalized raw risk score changed")
        legal_rank_mean.append(rank_mean)
        legal_tails.append((p95, p99, maximum))

    expected_top_indices = sorted(
        (index for index in range(len(legal_actions)) if index != baseline_index),
        key=lambda index: (
            -legal_rank_mean[index],
            action_key(legal_actions[index]).sort_key(),
        ),
    )[:ATTEMPT08_TOP_K]
    expected_top_keys = [
        action_key(legal_actions[index]).to_token() for index in expected_top_indices
    ]
    if (
        payload.get("learned_top8_original_legal_indices") != expected_top_indices
        or payload.get("learned_top8_action_keys") != expected_top_keys
    ):
        raise ValueError("Attempt08 learned top8 selection changed")
    for index, raw_row in enumerate(raw_legal_rows):
        row = _payload_mapping(raw_row, f"legal_actions[{index}]")
        if row.get("in_learned_top8") is not (index in set(expected_top_indices)):
            raise ValueError("Attempt08 learned top8 membership changed")

    proposal_actions = tuple(
        [*(legal_actions[index] for index in expected_top_indices), legal_actions[baseline_index]]
    )
    proposal_mapping = _payload_mapping(
        payload.get("proposal_mapping"), "proposal_mapping"
    )
    _validate_payload_mapping(proposal_mapping, proposal_actions, label="proposal")

    rerank = _payload_mapping(payload.get("rerank"), "rerank")
    _require_exact_keys(
        rerank,
        {
            "action_count",
            "action_keys",
            "action_set_digest",
            "action_order_digest",
            "sample_count",
            "prefix_semantics",
            "common_random_futures",
            "nonbaseline_order_rule",
            "ordered_nonbaseline_proposal_positions",
            "ordered_nonbaseline_action_keys",
            "actions",
        },
        "rerank",
    )
    if (
        rerank.get("prefix_semantics") != "full_R128_only"
        or rerank.get("common_random_futures") is not True
        or rerank.get("nonbaseline_order_rule")
        != "paired_mean_desc_then_ActionKey"
    ):
        raise ValueError("Attempt08 R128 protocol changed")
    rerank_rows = _validate_phase_payload(
        rerank,
        proposal_actions,
        baseline_position=ATTEMPT08_TOP_K,
        sample_count=ATTEMPT08_RERANK_SAMPLES,
        label="R128",
    )
    rerank_order = sorted(
        range(ATTEMPT08_TOP_K),
        key=lambda position: (
            -float(rerank_rows[position]["summary"]["mean"]),
            action_key(proposal_actions[position]).sort_key(),
        ),
    )
    expected_rerank_keys = [
        action_key(proposal_actions[position]).to_token()
        for position in rerank_order
    ]
    if (
        rerank.get("ordered_nonbaseline_proposal_positions") != rerank_order
        or rerank.get("ordered_nonbaseline_action_keys") != expected_rerank_keys
    ):
        raise ValueError("Attempt08 R128 order changed")

    top3_positions = tuple(rerank_order[:ATTEMPT08_RERANK_TOP_K])
    reserve_pool = tuple(rerank_order[ATTEMPT08_RERANK_TOP_K :])
    reserve_scores: dict[int, tuple[float, dict[str, float]]] = {}
    for position in reserve_pool:
        original_index = expected_top_indices[position]
        p95, p99, maximum = legal_tails[original_index]
        components = {
            "p95_over_22": p95 / 22.0,
            "p99_over_36": p99 / 36.0,
            "max_over_45": maximum / 45.0,
        }
        reserve_scores[position] = (max(components.values()), components)
    reserve_position = min(
        reserve_pool,
        key=lambda position: (
            reserve_scores[position][0],
            action_key(proposal_actions[position]).sort_key(),
        ),
    )
    member_positions = {*top3_positions, reserve_position}
    k4_positions = tuple(
        position for position in rerank_order if position in member_positions
    )
    k4_actions = tuple(proposal_actions[position] for position in k4_positions)
    k4 = _payload_mapping(payload.get("k4"), "k4")
    _require_exact_keys(
        k4,
        {
            "action_count",
            "action_keys",
            "action_set_digest",
            "action_order_digest",
            "selection_rule",
            "top3_rerank_positions",
            "top3_action_keys",
            "risk_reserve_rerank_position",
            "risk_reserve_r_rank",
            "risk_reserve_action_key",
            "risk_reserve_raw_components",
            "risk_reserve_normalized_score",
            "veto_traversal_rule",
            "veto_traversal_rerank_positions",
        },
        "k4",
    )
    for key, expected in _mapping_payload(k4_actions).items():
        if k4.get(key) != expected:
            raise ValueError(f"Attempt08 K4 {key} changed")
    expected_reserve_components = reserve_scores[reserve_position][1]
    reserve_components_payload = _payload_mapping(
        k4.get("risk_reserve_raw_components"), "k4.risk_reserve_raw_components"
    )
    _require_exact_keys(
        reserve_components_payload,
        {"p95_over_22", "p99_over_36", "max_over_45"},
        "k4.risk_reserve_raw_components",
    )
    if (
        k4.get("selection_rule")
        != (
            "R_top3_plus_ranks4_8_min_max_raw_p95_over22_"
            "raw_p99_over36_raw_max_over45_then_ActionKey"
        )
        or k4.get("top3_rerank_positions") != list(top3_positions)
        or k4.get("top3_action_keys")
        != [action_key(proposal_actions[position]).to_token() for position in top3_positions]
        or k4.get("risk_reserve_rerank_position") != reserve_position
        or k4.get("risk_reserve_r_rank") != rerank_order.index(reserve_position) + 1
        or k4.get("risk_reserve_action_key")
        != action_key(proposal_actions[reserve_position]).to_token()
        or reserve_components_payload != expected_reserve_components
        or not math.isclose(
            _payload_finite(
                k4.get("risk_reserve_normalized_score"),
                "k4.risk_reserve_normalized_score",
            ),
            reserve_scores[reserve_position][0],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or k4.get("veto_traversal_rule") != "original_R_order"
        or k4.get("veto_traversal_rerank_positions") != list(k4_positions)
    ):
        raise ValueError("Attempt08 K4 risk-reserve contract changed")

    veto_actions = (*k4_actions, legal_actions[baseline_index])
    veto = _payload_mapping(payload.get("veto"), "veto")
    _require_exact_keys(
        veto,
        {
            "action_count",
            "action_keys",
            "action_set_digest",
            "action_order_digest",
            "opened",
            "sample_count",
            "prefix_semantics",
            "common_random_futures",
            "scope",
            "thresholds",
            "traversal_action_keys",
            "checks_by_traversal_position",
            "first_passing_traversal_position",
            "first_passing_action_key",
            "selection_rule",
            "actions",
        },
        "veto",
    )
    expected_thresholds = {
        "mean_strictly_greater_than": ATTEMPT08_MIN_MEAN,
        "p05_at_least": ATTEMPT08_MIN_P05,
        "p01_at_least": ATTEMPT08_MIN_P01,
        "min_at_least": ATTEMPT08_MIN_VALUE,
    }
    thresholds = _payload_mapping(veto.get("thresholds"), "veto.thresholds")
    _require_exact_keys(
        thresholds, set(expected_thresholds), "veto.thresholds"
    )
    if (
        veto.get("opened") is not True
        or veto.get("prefix_semantics") != "full_V256_only"
        or veto.get("common_random_futures") is not True
        or veto.get("scope") != "frozen_K4_in_R_order_plus_explicit_baseline"
        or thresholds != expected_thresholds
        or veto.get("traversal_action_keys")
        != [action_key(action).to_token() for action in k4_actions]
        or veto.get("selection_rule")
        != "first_safe_in_frozen_R_order_else_baseline"
    ):
        raise ValueError("Attempt08 V256 protocol changed")
    veto_rows = _validate_phase_payload(
        veto,
        veto_actions,
        baseline_position=ATTEMPT08_K4,
        sample_count=ATTEMPT08_VETO_SAMPLES,
        label="V256",
    )
    veto_checks = [
        _veto_checks(row["summary"]) for row in veto_rows[:ATTEMPT08_K4]
    ]
    raw_checks = _payload_sequence(
        veto.get("checks_by_traversal_position"),
        "veto.checks_by_traversal_position",
    )
    if len(raw_checks) != ATTEMPT08_K4:
        raise ValueError("Attempt08 V256 check row count changed")
    for position, raw_checks_row in enumerate(raw_checks):
        checks_row = _payload_mapping(
            raw_checks_row, f"veto.checks_by_traversal_position[{position}]"
        )
        _require_exact_keys(
            checks_row,
            {"mean_gt_0", "p05_ge_neg22", "p01_ge_neg36", "min_ge_neg45"},
            f"veto.checks_by_traversal_position[{position}]",
        )
    passing_positions = [
        position for position, checks in enumerate(veto_checks) if all(checks.values())
    ]
    veto_selected_position = passing_positions[0] if passing_positions else None
    veto_selected_token = (
        action_key(veto_actions[veto_selected_position]).to_token()
        if veto_selected_position is not None
        else None
    )
    if (
        list(raw_checks) != veto_checks
        or veto.get("first_passing_traversal_position") != veto_selected_position
        or veto.get("first_passing_action_key") != veto_selected_token
    ):
        raise ValueError("Attempt08 V256 first-safe selection changed")

    stress = _payload_mapping(payload.get("stress"), "stress")
    _require_exact_keys(
        stress,
        {
            "action_count",
            "action_keys",
            "action_set_digest",
            "action_order_digest",
            "opened",
            "sample_count",
            "prefix_semantics",
            "common_random_futures",
            "scope",
            "locked_candidate_action_key",
            "min_at_least",
            "cancel_only",
            "may_promote_or_rerank",
            "pass",
            "cancelled",
            "paired_delta_vs_baseline",
            "raw_paired_deltas_vs_baseline",
            "raw_paired_deltas_sha256",
            "actions",
        },
        "stress",
    )
    stress_opened = veto_selected_position is not None
    stress_pass: bool | None = None
    stress_cancelled = False
    if stress_opened:
        stress_actions = (
            veto_actions[veto_selected_position],
            legal_actions[baseline_index],
        )
        if (
            stress.get("opened") is not True
            or stress.get("prefix_semantics") != "full_X512_only_if_V_fires"
            or stress.get("common_random_futures") is not True
            or stress.get("scope")
            != "locked_V_candidate_plus_explicit_baseline"
            or stress.get("locked_candidate_action_key") != veto_selected_token
            or stress.get("min_at_least") != ATTEMPT08_MIN_VALUE
            or stress.get("cancel_only") is not True
            or stress.get("may_promote_or_rerank") is not False
        ):
            raise ValueError("Attempt08 X512 lock/cancel protocol changed")
        stress_rows = _validate_phase_payload(
            stress,
            stress_actions,
            baseline_position=1,
            sample_count=ATTEMPT08_STRESS_SAMPLES,
            label="X512",
        )
        first_stress = stress_rows[0]
        stress_pass = float(first_stress["summary"]["min"]) >= ATTEMPT08_MIN_VALUE
        stress_cancelled = not stress_pass
        if (
            stress.get("pass") is not stress_pass
            or stress.get("cancelled") is not stress_cancelled
            or stress.get("paired_delta_vs_baseline") != first_stress["summary"]
            or stress.get("raw_paired_deltas_vs_baseline") != first_stress["raw"]
            or stress.get("raw_paired_deltas_sha256")
            != _raw_digest(first_stress["raw"])
        ):
            raise ValueError("Attempt08 X512 catastrophe decision changed")
    else:
        stress_actions = ()
        _validate_payload_mapping(
            stress, stress_actions, label="closed X512", exact=False
        )
        if (
            stress.get("opened") is not False
            or stress.get("sample_count") != 0
            or stress.get("prefix_semantics") != "full_X512_only_if_V_fires"
            or stress.get("common_random_futures") is not False
            or stress.get("scope") != "not_opened_because_V_did_not_fire"
            or stress.get("locked_candidate_action_key") is not None
            or stress.get("min_at_least") != ATTEMPT08_MIN_VALUE
            or stress.get("cancel_only") is not True
            or stress.get("may_promote_or_rerank") is not False
            or stress.get("pass") is not None
            or stress.get("cancelled") is not False
            or stress.get("paired_delta_vs_baseline") is not None
            or stress.get("raw_paired_deltas_vs_baseline") != []
            or stress.get("raw_paired_deltas_sha256") is not None
            or stress.get("actions") != []
        ):
            raise ValueError("Attempt08 closed X512 retained decision state")

    decision = _payload_mapping(payload.get("decision"), "decision")
    _require_exact_keys(
        decision,
        {
            "veto_selected_action_key",
            "veto_override_fired",
            "stress_opened",
            "stress_cancelled",
            "final_selected_action_key",
            "override_fired",
            "exact_baseline_fallback",
            "fallback_reason",
            "second_candidate_promotion_after_stress_cancel_allowed",
            "frozen_before_assessment_namespace_open",
        },
        "decision",
    )
    final_token = (
        veto_selected_token
        if veto_selected_token is not None and stress_pass is True
        else baseline_action_key
    )
    override_fired = final_token != baseline_action_key
    fallback_reason = (
        None
        if override_fired
        else (
            "no_v256_candidate_passed"
            if veto_selected_token is None
            else "x512_catastrophe_cancel"
        )
    )
    if (
        decision.get("veto_selected_action_key") != veto_selected_token
        or decision.get("veto_override_fired") is not (veto_selected_token is not None)
        or decision.get("stress_opened") is not stress_opened
        or decision.get("stress_cancelled") is not stress_cancelled
        or decision.get("final_selected_action_key") != final_token
        or decision.get("override_fired") is not override_fired
        or decision.get("exact_baseline_fallback") is not (not override_fired)
        or decision.get("fallback_reason") != fallback_reason
        or decision.get("second_candidate_promotion_after_stress_cancel_allowed")
        is not False
        or decision.get("frozen_before_assessment_namespace_open") is not True
    ):
        raise ValueError("Attempt08 locked final decision changed")

    assessment = _payload_mapping(payload.get("assessment"), "assessment")
    _require_exact_keys(
        assessment,
        {
            "action_count",
            "action_keys",
            "action_set_digest",
            "action_order_digest",
            "opened",
            "sample_count",
            "prefix_semantics",
            "common_random_futures",
            "scope",
            "execution_condition",
            "skipped_when_final_output_is_baseline",
            "diagnostics_only",
            "can_rerank_or_gate",
            "decision_frozen_before_namespace_open",
            "locked_final_action_key",
            "sample_best_action_key",
            "paired_delta_vs_baseline",
            "raw_paired_deltas_vs_baseline",
            "raw_paired_deltas_sha256",
            "actions",
        },
        "assessment",
    )
    if (
        assessment.get("prefix_semantics")
        != "full_A256_only_if_final_output_is_nonbaseline"
        or assessment.get("execution_condition")
        != "final_output_is_nonbaseline_after_X512"
        or assessment.get("skipped_when_final_output_is_baseline") is not True
        or assessment.get("diagnostics_only") is not True
        or assessment.get("can_rerank_or_gate") is not False
        or assessment.get("decision_frozen_before_namespace_open") is not True
        or assessment.get("locked_final_action_key") != final_token
    ):
        raise ValueError("Attempt08 A256 diagnostic lock changed")
    if override_fired:
        final_action = next(
            action
            for action in legal_actions
            if action_key(action).to_token() == final_token
        )
        assessment_actions = (final_action, legal_actions[baseline_index])
        if (
            assessment.get("opened") is not True
            or assessment.get("common_random_futures") is not True
            or assessment.get("scope")
            != "locked_nonbaseline_final_plus_explicit_baseline"
        ):
            raise ValueError("Attempt08 A256 fire scope changed")
        assessment_rows = _validate_phase_payload(
            assessment,
            assessment_actions,
            baseline_position=1,
            sample_count=ATTEMPT08_ASSESSMENT_SAMPLES,
            label="A256",
        )
        first_assessment = assessment_rows[0]
        best_mean = max(float(row["mean"]) for row in assessment_rows)
        best_position = min(
            [
                position
                for position, row in enumerate(assessment_rows)
                if float(row["mean"]) == best_mean
            ],
            key=lambda position: action_key(assessment_actions[position]).sort_key(),
        )
        if (
            assessment.get("sample_best_action_key")
            != action_key(assessment_actions[best_position]).to_token()
            or assessment.get("paired_delta_vs_baseline")
            != first_assessment["summary"]
            or assessment.get("raw_paired_deltas_vs_baseline")
            != first_assessment["raw"]
            or assessment.get("raw_paired_deltas_sha256")
            != _raw_digest(first_assessment["raw"])
        ):
            raise ValueError("Attempt08 A256 locked diagnostics changed")
        raw_assessment = list(first_assessment["raw"])
    else:
        assessment_actions = ()
        _validate_payload_mapping(
            assessment, assessment_actions, label="closed A256", exact=False
        )
        if (
            assessment.get("opened") is not False
            or assessment.get("sample_count") != 0
            or assessment.get("common_random_futures") is not False
            or assessment.get("scope")
            != "not_opened_because_final_output_is_baseline"
            or assessment.get("sample_best_action_key") is not None
            or assessment.get("paired_delta_vs_baseline") is not None
            or assessment.get("raw_paired_deltas_vs_baseline") != []
            or assessment.get("raw_paired_deltas_sha256") is not None
            or assessment.get("actions") != []
        ):
            raise ValueError("Attempt08 closed A256 retained diagnostics")
        raw_assessment = []

    expected_seeds = {
        "hand_external": config.hand_seed,
        "rerank_r128": config.rerank_seed,
        "veto_v256": config.veto_seed,
        "stress_x512": config.stress_seed,
        "assessment_a256": config.assessment_seed,
        "child_policy": config.child_policy_seed,
    }
    provenance = _payload_mapping(
        payload.get("seed_domain_provenance"), "seed_domain_provenance"
    )
    _require_exact_keys(
        provenance,
        {
            "domain_order",
            "hand_external",
            "rerank_r128",
            "veto_v256",
            "stress_x512",
            "assessment_a256",
            "child_policy",
            "all_six_base_seeds_pairwise_distinct",
            "hand_sampled_inside_teacher",
        },
        "seed_domain_provenance",
    )
    if (
        provenance.get("domain_order") != list(ATTEMPT08_RNG_DOMAINS)
        or provenance.get("all_six_base_seeds_pairwise_distinct") is not True
        or provenance.get("hand_sampled_inside_teacher") is not False
    ):
        raise ValueError("Attempt08 seed-domain protocol changed")
    for name, seed in expected_seeds.items():
        if provenance.get(name) != seed:
            raise ValueError(f"Attempt08 {name} seed changed")

    search_config = _payload_mapping(payload.get("search_config"), "search_config")
    expected_search_values = {
        "learned_nonbaseline_top_k": ATTEMPT08_TOP_K,
        "baseline_added_exactly_once": True,
        "rerank_samples": ATTEMPT08_RERANK_SAMPLES,
        "rerank_top_k": ATTEMPT08_RERANK_TOP_K,
        "risk_reserve_count": 1,
        "k4_size": ATTEMPT08_K4,
        "veto_samples": ATTEMPT08_VETO_SAMPLES,
        "stress_samples": ATTEMPT08_STRESS_SAMPLES,
        "assessment_samples": ATTEMPT08_ASSESSMENT_SAMPLES,
        "hand_seed": config.hand_seed,
        "rerank_seed": config.rerank_seed,
        "veto_seed": config.veto_seed,
        "stress_seed": config.stress_seed,
        "assessment_seed": config.assessment_seed,
        "child_policy_seed": config.child_policy_seed,
        "run_id": config.run_id,
        "batch_child_selectors": config.batch_child_selectors,
        "candidate_and_rerank_tie_break": "ActionKey",
        "raw_risk_reserve_tie_break": "ActionKey",
    }
    _require_exact_keys(
        search_config, set(expected_search_values), "search_config"
    )
    for key, expected in expected_search_values.items():
        if search_config.get(key) != expected:
            raise ValueError(f"Attempt08 search_config {key} changed")

    continuation = _payload_mapping(
        payload.get("continuation_policy"), "continuation_policy"
    )
    expected_continuation = {
        "t2_policy_id": ATTEMPT08_T2_POLICY_ID,
        "t2_resolution": "explicit_profile_never_current",
        "t2_runtime_status": "p2_fixed",
        "t3_selector": "m3_rust_evaluate_t3",
        "t3_candidate_samples": 1,
        "t3_evaluation_samples": 1,
        "t3_downstream_samples": 1,
        "hypothetical_t4_selector": "m3_rust_evaluate_t4",
        "hypothetical_t4_candidate_samples": 1,
        "hypothetical_t4_evaluation_samples": 1,
        "real_live_t4_exact_unchanged": True,
        "fresh_selector_per_phase": True,
        "child_cache_released_after_each_phase": True,
    }
    _require_exact_keys(
        continuation, set(expected_continuation), "continuation_policy"
    )
    if continuation != expected_continuation:
        raise ValueError("Attempt08 continuation policy changed")
    expected_live_schedule = [
        {
            "seat": step.seat,
            "street": step.street,
            "draw_offset": step.draw_offset,
        }
        for step in T1_SECOND_LIVE_SCHEDULE
    ]
    live_schedule = _payload_sequence(payload.get("live_schedule"), "live_schedule")
    if list(live_schedule) != expected_live_schedule:
        raise ValueError("Attempt08 live schedule changed")
    for index, raw_step in enumerate(live_schedule):
        step = _payload_mapping(raw_step, f"live_schedule[{index}]")
        _require_exact_keys(
            step, {"seat", "street", "draw_offset"}, f"live_schedule[{index}]"
        )
    if (
        payload.get("sample_independence")
        != "pairwise_disjoint_R128_V256_optional_X512_optional_fire_A256_particle_rng_keys"
        or payload.get("root_selection_lock")
        != (
            "top8_before_R128_then_K4_before_V256_then_V_choice_before_"
            "optional_X512_then_final_before_optional_fire_diagnostic_A256"
        )
    ):
        raise ValueError("Attempt08 phase lock/independence contract changed")

    rng = _payload_mapping(payload.get("rng_key_digests"), "rng_key_digests")
    belief = _payload_mapping(payload.get("belief_digests"), "belief_digests")
    child_counts = _payload_mapping(
        payload.get("phase_child_information_set_counts"),
        "phase_child_information_set_counts",
    )
    expected_phases = {"rerank_r128", "veto_v256"}
    if stress_opened:
        expected_phases.add("stress_x512")
    if override_fired:
        expected_phases.add("assessment_a256")
    if (
        set(rng) != expected_phases
        or set(belief) != expected_phases
        or set(child_counts) != expected_phases
    ):
        raise ValueError("Attempt08 opened phase provenance set changed")
    expected_counts = {
        "rerank_r128": ATTEMPT08_RERANK_SAMPLES,
        "veto_v256": ATTEMPT08_VETO_SAMPLES,
        "stress_x512": ATTEMPT08_STRESS_SAMPLES,
        "assessment_a256": ATTEMPT08_ASSESSMENT_SAMPLES,
    }
    all_keys: list[str] = []
    all_beliefs: list[str] = []
    phase_key_sets: list[list[str]] = []
    for phase in sorted(expected_phases):
        values = _payload_sequence(rng.get(phase), f"rng_key_digests.{phase}")
        if len(values) != expected_counts[phase] or not all(
            _is_sha256(value) for value in values
        ):
            raise ValueError(f"Attempt08 {phase} RNG key count/digest changed")
        keys = [str(value) for value in values]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Attempt08 {phase} RNG keys repeat")
        for previous in phase_key_sets:
            require_disjoint_root_rng_keys(previous, keys)
        phase_key_sets.append(keys)
        all_keys.extend(keys)
        belief_digest = belief.get(phase)
        if not _is_sha256(belief_digest):
            raise ValueError(f"Attempt08 {phase} belief digest changed")
        all_beliefs.append(str(belief_digest))
        count = child_counts.get(phase)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"Attempt08 {phase} child count changed")
    if len(all_keys) != len(set(all_keys)):
        raise ValueError("Attempt08 phase RNG namespaces overlap")
    if len(all_beliefs) != len(set(all_beliefs)):
        raise ValueError("Attempt08 phase belief digests overlap")

    memory = payload.get("memory_retention")
    if not isinstance(memory, Mapping) or memory != {
        "retained_particle_batches": 0,
        "retained_hidden_particle_payload": False,
        "retained_child_selector_caches": 0,
        "output_contains_only_value_vectors_and_opaque_digests": True,
    }:
        raise ValueError("Attempt08 memory-retention contract changed")

    def reject_private_or_profile_input(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key in {"opponent_private_discards", "root_profile", "runtime_profile"}:
                    raise ValueError(
                        f"Attempt08 forbidden private/profile input at {path}.{key}"
                    )
                reject_private_or_profile_input(child, f"{path}.{key}")
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for index, child in enumerate(value):
                reject_private_or_profile_input(child, f"{path}[{index}]")

    reject_private_or_profile_input(payload, "teacher")
    if (
        payload.get("teacher_value_status") != "diagnostic_not_match_EV"
        or payload.get("runtime_gate_allowed") is not False
        or payload.get("profile_activation_allowed") is not False
        or payload.get("current_profile_resolved") is not False
        or payload.get("development_only") is not True
    ):
        raise ValueError("Attempt08 output attempted runtime/profile activation")

    return {
        "selected_action_key": final_token,
        "override_fired": override_fired,
        "exact_baseline_fallback": not override_fired,
        "assessment_raw_paired_deltas_vs_baseline": raw_assessment,
        "rng_key_digests": {str(key): list(value) for key, value in rng.items()},
        "belief_digests": {str(key): str(value) for key, value in belief.items()},
    }


__all__ = [
    "ATTEMPT08_ASSESSMENT_SAMPLES",
    "ATTEMPT08_FROZEN_MODEL_ID",
    "ATTEMPT08_FROZEN_MODEL_SHA256",
    "ATTEMPT08_K4",
    "ATTEMPT08_MIN_MEAN",
    "ATTEMPT08_MIN_P01",
    "ATTEMPT08_MIN_P05",
    "ATTEMPT08_MIN_VALUE",
    "ATTEMPT08_RERANK_SAMPLES",
    "ATTEMPT08_RERANK_TOP_K",
    "ATTEMPT08_RNG_DOMAINS",
    "ATTEMPT08_SOLVER_ID",
    "ATTEMPT08_STRESS_SAMPLES",
    "ATTEMPT08_TEACHER_SCHEMA",
    "ATTEMPT08_T2_POLICY_ID",
    "ATTEMPT08_TOP_K",
    "ATTEMPT08_VETO_SAMPLES",
    "Attempt08RankScores",
    "Attempt08Ranker",
    "Attempt08TeacherConfig",
    "FrozenAttempt08LambdaRanker",
    "evaluate_attempt08_t1_second",
    "validate_attempt08_teacher_output",
]
