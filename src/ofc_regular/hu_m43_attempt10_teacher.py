"""Bounded, profile-blind Attempt10 T1-second search teacher.

Attempt10 is a fresh one-shot architecture.  It shares only the frozen
candidate model and generic rollout primitives with earlier attempts:

* Lambda Top12 plus the explicit baseline are reranked with R128;
* K8 is R's top four plus the four lowest Lambda-risk actions from R ranks
  five through twelve, retained in frozen R order;
* V256 is a coarse, independent all-survivor filter;
* X1024 and C512 are independent strict all-survivor filters;
* one C survivor is locked by normalized observed tail risk, mean, frozen R
  order, and ActionKey; and
* E256 is opened after the lock and is diagnostic-only.

Every rollout phase uses common random futures across actions within the phase,
and all seven root seed domains must be pairwise distinct.  Hidden opponent
discards and particle payloads are never retained.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
from dataclasses import dataclass
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
from .hu_m43_attempt06_teacher import _require_sha256, _require_stage9f_p2_policies
from .hu_m43_attempt08_teacher import (
    ATTEMPT08_FROZEN_MODEL_ID,
    ATTEMPT08_FROZEN_MODEL_SHA256,
    ATTEMPT08_T2_POLICY_ID,
    Attempt08RankScores,
    FrozenAttempt08LambdaRanker,
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


ATTEMPT10_TEACHER_SCHEMA = (
    "hu_m43_attempt10_t1_second_top12_r128_k8_v256_x1024_c512_e256_v1"
)
ATTEMPT10_SOLVER_ID = (
    "lambda_top12_r128_top4_risk4_v256_coarse_x1024_strict_"
    "c512_strict_tailrisk_lock_e256_m3_mc1_v1"
)
ATTEMPT10_FROZEN_MODEL_SHA256 = ATTEMPT08_FROZEN_MODEL_SHA256
ATTEMPT10_FROZEN_MODEL_ID = ATTEMPT08_FROZEN_MODEL_ID
ATTEMPT10_T2_POLICY_ID = ATTEMPT08_T2_POLICY_ID

ATTEMPT10_TOP_K = 12
ATTEMPT10_RERANK_SAMPLES = 128
ATTEMPT10_RERANK_TOP_K = 4
ATTEMPT10_RISK_RESERVE_COUNT = 4
ATTEMPT10_K8 = 8
ATTEMPT10_VETO_SAMPLES = 256
ATTEMPT10_STRESS_SAMPLES = 1024
ATTEMPT10_CONFIRMATION_SAMPLES = 512
ATTEMPT10_EVALUATION_SAMPLES = 256

ATTEMPT10_COARSE_MIN_MEAN = 0.0
ATTEMPT10_COARSE_MIN_P05 = -25.0
ATTEMPT10_COARSE_MIN_P01 = -40.0
ATTEMPT10_COARSE_MIN_VALUE = -50.0
ATTEMPT10_STRICT_MIN_MEAN = 0.0
ATTEMPT10_STRICT_MIN_P05 = -22.0
ATTEMPT10_STRICT_MIN_P01 = -36.0
ATTEMPT10_STRICT_MIN_VALUE = -45.0

ATTEMPT10_RNG_DOMAINS = (
    "hand_external",
    "rerank_r128",
    "veto_v256",
    "stress_x1024",
    "confirmation_c512",
    "evaluation_e256",
    "child_policy",
)

Attempt10RankScores = Attempt08RankScores
FrozenAttempt10LambdaRanker = FrozenAttempt08LambdaRanker


class Attempt10Ranker(Protocol):
    artifact_sha256: str
    model_id: str

    def score_actions(
        self,
        observation: ActorObservation,
        actions: Sequence[Action],
        *,
        baseline_index: int,
    ) -> Attempt10RankScores: ...


@dataclass(frozen=True)
class Attempt10TeacherConfig:
    """Hard-locked Attempt10 configuration for one canonical root."""

    frozen_model_sha256: str
    hand_seed: int
    rerank_seed: int
    veto_seed: int
    stress_seed: int
    confirmation_seed: int
    evaluation_seed: int
    child_policy_seed: int
    run_id: str
    candidate_top_k: int = ATTEMPT10_TOP_K
    rerank_samples: int = ATTEMPT10_RERANK_SAMPLES
    rerank_top_k: int = ATTEMPT10_RERANK_TOP_K
    risk_reserve_count: int = ATTEMPT10_RISK_RESERVE_COUNT
    k8_size: int = ATTEMPT10_K8
    veto_samples: int = ATTEMPT10_VETO_SAMPLES
    stress_samples: int = ATTEMPT10_STRESS_SAMPLES
    confirmation_samples: int = ATTEMPT10_CONFIRMATION_SAMPLES
    evaluation_samples: int = ATTEMPT10_EVALUATION_SAMPLES
    coarse_min_mean: float = ATTEMPT10_COARSE_MIN_MEAN
    coarse_min_p05: float = ATTEMPT10_COARSE_MIN_P05
    coarse_min_p01: float = ATTEMPT10_COARSE_MIN_P01
    coarse_min_value: float = ATTEMPT10_COARSE_MIN_VALUE
    strict_min_mean: float = ATTEMPT10_STRICT_MIN_MEAN
    strict_min_p05: float = ATTEMPT10_STRICT_MIN_P05
    strict_min_p01: float = ATTEMPT10_STRICT_MIN_P01
    strict_min_value: float = ATTEMPT10_STRICT_MIN_VALUE
    t2_policy_id: str = ATTEMPT10_T2_POLICY_ID
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
        if self.frozen_model_sha256 != ATTEMPT10_FROZEN_MODEL_SHA256:
            raise ValueError("Attempt10 requires the frozen candidate-only Lambda artifact")
        fixed_ints = {
            "candidate_top_k": ATTEMPT10_TOP_K,
            "rerank_samples": ATTEMPT10_RERANK_SAMPLES,
            "rerank_top_k": ATTEMPT10_RERANK_TOP_K,
            "risk_reserve_count": ATTEMPT10_RISK_RESERVE_COUNT,
            "k8_size": ATTEMPT10_K8,
            "veto_samples": ATTEMPT10_VETO_SAMPLES,
            "stress_samples": ATTEMPT10_STRESS_SAMPLES,
            "confirmation_samples": ATTEMPT10_CONFIRMATION_SAMPLES,
            "evaluation_samples": ATTEMPT10_EVALUATION_SAMPLES,
            "t3_candidate_samples": 1,
            "t3_evaluation_samples": 1,
            "t3_downstream_samples": 1,
            "t4_candidate_samples": 1,
            "t4_evaluation_samples": 1,
        }
        for name, expected in fixed_ints.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value != expected:
                raise ValueError(f"Attempt10 {name} is fixed at {expected}")
        fixed_floats = {
            "coarse_min_mean": ATTEMPT10_COARSE_MIN_MEAN,
            "coarse_min_p05": ATTEMPT10_COARSE_MIN_P05,
            "coarse_min_p01": ATTEMPT10_COARSE_MIN_P01,
            "coarse_min_value": ATTEMPT10_COARSE_MIN_VALUE,
            "strict_min_mean": ATTEMPT10_STRICT_MIN_MEAN,
            "strict_min_p05": ATTEMPT10_STRICT_MIN_P05,
            "strict_min_p01": ATTEMPT10_STRICT_MIN_P01,
            "strict_min_value": ATTEMPT10_STRICT_MIN_VALUE,
        }
        for name, expected in fixed_floats.items():
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) != expected
            ):
                raise ValueError(f"Attempt10 {name} is fixed at {expected}")
        seed_names = (
            "hand_seed",
            "rerank_seed",
            "veto_seed",
            "stress_seed",
            "confirmation_seed",
            "evaluation_seed",
            "child_policy_seed",
        )
        for name in seed_names:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"Attempt10 {name} must be an integer")
        if len({getattr(self, name) for name in seed_names}) != len(seed_names):
            raise ValueError("Attempt10 hand/R/V/X/C/E/child seeds must all be distinct")
        if self.t2_policy_id != ATTEMPT10_T2_POLICY_ID:
            raise ValueError("Attempt10 T2 policy is fixed at stage9f_p2")
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("Attempt10 run_id must not be empty")
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


def _raw_digest(values: Sequence[float]) -> str:
    return hashlib.sha256(
        json.dumps(
            [float(value) for value in values],
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _mapping_payload(actions: Sequence[Action]) -> dict[str, Any]:
    return {
        "action_count": len(actions),
        "action_keys": [action_key(action).to_token() for action in actions],
        "action_set_digest": legal_action_set_digest(actions),
        "action_order_digest": ordered_action_mapping_digest(actions),
    }


def _paired_row(candidate: _ActionScores, baseline: _ActionScores) -> dict[str, Any]:
    deltas = [
        float(left - right)
        for left, right in zip(candidate.values, baseline.values, strict=True)
    ]
    if not deltas or not all(math.isfinite(value) for value in deltas):
        raise ValueError("Attempt10 paired deltas must be finite and non-empty")
    return {
        "paired_delta_vs_baseline": _paired_delta_summary(candidate, baseline),
        "raw_paired_deltas_vs_baseline": deltas,
        "raw_paired_deltas_sha256": _raw_digest(deltas),
        "opaque_action_values_sha256": _raw_digest(candidate.values),
    }


def _phase_actions(
    actions: Sequence[Action],
    scores: Sequence[_ActionScores],
    *,
    baseline_position: int,
) -> list[dict[str, Any]]:
    if len(actions) != len(scores) or not 0 <= baseline_position < len(actions):
        raise ValueError("Attempt10 phase action mapping is invalid")
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


def _checks(
    summary: Mapping[str, Any],
    *,
    min_mean: float,
    min_p05: float,
    min_p01: float,
    min_value: float,
) -> dict[str, bool]:
    return {
        "mean_gt_0": float(summary["mean"]) > min_mean,
        "p05_at_least": float(summary["p05"]) >= min_p05,
        "p01_at_least": float(summary["p01"]) >= min_p01,
        "min_at_least": float(summary["min"]) >= min_value,
    }


def _lambda_risk_components(
    rank_scores: Attempt10RankScores, index: int
) -> dict[str, float]:
    return {
        "p95_over_22": float(rank_scores.raw_downside_p95[index]) / 22.0,
        "p99_over_36": float(rank_scores.raw_downside_p99[index]) / 36.0,
        "max_over_45": float(rank_scores.raw_downside_max[index]) / 45.0,
    }


def _observed_tail_risk(summary: Mapping[str, Any]) -> tuple[float, dict[str, float]]:
    components = {
        "loss95_over_22": max(0.0, -float(summary["p05"])) / 22.0,
        "loss99_over_36": max(0.0, -float(summary["p01"])) / 36.0,
        "lossmax_over_45": max(0.0, -float(summary["min"])) / 45.0,
    }
    return max(components.values()), components


def _score_phase(
    observation: ActorObservation,
    actions: Sequence[Action],
    *,
    phase: str,
    seed: int,
    sample_count: int,
    config: Attempt10TeacherConfig,
    t2_policies: Mapping[str, object],
    prior_rng_keys: Sequence[Sequence[str]],
    library: Any | None,
) -> _PhaseResult:
    batch = sample_hidden_card_particles(
        observation,
        base_seed=seed,
        run_id=f"{config.run_id}:{phase}",
        sample_count=sample_count,
    )
    batch.validate_against(observation)
    keys = tuple(particle.rng_key_digest for particle in batch.particles)
    if len(keys) != sample_count or len(set(keys)) != sample_count:
        raise ValueError(f"Attempt10 {phase} particle RNG keys are invalid")
    for existing in prior_rng_keys:
        require_disjoint_root_rng_keys(existing, keys)
    selector = _ChildSelector(
        t2_policies=t2_policies,
        config=config.m4_config(),
        library=library,
    )
    scorer = _score_actions_batched if config.batch_child_selectors else _score_actions
    scored = tuple(scorer(observation, actions, batch, selector))
    if len(scored) != len(actions):
        raise ValueError(f"Attempt10 {phase} scorer returned the wrong action count")
    for row in scored:
        if len(row.values) != sample_count or not all(
            math.isfinite(float(value)) for value in row.values
        ):
            raise ValueError(f"Attempt10 {phase} scorer returned invalid values")
    result = _PhaseResult(scored, batch.digest(), keys, len(selector.cache))
    selector.cache.clear()
    del selector
    del batch
    gc.collect()
    return result


def _closed_phase(*, reason: str, sample_count: int) -> dict[str, Any]:
    return {
        **_mapping_payload(()),
        "opened": False,
        "sample_count": 0,
        "configured_sample_count": sample_count,
        "common_random_futures": False,
        "scope": reason,
        "retained_action_keys": [],
        "actions": [],
    }


def evaluate_attempt10_t1_second(
    observation: ActorObservation,
    *,
    baseline_action_key: str,
    ranker: Attempt10Ranker,
    t2_policies: Mapping[str, object],
    config: Attempt10TeacherConfig,
    library: Any | None = None,
) -> dict[str, Any]:
    """Evaluate the fixed Attempt10 search without activating a policy."""

    require_t1_second_root(observation)
    if ranker.artifact_sha256 != config.frozen_model_sha256:
        raise ValueError("Attempt10 ranker artifact hash disagrees with config")
    _require_stage9f_p2_policies(t2_policies)
    legal_actions = tuple(
        generate_turn_actions(observation.hero_board, observation.dealt_cards)
    )
    baseline_matches = [
        index
        for index, action in enumerate(legal_actions)
        if action_key(action).to_token() == baseline_action_key
    ]
    if len(baseline_matches) != 1:
        raise ValueError("Attempt10 explicit baseline ActionKey is not uniquely legal")
    baseline_index = baseline_matches[0]
    baseline_token = action_key(legal_actions[baseline_index]).to_token()

    rank_scores = ranker.score_actions(
        observation, legal_actions, baseline_index=baseline_index
    )
    rank_scores.validate(len(legal_actions))
    nonbaseline_indices = [
        index for index in range(len(legal_actions)) if index != baseline_index
    ]
    if len(nonbaseline_indices) < config.candidate_top_k:
        raise ValueError("Attempt10 root has fewer than twelve nonbaseline actions")
    nonbaseline_indices.sort(
        key=lambda index: (
            -float(rank_scores.rank_mean[index]),
            action_key(legal_actions[index]).sort_key(),
        )
    )
    top_indices = tuple(nonbaseline_indices[: config.candidate_top_k])
    proposal_indices = (*top_indices, baseline_index)
    proposal_actions = tuple(legal_actions[index] for index in proposal_indices)
    if len({action_key(action) for action in proposal_actions}) != 13:
        raise AssertionError("Attempt10 proposal set must be top12 plus baseline once")

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
        baseline_position=len(proposal_actions) - 1,
    )
    rerank_order = tuple(
        sorted(
            range(config.candidate_top_k),
            key=lambda position: (
                -float(rerank_rows[position]["paired_delta_vs_baseline"]["mean"]),
                action_key(proposal_actions[position]).sort_key(),
            ),
        )
    )
    top4_positions = rerank_order[: config.rerank_top_k]
    reserve_pool = rerank_order[config.rerank_top_k :]
    if len(reserve_pool) != 8:
        raise AssertionError("Attempt10 reserve pool must be R ranks five through twelve")
    reserve_risks = {
        position: (
            max(
                _lambda_risk_components(
                    rank_scores, proposal_indices[position]
                ).values()
            ),
            _lambda_risk_components(rank_scores, proposal_indices[position]),
        )
        for position in reserve_pool
    }
    reserve_positions = tuple(
        sorted(
            reserve_pool,
            key=lambda position: (
                reserve_risks[position][0],
                action_key(proposal_actions[position]).sort_key(),
            ),
        )[: config.risk_reserve_count]
    )
    k8_set = {*top4_positions, *reserve_positions}
    if len(k8_set) != config.k8_size:
        raise AssertionError("Attempt10 K8 must contain eight unique actions")
    k8_positions = tuple(position for position in rerank_order if position in k8_set)
    k8_actions = tuple(proposal_actions[position] for position in k8_positions)

    veto_actions = (*k8_actions, legal_actions[baseline_index])
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
        veto_actions, veto_result.scores, baseline_position=len(veto_actions) - 1
    )
    veto_checks = [
        _checks(
            row["paired_delta_vs_baseline"],
            min_mean=config.coarse_min_mean,
            min_p05=config.coarse_min_p05,
            min_p01=config.coarse_min_p01,
            min_value=config.coarse_min_value,
        )
        for row in veto_rows[:-1]
    ]
    veto_retained_positions = tuple(
        position for position, checks in enumerate(veto_checks) if all(checks.values())
    )
    veto_retained_actions = tuple(
        veto_actions[position] for position in veto_retained_positions
    )

    stress_retained_actions: tuple[Action, ...] = ()
    if veto_retained_actions:
        stress_actions = (*veto_retained_actions, legal_actions[baseline_index])
        stress_result = _score_phase(
            observation,
            stress_actions,
            phase="stress_x1024",
            seed=config.stress_seed,
            sample_count=config.stress_samples,
            config=config,
            t2_policies=t2_policies,
            prior_rng_keys=prior_rng_keys,
            library=library,
        )
        remember("stress_x1024", stress_result)
        stress_rows = _phase_actions(
            stress_actions,
            stress_result.scores,
            baseline_position=len(stress_actions) - 1,
        )
        stress_checks = [
            _checks(
                row["paired_delta_vs_baseline"],
                min_mean=config.strict_min_mean,
                min_p05=config.strict_min_p05,
                min_p01=config.strict_min_p01,
                min_value=config.strict_min_value,
            )
            for row in stress_rows[:-1]
        ]
        stress_retained_positions = tuple(
            position
            for position, checks in enumerate(stress_checks)
            if all(checks.values())
        )
        stress_retained_actions = tuple(
            stress_actions[position] for position in stress_retained_positions
        )
        stress_payload = {
            **_mapping_payload(stress_actions),
            "opened": True,
            "sample_count": config.stress_samples,
            "configured_sample_count": config.stress_samples,
            "common_random_futures": True,
            "scope": "all_V256_survivors_in_frozen_R_order_plus_explicit_baseline",
            "thresholds": {
                "mean_strictly_greater_than": config.strict_min_mean,
                "p05_at_least": config.strict_min_p05,
                "p01_at_least": config.strict_min_p01,
                "min_at_least": config.strict_min_value,
            },
            "checks_by_position": stress_checks,
            "retained_positions": list(stress_retained_positions),
            "retained_action_keys": [
                action_key(action).to_token() for action in stress_retained_actions
            ],
            "candidate_fallback_allowed": True,
            "actions": stress_rows,
        }
    else:
        stress_payload = _closed_phase(
            reason="not_opened_because_V256_retained_no_candidate",
            sample_count=config.stress_samples,
        )
        stress_payload.update(
            {
                "thresholds": {
                    "mean_strictly_greater_than": config.strict_min_mean,
                    "p05_at_least": config.strict_min_p05,
                    "p01_at_least": config.strict_min_p01,
                    "min_at_least": config.strict_min_value,
                },
                "checks_by_position": [],
                "retained_positions": [],
                "candidate_fallback_allowed": True,
            }
        )

    confirmation_selected_action: Action | None = None
    confirmation_selected_position: int | None = None
    confirmation_retained_positions: tuple[int, ...] = ()
    if stress_retained_actions:
        confirmation_actions = (*stress_retained_actions, legal_actions[baseline_index])
        confirmation_result = _score_phase(
            observation,
            confirmation_actions,
            phase="confirmation_c512",
            seed=config.confirmation_seed,
            sample_count=config.confirmation_samples,
            config=config,
            t2_policies=t2_policies,
            prior_rng_keys=prior_rng_keys,
            library=library,
        )
        remember("confirmation_c512", confirmation_result)
        confirmation_rows = _phase_actions(
            confirmation_actions,
            confirmation_result.scores,
            baseline_position=len(confirmation_actions) - 1,
        )
        confirmation_checks = [
            _checks(
                row["paired_delta_vs_baseline"],
                min_mean=config.strict_min_mean,
                min_p05=config.strict_min_p05,
                min_p01=config.strict_min_p01,
                min_value=config.strict_min_value,
            )
            for row in confirmation_rows[:-1]
        ]
        confirmation_retained_positions = tuple(
            position
            for position, checks in enumerate(confirmation_checks)
            if all(checks.values())
        )
        risk_by_position = {
            position: _observed_tail_risk(
                confirmation_rows[position]["paired_delta_vs_baseline"]
            )
            for position in confirmation_retained_positions
        }
        if confirmation_retained_positions:
            confirmation_selected_position = min(
                confirmation_retained_positions,
                key=lambda position: (
                    risk_by_position[position][0],
                    -float(
                        confirmation_rows[position]["paired_delta_vs_baseline"]["mean"]
                    ),
                    position,
                    action_key(confirmation_actions[position]).sort_key(),
                ),
            )
            confirmation_selected_action = confirmation_actions[
                confirmation_selected_position
            ]
        confirmation_payload = {
            **_mapping_payload(confirmation_actions),
            "opened": True,
            "sample_count": config.confirmation_samples,
            "configured_sample_count": config.confirmation_samples,
            "common_random_futures": True,
            "scope": "all_X1024_survivors_in_frozen_R_order_plus_explicit_baseline",
            "thresholds": {
                "mean_strictly_greater_than": config.strict_min_mean,
                "p05_at_least": config.strict_min_p05,
                "p01_at_least": config.strict_min_p01,
                "min_at_least": config.strict_min_value,
            },
            "checks_by_position": confirmation_checks,
            "retained_positions": list(confirmation_retained_positions),
            "retained_action_keys": [
                action_key(confirmation_actions[position]).to_token()
                for position in confirmation_retained_positions
            ],
            "normalized_tail_risk_by_position": {
                str(position): {
                    "score": risk_by_position[position][0],
                    "components": risk_by_position[position][1],
                }
                for position in confirmation_retained_positions
            },
            "selected_position": confirmation_selected_position,
            "selected_action_key": (
                action_key(confirmation_selected_action).to_token()
                if confirmation_selected_action is not None
                else None
            ),
            "selection_rule": (
                "min_normalized_tail_risk_then_mean_desc_then_frozen_R_order_"
                "then_ActionKey_else_baseline"
            ),
            "actions": confirmation_rows,
        }
    else:
        confirmation_payload = _closed_phase(
            reason="not_opened_because_X1024_retained_no_candidate",
            sample_count=config.confirmation_samples,
        )
        confirmation_payload.update(
            {
                "thresholds": {
                    "mean_strictly_greater_than": config.strict_min_mean,
                    "p05_at_least": config.strict_min_p05,
                    "p01_at_least": config.strict_min_p01,
                    "min_at_least": config.strict_min_value,
                },
                "checks_by_position": [],
                "retained_positions": [],
                "normalized_tail_risk_by_position": {},
                "selected_position": None,
                "selected_action_key": None,
                "selection_rule": (
                    "min_normalized_tail_risk_then_mean_desc_then_frozen_R_order_"
                    "then_ActionKey_else_baseline"
                ),
            }
        )

    final_action = confirmation_selected_action or legal_actions[baseline_index]
    final_token = action_key(final_action).to_token()
    override_fired = final_token != baseline_token
    if not veto_retained_actions:
        fallback_reason = "no_v256_candidate_passed"
    elif not stress_retained_actions:
        fallback_reason = "no_x1024_candidate_passed"
    elif confirmation_selected_action is None:
        fallback_reason = "no_c512_candidate_passed"
    else:
        fallback_reason = None

    if override_fired:
        evaluation_actions = (final_action, legal_actions[baseline_index])
        evaluation_result = _score_phase(
            observation,
            evaluation_actions,
            phase="evaluation_e256",
            seed=config.evaluation_seed,
            sample_count=config.evaluation_samples,
            config=config,
            t2_policies=t2_policies,
            prior_rng_keys=prior_rng_keys,
            library=library,
        )
        remember("evaluation_e256", evaluation_result)
        evaluation_rows = _phase_actions(
            evaluation_actions, evaluation_result.scores, baseline_position=1
        )
        means = tuple(row.mean for row in evaluation_result.scores)
        best_mean = max(means)
        evaluation_best_position = min(
            (index for index, value in enumerate(means) if value == best_mean),
            key=lambda position: action_key(evaluation_actions[position]).sort_key(),
        )
        evaluation_payload = {
            **_mapping_payload(evaluation_actions),
            "opened": True,
            "sample_count": config.evaluation_samples,
            "configured_sample_count": config.evaluation_samples,
            "common_random_futures": True,
            "scope": "locked_final_nonbaseline_plus_explicit_baseline",
            "locked_final_action_key": final_token,
            "sample_best_action_key": action_key(
                evaluation_actions[evaluation_best_position]
            ).to_token(),
            "diagnostics_only": True,
            "can_rerank_or_gate": False,
            "decision_frozen_before_namespace_open": True,
            "retained_action_keys": [final_token],
            "actions": evaluation_rows,
        }
    else:
        evaluation_payload = _closed_phase(
            reason="not_opened_because_final_output_is_baseline",
            sample_count=config.evaluation_samples,
        )
        evaluation_payload.update(
            {
                "locked_final_action_key": final_token,
                "sample_best_action_key": None,
                "diagnostics_only": True,
                "can_rerank_or_gate": False,
                "decision_frozen_before_namespace_open": True,
            }
        )

    legal_rows: list[dict[str, Any]] = []
    top_set = set(top_indices)
    for index, action in enumerate(legal_actions):
        components = _lambda_risk_components(rank_scores, index)
        legal_rows.append(
            {
                "original_legal_index": index,
                "action_key": action_key(action).to_token(),
                "legal": True,
                "illegal_action_masked": False,
                "model_rank_mean": float(rank_scores.rank_mean[index]),
                "model_rank_disagreement": float(rank_scores.rank_disagreement[index]),
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
                "in_learned_top12": index in top_set,
                "is_explicit_baseline": index == baseline_index,
            }
        )

    payload: dict[str, Any] = {
        "status": "ok",
        "schema": ATTEMPT10_TEACHER_SCHEMA,
        "solver_id": ATTEMPT10_SOLVER_ID,
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
            "profile_runtime_feature": False,
            "runtime_authorized": False,
        },
        "legal_action_mapping": _mapping_payload(legal_actions),
        "legal_action_mask": [True for _ in legal_actions],
        "illegal_action_mask": [False for _ in legal_actions],
        "legal_actions": legal_rows,
        "baseline_action_key": baseline_token,
        "baseline_original_legal_index": baseline_index,
        "learned_top12_original_legal_indices": list(top_indices),
        "learned_top12_action_keys": [
            action_key(legal_actions[index]).to_token() for index in top_indices
        ],
        "proposal_mapping": _mapping_payload(proposal_actions),
        "rerank": {
            **_mapping_payload(proposal_actions),
            "sample_count": config.rerank_samples,
            "common_random_futures": True,
            "ordered_nonbaseline_proposal_positions": list(rerank_order),
            "ordered_nonbaseline_action_keys": [
                action_key(proposal_actions[position]).to_token()
                for position in rerank_order
            ],
            "actions": rerank_rows,
        },
        "k8": {
            **_mapping_payload(k8_actions),
            "selection_rule": "R_top4_plus_four_lowest_Lambda_risk_from_R_ranks_5_to_12",
            "top4_rerank_positions": list(top4_positions),
            "risk_reserve_rerank_positions": list(reserve_positions),
            "risk_reserve_action_keys": [
                action_key(proposal_actions[position]).to_token()
                for position in reserve_positions
            ],
            "risk_reserve_normalized_scores": {
                str(position): reserve_risks[position][0]
                for position in reserve_positions
            },
            "risk_reserve_raw_components": {
                str(position): reserve_risks[position][1]
                for position in reserve_positions
            },
            "veto_traversal_rule": "frozen_R_order",
            "veto_traversal_rerank_positions": list(k8_positions),
        },
        "veto": {
            **_mapping_payload(veto_actions),
            "opened": True,
            "sample_count": config.veto_samples,
            "configured_sample_count": config.veto_samples,
            "common_random_futures": True,
            "scope": "frozen_K8_in_R_order_plus_explicit_baseline",
            "thresholds": {
                "mean_strictly_greater_than": config.coarse_min_mean,
                "p05_at_least": config.coarse_min_p05,
                "p01_at_least": config.coarse_min_p01,
                "min_at_least": config.coarse_min_value,
            },
            "checks_by_traversal_position": veto_checks,
            "retained_traversal_positions": list(veto_retained_positions),
            "retained_action_keys": [
                action_key(action).to_token() for action in veto_retained_actions
            ],
            "selection_rule": "retain_all_coarse_safe_in_frozen_R_order",
            "actions": veto_rows,
        },
        "stress": stress_payload,
        "confirmation": confirmation_payload,
        "decision": {
            "final_selected_action_key": final_token,
            "override_fired": override_fired,
            "exact_baseline_fallback": not override_fired,
            "fallback_reason": fallback_reason,
            "candidate_fallback_across_V_X_C_allowed": True,
            "frozen_before_evaluation_namespace_open": True,
        },
        "evaluation": evaluation_payload,
        "belief_digests": belief_digests,
        "rng_key_digests": rng_key_digests,
        "phase_child_information_set_counts": child_counts,
        "sample_independence": (
            "pairwise_disjoint_R128_V256_optional_X1024_optional_C512_"
            "optional_fire_E256_particle_rng_keys"
        ),
        "seed_domain_provenance": {
            "domain_order": list(ATTEMPT10_RNG_DOMAINS),
            "hand_external": config.hand_seed,
            "rerank_r128": config.rerank_seed,
            "veto_v256": config.veto_seed,
            "stress_x1024": config.stress_seed,
            "confirmation_c512": config.confirmation_seed,
            "evaluation_e256": config.evaluation_seed,
            "child_policy": config.child_policy_seed,
            "all_seven_base_seeds_pairwise_distinct": True,
            "hand_sampled_inside_teacher": False,
        },
        "root_selection_lock": (
            "top12_before_R128_K8_before_V256_all_before_X1024_all_before_"
            "C512_tailrisk_lock_then_final_before_diagnostic_E256"
        ),
        "search_config": {
            "learned_nonbaseline_top_k": config.candidate_top_k,
            "baseline_added_exactly_once": True,
            "rerank_samples": config.rerank_samples,
            "rerank_top_k": config.rerank_top_k,
            "risk_reserve_count": config.risk_reserve_count,
            "k8_size": config.k8_size,
            "veto_samples": config.veto_samples,
            "stress_samples": config.stress_samples,
            "confirmation_samples": config.confirmation_samples,
            "evaluation_samples": config.evaluation_samples,
            "hand_seed": config.hand_seed,
            "rerank_seed": config.rerank_seed,
            "veto_seed": config.veto_seed,
            "stress_seed": config.stress_seed,
            "confirmation_seed": config.confirmation_seed,
            "evaluation_seed": config.evaluation_seed,
            "child_policy_seed": config.child_policy_seed,
            "run_id": config.run_id,
            "batch_child_selectors": config.batch_child_selectors,
        },
        "continuation_policy": {
            "t2_policy_id": ATTEMPT10_T2_POLICY_ID,
            "t2_resolution": "explicit_profile_never_current",
            "t3_selector": "m3_rust_evaluate_t3",
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
    validate_attempt10_teacher_output(
        observation,
        baseline_action_key=baseline_token,
        payload=payload,
        config=config,
    )
    return payload


def _phase_raw_contract(
    phase: Mapping[str, Any],
    *,
    sample_count: int,
    baseline_token: str,
    legal_by_token: Mapping[str, Action],
    expected_action_keys: Sequence[str],
) -> None:
    keys = phase.get("action_keys")
    rows = phase.get("actions")
    if not isinstance(keys, list) or not isinstance(rows, list) or len(keys) != len(rows):
        raise ValueError("Attempt10 phase action mapping changed")
    if keys != list(expected_action_keys) or len(set(keys)) != len(keys):
        raise ValueError("Attempt10 phase action subset/order changed")
    try:
        mapped_actions = tuple(legal_by_token[token] for token in keys)
    except KeyError as exc:
        raise ValueError("Attempt10 phase contains a nonlegal ActionKey") from exc
    for name, value in _mapping_payload(mapped_actions).items():
        if phase.get(name) != value:
            raise ValueError("Attempt10 phase action mapping digest changed")
    if keys and keys[-1] != baseline_token:
        raise ValueError("Attempt10 explicit baseline must be last in each open phase")
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping) or row.get("phase_position") != position:
            raise ValueError("Attempt10 phase position changed")
        if row.get("action_key") != keys[position]:
            raise ValueError("Attempt10 phase ActionKey mapping changed")
        raw = row.get("raw_paired_deltas_vs_baseline")
        if not isinstance(raw, list) or len(raw) != sample_count:
            raise ValueError("Attempt10 phase raw paired vector changed")
        if not all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in raw
        ):
            raise ValueError("Attempt10 phase raw paired vector is invalid")
        if row.get("raw_paired_deltas_sha256") != _raw_digest(raw):
            raise ValueError("Attempt10 phase raw paired digest changed")
        summary = row.get("paired_delta_vs_baseline")
        if not isinstance(summary, Mapping):
            raise ValueError("Attempt10 phase paired summary changed")
        expected = {
            "mean": float(np.mean(raw)),
            "p05": float(np.quantile(raw, 0.05, method="linear")),
            "p01": float(np.quantile(raw, 0.01, method="linear")),
            "min": float(min(raw)),
        }
        for name, value in expected.items():
            if float(summary.get(name)) != value:
                raise ValueError("Attempt10 phase paired summary changed")
    if rows and any(
        float(value) != 0.0
        for value in rows[-1]["raw_paired_deltas_vs_baseline"]
    ):
        raise ValueError("Attempt10 explicit baseline paired vector changed")


def validate_attempt10_teacher_output(
    observation: ActorObservation,
    *,
    baseline_action_key: str,
    payload: Mapping[str, Any],
    config: Attempt10TeacherConfig,
) -> dict[str, Any]:
    """Recompute the bounded Attempt10 decision and fail closed on drift."""

    require_t1_second_root(observation)
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
    if any(
        token in encoded
        for token in ("opponent_private_discard", "opponent_hidden", '"particles"')
    ):
        raise ValueError("Attempt10 output contains hidden opponent information")
    if (
        payload.get("schema") != ATTEMPT10_TEACHER_SCHEMA
        or payload.get("solver_id") != ATTEMPT10_SOLVER_ID
        or payload.get("status") != "ok"
        or payload.get("policy_observation") != observation.to_dict()
        or payload.get("observation_fingerprint") != observation.fingerprint()
        or payload.get("seat") != "second"
        or payload.get("street") != "T1"
        or payload.get("to_act_order") != "second"
    ):
        raise ValueError("Attempt10 top-level identity changed")
    legal_actions = tuple(
        generate_turn_actions(observation.hero_board, observation.dealt_cards)
    )
    expected_mapping = _mapping_payload(legal_actions)
    if payload.get("legal_action_mapping") != expected_mapping:
        raise ValueError("Attempt10 complete legal action mapping changed")
    if payload.get("legal_action_mask") != [True] * len(legal_actions) or payload.get(
        "illegal_action_mask"
    ) != [False] * len(legal_actions):
        raise ValueError("Attempt10 illegal action masking changed")
    legal_by_token = {
        action_key(action).to_token(): action for action in legal_actions
    }
    if baseline_action_key not in legal_by_token:
        raise ValueError("Attempt10 baseline ActionKey is not legal")
    baseline_index = next(
        index
        for index, action in enumerate(legal_actions)
        if action_key(action).to_token() == baseline_action_key
    )
    if (
        payload.get("baseline_action_key") != baseline_action_key
        or payload.get("baseline_original_legal_index") != baseline_index
    ):
        raise ValueError("Attempt10 baseline mapping changed")
    legal_rows = payload.get("legal_actions")
    if not isinstance(legal_rows, list) or len(legal_rows) != len(legal_actions):
        raise ValueError("Attempt10 legal action rows changed")
    for index, (row, action) in enumerate(zip(legal_rows, legal_actions, strict=True)):
        if (
            not isinstance(row, Mapping)
            or row.get("original_legal_index") != index
            or row.get("action_key") != action_key(action).to_token()
            or row.get("is_explicit_baseline") is not (index == baseline_index)
        ):
            raise ValueError("Attempt10 legal action index/order changed")

    expected_top_indices = sorted(
        (index for index in range(len(legal_actions)) if index != baseline_index),
        key=lambda index: (
            -float(legal_rows[index]["model_rank_mean"]),
            action_key(legal_actions[index]).sort_key(),
        ),
    )[:ATTEMPT10_TOP_K]
    expected_top_keys = [
        action_key(legal_actions[index]).to_token() for index in expected_top_indices
    ]
    if (
        payload.get("learned_top12_original_legal_indices") != expected_top_indices
        or payload.get("learned_top12_action_keys") != expected_top_keys
    ):
        raise ValueError("Attempt10 learned top12 action index/order changed")
    proposal_keys = [*expected_top_keys, baseline_action_key]
    proposal_actions = tuple(legal_by_token[token] for token in proposal_keys)
    if payload.get("proposal_mapping") != _mapping_payload(proposal_actions):
        raise ValueError("Attempt10 proposal mapping changed")

    rerank = payload.get("rerank")
    k8 = payload.get("k8")
    veto = payload.get("veto")
    stress = payload.get("stress")
    confirmation = payload.get("confirmation")
    decision = payload.get("decision")
    evaluation = payload.get("evaluation")
    if not all(
        isinstance(value, Mapping)
        for value in (rerank, k8, veto, stress, confirmation, decision, evaluation)
    ):
        raise ValueError("Attempt10 phase payload changed")
    _phase_raw_contract(
        rerank,
        sample_count=128,
        baseline_token=baseline_action_key,
        legal_by_token=legal_by_token,
        expected_action_keys=proposal_keys,
    )
    rerank_order = sorted(
        range(ATTEMPT10_TOP_K),
        key=lambda position: (
            -float(rerank["actions"][position]["paired_delta_vs_baseline"]["mean"]),
            action_key(proposal_actions[position]).sort_key(),
        ),
    )
    if (
        rerank.get("ordered_nonbaseline_proposal_positions") != rerank_order
        or rerank.get("ordered_nonbaseline_action_keys")
        != [proposal_keys[position] for position in rerank_order]
    ):
        raise ValueError("Attempt10 frozen R order changed")
    top4_positions = rerank_order[:4]
    reserve_pool = rerank_order[4:]
    reserve_positions = sorted(
        reserve_pool,
        key=lambda position: (
            float(
                legal_rows[expected_top_indices[position]][
                    "normalized_raw_risk_score"
                ]
            ),
            action_key(proposal_actions[position]).sort_key(),
        ),
    )[:4]
    k8_set = {*top4_positions, *reserve_positions}
    k8_positions = [position for position in rerank_order if position in k8_set]
    k8_keys = [proposal_keys[position] for position in k8_positions]
    if (
        k8.get("top4_rerank_positions") != top4_positions
        or k8.get("risk_reserve_rerank_positions") != reserve_positions
        or k8.get("veto_traversal_rerank_positions") != k8_positions
        or any(
            k8.get(name) != value
            for name, value in _mapping_payload(
                tuple(legal_by_token[token] for token in k8_keys)
            ).items()
        )
    ):
        raise ValueError("Attempt10 K8 subset/order changed")

    veto_keys = [*k8_keys, baseline_action_key]
    _phase_raw_contract(
        veto,
        sample_count=256,
        baseline_token=baseline_action_key,
        legal_by_token=legal_by_token,
        expected_action_keys=veto_keys,
    )
    veto_expected = [
        position
        for position, row in enumerate(veto["actions"][:-1])
        if all(
            _checks(
                row["paired_delta_vs_baseline"],
                min_mean=config.coarse_min_mean,
                min_p05=config.coarse_min_p05,
                min_p01=config.coarse_min_p01,
                min_value=config.coarse_min_value,
            ).values()
        )
    ]
    veto_expected_keys = [veto_keys[position] for position in veto_expected]
    if (
        veto.get("retained_traversal_positions") != veto_expected
        or veto.get("retained_action_keys") != veto_expected_keys
    ):
        raise ValueError("Attempt10 V256 retained set changed")
    if stress.get("opened") is not bool(veto_expected):
        raise ValueError("Attempt10 X1024 conditional-open boundary changed")

    stress_expected: list[int] = []
    stress_expected_keys: list[str] = []
    if stress.get("opened"):
        stress_keys = [*veto_expected_keys, baseline_action_key]
        _phase_raw_contract(
            stress,
            sample_count=1024,
            baseline_token=baseline_action_key,
            legal_by_token=legal_by_token,
            expected_action_keys=stress_keys,
        )
        stress_expected = [
            position
            for position, row in enumerate(stress["actions"][:-1])
            if all(
                _checks(
                    row["paired_delta_vs_baseline"],
                    min_mean=config.strict_min_mean,
                    min_p05=config.strict_min_p05,
                    min_p01=config.strict_min_p01,
                    min_value=config.strict_min_value,
                ).values()
            )
        ]
        stress_expected_keys = [stress_keys[position] for position in stress_expected]
        if (
            stress.get("retained_positions") != stress_expected
            or stress.get("retained_action_keys") != stress_expected_keys
        ):
            raise ValueError("Attempt10 X1024 retained set changed")
    elif stress.get("actions") != []:
        raise ValueError("Attempt10 closed X1024 retained action rows")
    if confirmation.get("opened") is not bool(stress_expected):
        raise ValueError("Attempt10 C512 conditional-open boundary changed")

    selected = baseline_action_key
    if confirmation.get("opened"):
        confirmation_keys = [*stress_expected_keys, baseline_action_key]
        _phase_raw_contract(
            confirmation,
            sample_count=512,
            baseline_token=baseline_action_key,
            legal_by_token=legal_by_token,
            expected_action_keys=confirmation_keys,
        )
        confirmation_expected = [
            position
            for position, row in enumerate(confirmation["actions"][:-1])
            if all(
                _checks(
                    row["paired_delta_vs_baseline"],
                    min_mean=config.strict_min_mean,
                    min_p05=config.strict_min_p05,
                    min_p01=config.strict_min_p01,
                    min_value=config.strict_min_value,
                ).values()
            )
        ]
        confirmation_expected_keys = [
            confirmation_keys[position] for position in confirmation_expected
        ]
        risk_by_position = {
            position: _observed_tail_risk(
                confirmation["actions"][position]["paired_delta_vs_baseline"]
            )
            for position in confirmation_expected
        }
        selected_position = (
            min(
                confirmation_expected,
                key=lambda position: (
                    risk_by_position[position][0],
                    -float(
                        confirmation["actions"][position][
                            "paired_delta_vs_baseline"
                        ]["mean"]
                    ),
                    position,
                    action_key(legal_by_token[confirmation_keys[position]]).sort_key(),
                ),
            )
            if confirmation_expected
            else None
        )
        expected_risk_payload = {
            str(position): {
                "score": risk_by_position[position][0],
                "components": risk_by_position[position][1],
            }
            for position in confirmation_expected
        }
        if (
            confirmation.get("retained_positions") != confirmation_expected
            or confirmation.get("retained_action_keys") != confirmation_expected_keys
            or confirmation.get("normalized_tail_risk_by_position")
            != expected_risk_payload
            or confirmation.get("selected_position") != selected_position
        ):
            raise ValueError("Attempt10 C512 risk-lock selection changed")
        if selected_position is not None:
            selected = confirmation_keys[selected_position]
        if confirmation.get("selected_action_key") != (
            selected if selected_position is not None else None
        ):
            raise ValueError("Attempt10 C512 selected ActionKey changed")
    elif confirmation.get("actions") != []:
        raise ValueError("Attempt10 closed C512 retained action rows")

    if decision.get("final_selected_action_key") != selected:
        raise ValueError("Attempt10 locked final action changed")
    fired = selected != baseline_action_key
    expected_reason = None
    if not veto_expected:
        expected_reason = "no_v256_candidate_passed"
    elif not stress_expected:
        expected_reason = "no_x1024_candidate_passed"
    elif selected == baseline_action_key:
        expected_reason = "no_c512_candidate_passed"
    if (
        decision.get("override_fired") is not fired
        or decision.get("exact_baseline_fallback") is fired
        or decision.get("fallback_reason") != expected_reason
    ):
        raise ValueError("Attempt10 decision fallback contract changed")
    if evaluation.get("opened"):
        _phase_raw_contract(
            evaluation,
            sample_count=256,
            baseline_token=baseline_action_key,
            legal_by_token=legal_by_token,
            expected_action_keys=[selected, baseline_action_key],
        )
    elif evaluation.get("actions") != []:
        raise ValueError("Attempt10 closed E256 retained action rows")
    if (
        evaluation.get("opened") is not fired
        or evaluation.get("locked_final_action_key") != selected
        or evaluation.get("diagnostics_only") is not True
        or evaluation.get("can_rerank_or_gate") is not False
    ):
        raise ValueError("Attempt10 E256 diagnostic lock changed")

    key_sets = [set(values) for values in payload.get("rng_key_digests", {}).values()]
    for left_index, left in enumerate(key_sets):
        for right in key_sets[left_index + 1 :]:
            if not left.isdisjoint(right):
                raise ValueError("Attempt10 phase RNG namespaces overlap")
    provenance = payload.get("seed_domain_provenance")
    if not isinstance(provenance, Mapping) or provenance.get("domain_order") != list(
        ATTEMPT10_RNG_DOMAINS
    ):
        raise ValueError("Attempt10 seven-domain seed provenance changed")
    expected_provenance = {
        "hand_external": config.hand_seed,
        "rerank_r128": config.rerank_seed,
        "veto_v256": config.veto_seed,
        "stress_x1024": config.stress_seed,
        "confirmation_c512": config.confirmation_seed,
        "evaluation_e256": config.evaluation_seed,
        "child_policy": config.child_policy_seed,
    }
    if any(provenance.get(name) != value for name, value in expected_provenance.items()):
        raise ValueError("Attempt10 seed provenance changed")
    return {
        "schema": "hu_m43_attempt10_teacher_validation_v1",
        "selected_action_key": selected,
        "override_fired": fired,
        "exact_baseline_fallback": not fired,
        "opened_phases": list(payload.get("rng_key_digests", {})),
    }


__all__ = [
    "ATTEMPT10_COARSE_MIN_MEAN",
    "ATTEMPT10_COARSE_MIN_P01",
    "ATTEMPT10_COARSE_MIN_P05",
    "ATTEMPT10_COARSE_MIN_VALUE",
    "ATTEMPT10_CONFIRMATION_SAMPLES",
    "ATTEMPT10_EVALUATION_SAMPLES",
    "ATTEMPT10_FROZEN_MODEL_ID",
    "ATTEMPT10_FROZEN_MODEL_SHA256",
    "ATTEMPT10_K8",
    "ATTEMPT10_RERANK_SAMPLES",
    "ATTEMPT10_RNG_DOMAINS",
    "ATTEMPT10_SOLVER_ID",
    "ATTEMPT10_STRESS_SAMPLES",
    "ATTEMPT10_STRICT_MIN_MEAN",
    "ATTEMPT10_STRICT_MIN_P01",
    "ATTEMPT10_STRICT_MIN_P05",
    "ATTEMPT10_STRICT_MIN_VALUE",
    "ATTEMPT10_TEACHER_SCHEMA",
    "ATTEMPT10_TOP_K",
    "ATTEMPT10_VETO_SAMPLES",
    "Attempt10RankScores",
    "Attempt10Ranker",
    "Attempt10TeacherConfig",
    "FrozenAttempt10LambdaRanker",
    "evaluate_attempt10_t1_second",
    "validate_attempt10_teacher_output",
]
