"""Bounded, profile-blind Attempt12 T1-second search teacher.

Attempt12 fixes the two structural weaknesses exposed by Attempt11 without
using the closed Attempt11 population to select a threshold:

* every unique legal nonbaseline action plus the explicit baseline is reranked
  with R128; Lambda is used only for the raw-risk reserve;
* K=min(8,n) contains the first min(4,n) actions in frozen R order and the
  lowest-Lambda-risk actions from the remaining R positions;
* V256 is the sole coarse filter.  Its raw minimum is diagnostic and cannot
  reject an action;
* every V survivor is evaluated on both independent X1024 and C1024 batches
  in exactly the same frozen order;
* X and C never filter independently.  Their paired vectors are pooled into a
  single 2,048-sample eligibility and tail-risk decision; and
* E512 is opened only after the action lock and is diagnostic-only.

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


ATTEMPT12_TEACHER_SCHEMA = (
    "hu_m43_attempt12_t1_second_alllegal_r128_kmin8_v256_x1024_c1024_"
    "pooled2048_e512_v1"
)
ATTEMPT12_SOLVER_ID = (
    "alllegal_r128_head4_lambda_riskfill_kmin8_v256_coarse_min_diagnostic_"
    "parallel_x1024_c1024_pooled2048_tailrisk_lock_e512_m3_mc1_v1"
)
ATTEMPT12_FROZEN_MODEL_SHA256 = ATTEMPT08_FROZEN_MODEL_SHA256
ATTEMPT12_FROZEN_MODEL_ID = ATTEMPT08_FROZEN_MODEL_ID
ATTEMPT12_T2_POLICY_ID = ATTEMPT08_T2_POLICY_ID

ATTEMPT12_ALL_LEGAL_CANDIDATES = True
ATTEMPT12_RERANK_SAMPLES = 128
ATTEMPT12_RERANK_HEAD_MAX = 4
ATTEMPT12_SHORTLIST_MAX = 8
ATTEMPT12_VETO_SAMPLES = 256
ATTEMPT12_STRESS_SAMPLES = 1024
ATTEMPT12_CONFIRMATION_SAMPLES = 1024
ATTEMPT12_EVALUATION_SAMPLES = 512
ATTEMPT12_POOLED_SAMPLES = ATTEMPT12_STRESS_SAMPLES + ATTEMPT12_CONFIRMATION_SAMPLES

ATTEMPT12_COARSE_MIN_MEAN = 0.0
ATTEMPT12_COARSE_MIN_P05 = -25.0
ATTEMPT12_COARSE_MIN_P01 = -40.0
ATTEMPT12_COARSE_MIN_VALUE = -50.0
ATTEMPT12_STRICT_MIN_MEAN = 0.0
ATTEMPT12_STRICT_MIN_P05 = -22.0
ATTEMPT12_STRICT_MIN_P01 = -36.0
ATTEMPT12_STRICT_MIN_VALUE = -45.0

ATTEMPT12_RNG_DOMAINS = (
    "hand_external",
    "rerank_r128",
    "veto_v256",
    "stress_x1024",
    "confirmation_c1024",
    "evaluation_e512",
    "child_policy",
)

Attempt12RankScores = Attempt08RankScores
FrozenAttempt12LambdaRanker = FrozenAttempt08LambdaRanker


class Attempt12Ranker(Protocol):
    artifact_sha256: str
    model_id: str

    def score_actions(
        self,
        observation: ActorObservation,
        actions: Sequence[Action],
        *,
        baseline_index: int,
    ) -> Attempt12RankScores: ...


@dataclass(frozen=True)
class Attempt12TeacherConfig:
    """Hard-locked Attempt12 configuration for one canonical root."""

    frozen_model_sha256: str
    hand_seed: int
    rerank_seed: int
    veto_seed: int
    stress_seed: int
    confirmation_seed: int
    evaluation_seed: int
    child_policy_seed: int
    run_id: str
    all_legal_candidates: bool = ATTEMPT12_ALL_LEGAL_CANDIDATES
    rerank_samples: int = ATTEMPT12_RERANK_SAMPLES
    rerank_head_max: int = ATTEMPT12_RERANK_HEAD_MAX
    shortlist_max: int = ATTEMPT12_SHORTLIST_MAX
    veto_samples: int = ATTEMPT12_VETO_SAMPLES
    stress_samples: int = ATTEMPT12_STRESS_SAMPLES
    confirmation_samples: int = ATTEMPT12_CONFIRMATION_SAMPLES
    evaluation_samples: int = ATTEMPT12_EVALUATION_SAMPLES
    coarse_min_mean: float = ATTEMPT12_COARSE_MIN_MEAN
    coarse_min_p05: float = ATTEMPT12_COARSE_MIN_P05
    coarse_min_p01: float = ATTEMPT12_COARSE_MIN_P01
    coarse_min_value: float = ATTEMPT12_COARSE_MIN_VALUE
    strict_min_mean: float = ATTEMPT12_STRICT_MIN_MEAN
    strict_min_p05: float = ATTEMPT12_STRICT_MIN_P05
    strict_min_p01: float = ATTEMPT12_STRICT_MIN_P01
    strict_min_value: float = ATTEMPT12_STRICT_MIN_VALUE
    t2_policy_id: str = ATTEMPT12_T2_POLICY_ID
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
        if self.frozen_model_sha256 != ATTEMPT12_FROZEN_MODEL_SHA256:
            raise ValueError("Attempt12 requires the frozen Lambda raw-risk artifact")
        fixed_ints = {
            "rerank_samples": ATTEMPT12_RERANK_SAMPLES,
            "rerank_head_max": ATTEMPT12_RERANK_HEAD_MAX,
            "shortlist_max": ATTEMPT12_SHORTLIST_MAX,
            "veto_samples": ATTEMPT12_VETO_SAMPLES,
            "stress_samples": ATTEMPT12_STRESS_SAMPLES,
            "confirmation_samples": ATTEMPT12_CONFIRMATION_SAMPLES,
            "evaluation_samples": ATTEMPT12_EVALUATION_SAMPLES,
            "t3_candidate_samples": 1,
            "t3_evaluation_samples": 1,
            "t3_downstream_samples": 1,
            "t4_candidate_samples": 1,
            "t4_evaluation_samples": 1,
        }
        for name, expected in fixed_ints.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value != expected:
                raise ValueError(f"Attempt12 {name} is fixed at {expected}")
        if self.all_legal_candidates is not True:
            raise ValueError("Attempt12 requires every unique legal nonbaseline action")
        fixed_floats = {
            "coarse_min_mean": ATTEMPT12_COARSE_MIN_MEAN,
            "coarse_min_p05": ATTEMPT12_COARSE_MIN_P05,
            "coarse_min_p01": ATTEMPT12_COARSE_MIN_P01,
            "coarse_min_value": ATTEMPT12_COARSE_MIN_VALUE,
            "strict_min_mean": ATTEMPT12_STRICT_MIN_MEAN,
            "strict_min_p05": ATTEMPT12_STRICT_MIN_P05,
            "strict_min_p01": ATTEMPT12_STRICT_MIN_P01,
            "strict_min_value": ATTEMPT12_STRICT_MIN_VALUE,
        }
        for name, expected in fixed_floats.items():
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) != expected
            ):
                raise ValueError(f"Attempt12 {name} is fixed at {expected}")
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
                raise TypeError(f"Attempt12 {name} must be an integer")
        if len({getattr(self, name) for name in seed_names}) != len(seed_names):
            raise ValueError("Attempt12 hand/R/V/X/C/E/child seeds must all be distinct")
        if self.t2_policy_id != ATTEMPT12_T2_POLICY_ID:
            raise ValueError("Attempt12 T2 policy is fixed at stage9f_p2")
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("Attempt12 run_id must not be empty")
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
        raise ValueError("Attempt12 paired deltas must be finite and non-empty")
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
        raise ValueError("Attempt12 phase action mapping is invalid")
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
) -> dict[str, bool]:
    """Return the only three eligibility checks allowed in Attempt12.

    Raw minima are intentionally absent.  They remain attested in every paired
    summary and are exposed separately as diagnostics, but sample-count-driven
    extrema may not filter or rank an action.
    """

    return {
        "mean_gt_0": float(summary["mean"]) > min_mean,
        "p05_at_least": float(summary["p05"]) >= min_p05,
        "p01_at_least": float(summary["p01"]) >= min_p01,
    }


def _diagnostic_min(summary: Mapping[str, Any], *, reference: float) -> dict[str, Any]:
    return {
        "raw_min": float(summary["min"]),
        "reference": float(reference),
        "at_least_reference": float(summary["min"]) >= float(reference),
        "can_filter_or_rank": False,
    }


def _lambda_risk_components(
    rank_scores: Attempt12RankScores, index: int
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
    }
    return max(components.values()), components


def _summary_from_raw(values: Sequence[float]) -> dict[str, Any]:
    raw = tuple(float(value) for value in values)
    if not raw or not all(math.isfinite(value) for value in raw):
        raise ValueError("Attempt12 pooled paired vector must be finite and non-empty")
    zeros = _ActionScores(tuple(0.0 for _ in raw))
    return _paired_delta_summary(_ActionScores(raw), zeros)


def _pooled_actions(
    actions: Sequence[Action],
    stress_rows: Sequence[Mapping[str, Any]],
    confirmation_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not (
        len(actions) == len(stress_rows) == len(confirmation_rows)
        and actions
    ):
        raise ValueError("Attempt12 pooled X/C action mapping is invalid")
    output: list[dict[str, Any]] = []
    for position, (action, stress, confirmation) in enumerate(
        zip(actions, stress_rows, confirmation_rows, strict=True)
    ):
        token = action_key(action).to_token()
        if stress.get("action_key") != token or confirmation.get("action_key") != token:
            raise ValueError("Attempt12 X/C ActionKey order differs before pooling")
        x_raw = stress.get("raw_paired_deltas_vs_baseline")
        c_raw = confirmation.get("raw_paired_deltas_vs_baseline")
        if not isinstance(x_raw, list) or not isinstance(c_raw, list):
            raise ValueError("Attempt12 X/C raw vectors are missing")
        raw = [float(value) for value in (*x_raw, *c_raw)]
        output.append(
            {
                "phase_position": position,
                "action_key": token,
                "is_explicit_baseline": position == len(actions) - 1,
                "paired_delta_vs_baseline": _summary_from_raw(raw),
                "raw_paired_deltas_vs_baseline": raw,
                "raw_paired_deltas_sha256": _raw_digest(raw),
                "source_raw_paired_deltas_sha256": {
                    "stress_x1024": stress["raw_paired_deltas_sha256"],
                    "confirmation_c1024": confirmation[
                        "raw_paired_deltas_sha256"
                    ],
                },
            }
        )
    return output


def _score_phase(
    observation: ActorObservation,
    actions: Sequence[Action],
    *,
    phase: str,
    seed: int,
    sample_count: int,
    config: Attempt12TeacherConfig,
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
        raise ValueError(f"Attempt12 {phase} particle RNG keys are invalid")
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
        raise ValueError(f"Attempt12 {phase} scorer returned the wrong action count")
    for row in scored:
        if len(row.values) != sample_count or not all(
            math.isfinite(float(value)) for value in row.values
        ):
            raise ValueError(f"Attempt12 {phase} scorer returned invalid values")
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


def evaluate_attempt12_t1_second(
    observation: ActorObservation,
    *,
    baseline_action_key: str,
    ranker: Attempt12Ranker,
    t2_policies: Mapping[str, object],
    config: Attempt12TeacherConfig,
    library: Any | None = None,
) -> dict[str, Any]:
    """Evaluate the fixed Attempt12 search without activating a policy."""

    require_t1_second_root(observation)
    if ranker.artifact_sha256 != config.frozen_model_sha256:
        raise ValueError("Attempt12 ranker artifact hash disagrees with config")
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
        raise ValueError("Attempt12 explicit baseline ActionKey is not uniquely legal")
    baseline_index = baseline_matches[0]
    baseline_token = action_key(legal_actions[baseline_index]).to_token()

    rank_scores = ranker.score_actions(
        observation, legal_actions, baseline_index=baseline_index
    )
    rank_scores.validate(len(legal_actions))
    legal_tokens = tuple(action_key(action).to_token() for action in legal_actions)
    if len(set(legal_tokens)) != len(legal_tokens):
        raise ValueError("Attempt12 complete legal action mapping is not unique")
    nonbaseline_indices = [
        index for index in range(len(legal_actions)) if index != baseline_index
    ]
    nonbaseline_indices.sort(
        key=lambda index: action_key(legal_actions[index]).sort_key()
    )
    # Attempt12 opens R128 over the complete legal nonbaseline set.  Lambda
    # rank_mean cannot exclude or reorder a proposal; its raw downside heads
    # are consulted only when filling the risk reserve below.
    top_indices = tuple(nonbaseline_indices)
    candidate_count = len(top_indices)
    proposal_indices = (*top_indices, baseline_index)
    proposal_actions = tuple(legal_actions[index] for index in proposal_indices)
    if len({action_key(action) for action in proposal_actions}) != candidate_count + 1:
        raise AssertionError(
            "Attempt12 proposal set must contain unique candidates plus baseline once"
        )

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
            range(candidate_count),
            key=lambda position: (
                -float(rerank_rows[position]["paired_delta_vs_baseline"]["mean"]),
                action_key(proposal_actions[position]).sort_key(),
            ),
        )
    )
    shortlist_size = min(config.shortlist_max, candidate_count)
    head_count = min(config.rerank_head_max, candidate_count)
    reserve_count = shortlist_size - head_count
    head_positions = rerank_order[:head_count]
    reserve_pool = rerank_order[head_count:]
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
        )[:reserve_count]
    )
    shortlist_set = {*head_positions, *reserve_positions}
    if len(shortlist_set) != shortlist_size:
        raise AssertionError("Attempt12 shortlist must contain K unique actions")
    shortlist_positions = tuple(
        position for position in rerank_order if position in shortlist_set
    )
    shortlist_actions = tuple(
        proposal_actions[position] for position in shortlist_positions
    )

    veto_actions = (*shortlist_actions, legal_actions[baseline_index])
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
        )
        for row in veto_rows[:-1]
    ]
    veto_min_diagnostics = [
        _diagnostic_min(
            row["paired_delta_vs_baseline"], reference=config.coarse_min_value
        )
        for row in veto_rows[:-1]
    ]
    veto_retained_positions = tuple(
        position for position, checks in enumerate(veto_checks) if all(checks.values())
    )
    veto_retained_actions = tuple(
        veto_actions[position] for position in veto_retained_positions
    )

    pooled_selected_action: Action | None = None
    pooled_selected_position: int | None = None
    pooled_eligible_positions: tuple[int, ...] = ()
    if veto_retained_actions:
        # X and C deliberately receive the identical frozen V survivor scope.
        # Neither phase can filter, rerank, promote, or select by itself.
        pooled_scope_actions = (*veto_retained_actions, legal_actions[baseline_index])
        stress_result = _score_phase(
            observation,
            pooled_scope_actions,
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
            pooled_scope_actions,
            stress_result.scores,
            baseline_position=len(pooled_scope_actions) - 1,
        )
        confirmation_result = _score_phase(
            observation,
            pooled_scope_actions,
            phase="confirmation_c1024",
            seed=config.confirmation_seed,
            sample_count=config.confirmation_samples,
            config=config,
            t2_policies=t2_policies,
            prior_rng_keys=prior_rng_keys,
            library=library,
        )
        remember("confirmation_c1024", confirmation_result)
        confirmation_rows = _phase_actions(
            pooled_scope_actions,
            confirmation_result.scores,
            baseline_position=len(pooled_scope_actions) - 1,
        )
        pooled_rows = _pooled_actions(
            pooled_scope_actions, stress_rows, confirmation_rows
        )
        pooled_checks = [
            _checks(
                row["paired_delta_vs_baseline"],
                min_mean=config.strict_min_mean,
                min_p05=config.strict_min_p05,
                min_p01=config.strict_min_p01,
            )
            for row in pooled_rows[:-1]
        ]
        pooled_min_diagnostics = [
            _diagnostic_min(
                row["paired_delta_vs_baseline"], reference=config.strict_min_value
            )
            for row in pooled_rows[:-1]
        ]
        pooled_eligible_positions = tuple(
            position
            for position, checks in enumerate(pooled_checks)
            if all(checks.values())
        )
        risk_by_position = {
            position: _observed_tail_risk(
                pooled_rows[position]["paired_delta_vs_baseline"]
            )
            for position in pooled_eligible_positions
        }
        if pooled_eligible_positions:
            pooled_selected_position = min(
                pooled_eligible_positions,
                key=lambda position: (
                    risk_by_position[position][0],
                    -float(
                        pooled_rows[position]["paired_delta_vs_baseline"]["mean"]
                    ),
                    position,
                    action_key(pooled_scope_actions[position]).sort_key(),
                ),
            )
            pooled_selected_action = pooled_scope_actions[pooled_selected_position]
        common_phase_payload = {
            "opened": True,
            "scope": (
                "all_V256_survivors_in_frozen_R_order_plus_explicit_baseline"
            ),
            "filtering_allowed": False,
            "reranking_allowed": False,
            "selection_allowed": False,
            "pooled_decision_input": True,
            "retained_positions": list(range(len(veto_retained_actions))),
            "retained_action_keys": [
                action_key(action).to_token() for action in veto_retained_actions
            ],
        }
        stress_payload = {
            **_mapping_payload(pooled_scope_actions),
            **common_phase_payload,
            "sample_count": config.stress_samples,
            "configured_sample_count": config.stress_samples,
            "common_random_futures": True,
            "actions": stress_rows,
        }
        confirmation_payload = {
            **_mapping_payload(pooled_scope_actions),
            **common_phase_payload,
            "sample_count": config.confirmation_samples,
            "configured_sample_count": config.confirmation_samples,
            "common_random_futures": True,
            "actions": confirmation_rows,
        }
        pooled_payload = {
            **_mapping_payload(pooled_scope_actions),
            "opened": True,
            "sample_count": ATTEMPT12_POOLED_SAMPLES,
            "configured_sample_count": ATTEMPT12_POOLED_SAMPLES,
            "source_phases": ["stress_x1024", "confirmation_c1024"],
            "source_phases_independent": True,
            "common_random_futures_within_each_source_phase": True,
            "scope": "concatenated_X1024_then_C1024_raw_paired_vectors",
            "thresholds": {
                "mean_strictly_greater_than": config.strict_min_mean,
                "p05_at_least": config.strict_min_p05,
                "p01_at_least": config.strict_min_p01,
                "raw_min_reference_diagnostic_only": config.strict_min_value,
            },
            "checks_by_position": pooled_checks,
            "raw_min_diagnostics_by_position": pooled_min_diagnostics,
            "eligible_positions": list(pooled_eligible_positions),
            "eligible_action_keys": [
                action_key(pooled_scope_actions[position]).to_token()
                for position in pooled_eligible_positions
            ],
            "normalized_tail_risk_by_position": {
                str(position): {
                    "score": risk_by_position[position][0],
                    "components": risk_by_position[position][1],
                }
                for position in pooled_eligible_positions
            },
            "selected_position": pooled_selected_position,
            "selected_action_key": (
                action_key(pooled_selected_action).to_token()
                if pooled_selected_action is not None
                else None
            ),
            "selection_rule": (
                "min_normalized_p05_p01_tail_risk_then_pooled_mean_desc_then_"
                "frozen_R_order_then_ActionKey_else_baseline"
            ),
            "raw_min_can_filter_or_rank": False,
            "actions": pooled_rows,
        }
    else:
        stress_payload = _closed_phase(
            reason="not_opened_because_V256_retained_no_candidate",
            sample_count=config.stress_samples,
        )
        confirmation_payload = _closed_phase(
            reason="not_opened_because_V256_retained_no_candidate",
            sample_count=config.confirmation_samples,
        )
        for phase_payload in (stress_payload, confirmation_payload):
            phase_payload.update(
                {
                    "filtering_allowed": False,
                    "reranking_allowed": False,
                    "selection_allowed": False,
                    "pooled_decision_input": True,
                    "retained_positions": [],
                }
            )
        pooled_payload = _closed_phase(
            reason="not_opened_because_V256_retained_no_candidate",
            sample_count=ATTEMPT12_POOLED_SAMPLES,
        )
        pooled_payload.update(
            {
                "source_phases": ["stress_x1024", "confirmation_c1024"],
                "source_phases_independent": True,
                "common_random_futures_within_each_source_phase": True,
                "checks_by_position": [],
                "raw_min_diagnostics_by_position": [],
                "eligible_positions": [],
                "eligible_action_keys": [],
                "normalized_tail_risk_by_position": {},
                "selected_position": None,
                "selected_action_key": None,
                "raw_min_can_filter_or_rank": False,
            }
        )

    final_action = pooled_selected_action or legal_actions[baseline_index]
    final_token = action_key(final_action).to_token()
    override_fired = final_token != baseline_token
    if not veto_retained_actions:
        fallback_reason = "no_v256_candidate_passed"
    elif pooled_selected_action is None:
        fallback_reason = "no_pooled_x1024_c1024_candidate_passed"
    else:
        fallback_reason = None

    if override_fired:
        evaluation_actions = (final_action, legal_actions[baseline_index])
        evaluation_result = _score_phase(
            observation,
            evaluation_actions,
            phase="evaluation_e512",
            seed=config.evaluation_seed,
            sample_count=config.evaluation_samples,
            config=config,
            t2_policies=t2_policies,
            prior_rng_keys=prior_rng_keys,
            library=library,
        )
        remember("evaluation_e512", evaluation_result)
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
                "in_all_legal_candidate_set": index in top_set,
                "is_explicit_baseline": index == baseline_index,
            }
        )

    payload: dict[str, Any] = {
        "status": "ok",
        "schema": ATTEMPT12_TEACHER_SCHEMA,
        "solver_id": ATTEMPT12_SOLVER_ID,
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
            "purpose": "raw_risk_reserve_only_all_legal_R128",
            "profile_runtime_feature": False,
            "runtime_authorized": False,
        },
        "legal_action_mapping": _mapping_payload(legal_actions),
        "legal_action_mask": [True for _ in legal_actions],
        "illegal_action_mask": [False for _ in legal_actions],
        "legal_actions": legal_rows,
        "baseline_action_key": baseline_token,
        "baseline_original_legal_index": baseline_index,
        "candidate_nonbaseline_count": candidate_count,
        "all_legal_nonbaseline_original_legal_indices": list(top_indices),
        "all_legal_nonbaseline_action_keys": [
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
        "shortlist": {
            **_mapping_payload(shortlist_actions),
            "nonbaseline_count": shortlist_size,
            "selection_rule": (
                "R_head_min4_then_minimum_Lambda_risk_fill_to_K_min8_"
                "without_replacement"
            ),
            "head_rerank_positions": list(head_positions),
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
            "veto_traversal_rerank_positions": list(shortlist_positions),
        },
        "veto": {
            **_mapping_payload(veto_actions),
            "opened": True,
            "sample_count": config.veto_samples,
            "configured_sample_count": config.veto_samples,
            "common_random_futures": True,
            "scope": "variable_K_in_R_order_plus_explicit_baseline",
            "thresholds": {
                "mean_strictly_greater_than": config.coarse_min_mean,
                "p05_at_least": config.coarse_min_p05,
                "p01_at_least": config.coarse_min_p01,
                "raw_min_reference_diagnostic_only": config.coarse_min_value,
            },
            "checks_by_traversal_position": veto_checks,
            "raw_min_diagnostics_by_traversal_position": veto_min_diagnostics,
            "raw_min_can_filter_or_rank": False,
            "retained_traversal_positions": list(veto_retained_positions),
            "retained_action_keys": [
                action_key(action).to_token() for action in veto_retained_actions
            ],
            "selection_rule": "retain_all_coarse_safe_in_frozen_R_order",
            "actions": veto_rows,
        },
        "stress": stress_payload,
        "confirmation": confirmation_payload,
        "pooled": pooled_payload,
        "decision": {
            "final_selected_action_key": final_token,
            "override_fired": override_fired,
            "exact_baseline_fallback": not override_fired,
            "fallback_reason": fallback_reason,
            "candidate_fallback_after_V_or_pooled_allowed": True,
            "frozen_before_evaluation_namespace_open": True,
        },
        "evaluation": evaluation_payload,
        "belief_digests": belief_digests,
        "rng_key_digests": rng_key_digests,
        "phase_child_information_set_counts": child_counts,
        "sample_independence": (
            "pairwise_disjoint_R128_V256_optional_X1024_optional_C1024_"
            "optional_fire_E512_particle_rng_keys"
        ),
        "seed_domain_provenance": {
            "domain_order": list(ATTEMPT12_RNG_DOMAINS),
            "hand_external": config.hand_seed,
            "rerank_r128": config.rerank_seed,
            "veto_v256": config.veto_seed,
            "stress_x1024": config.stress_seed,
            "confirmation_c1024": config.confirmation_seed,
            "evaluation_e512": config.evaluation_seed,
            "child_policy": config.child_policy_seed,
            "all_seven_base_seeds_pairwise_distinct": True,
            "hand_sampled_inside_teacher": False,
        },
        "root_selection_lock": (
            "all_unique_legal_nonbaseline_before_R128_Kmin8_before_V256_then_"
            "identical_X1024_C1024_scopes_before_pooled2048_tailrisk_lock_then_"
            "final_before_diagnostic_E512"
        ),
        "search_config": {
            "all_legal_candidates": config.all_legal_candidates,
            "candidate_nonbaseline_count": candidate_count,
            "baseline_added_exactly_once": True,
            "rerank_samples": config.rerank_samples,
            "rerank_head_max": config.rerank_head_max,
            "shortlist_max": config.shortlist_max,
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
            "t2_policy_id": ATTEMPT12_T2_POLICY_ID,
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
    validate_attempt12_teacher_output(
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
        raise ValueError("Attempt12 phase action mapping changed")
    if keys != list(expected_action_keys) or len(set(keys)) != len(keys):
        raise ValueError("Attempt12 phase action subset/order changed")
    try:
        mapped_actions = tuple(legal_by_token[token] for token in keys)
    except KeyError as exc:
        raise ValueError("Attempt12 phase contains a nonlegal ActionKey") from exc
    for name, value in _mapping_payload(mapped_actions).items():
        if phase.get(name) != value:
            raise ValueError("Attempt12 phase action mapping digest changed")
    if keys and keys[-1] != baseline_token:
        raise ValueError("Attempt12 explicit baseline must be last in each open phase")
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping) or row.get("phase_position") != position:
            raise ValueError("Attempt12 phase position changed")
        if row.get("action_key") != keys[position]:
            raise ValueError("Attempt12 phase ActionKey mapping changed")
        raw = row.get("raw_paired_deltas_vs_baseline")
        if not isinstance(raw, list) or len(raw) != sample_count:
            raise ValueError("Attempt12 phase raw paired vector changed")
        if not all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in raw
        ):
            raise ValueError("Attempt12 phase raw paired vector is invalid")
        if row.get("raw_paired_deltas_sha256") != _raw_digest(raw):
            raise ValueError("Attempt12 phase raw paired digest changed")
        summary = row.get("paired_delta_vs_baseline")
        if not isinstance(summary, Mapping):
            raise ValueError("Attempt12 phase paired summary changed")
        expected = {
            "mean": float(np.mean(raw)),
            "p05": float(np.quantile(raw, 0.05, method="linear")),
            "p01": float(np.quantile(raw, 0.01, method="linear")),
            "min": float(min(raw)),
        }
        for name, value in expected.items():
            if float(summary.get(name)) != value:
                raise ValueError("Attempt12 phase paired summary changed")
    if rows and any(
        float(value) != 0.0
        for value in rows[-1]["raw_paired_deltas_vs_baseline"]
    ):
        raise ValueError("Attempt12 explicit baseline paired vector changed")


def _open_phase_metadata(phase: Mapping[str, Any], *, sample_count: int) -> None:
    if (
        phase.get("opened") is not True
        or phase.get("sample_count") != sample_count
        or phase.get("configured_sample_count") != sample_count
        or phase.get("common_random_futures") is not True
    ):
        raise ValueError("Attempt12 open phase sample/count metadata changed")


def _closed_phase_contract(
    phase: Mapping[str, Any], *, configured_sample_count: int
) -> None:
    empty_mapping = _mapping_payload(())
    if (
        any(phase.get(name) != value for name, value in empty_mapping.items())
        or phase.get("opened") is not False
        or phase.get("sample_count") != 0
        or phase.get("configured_sample_count") != configured_sample_count
        or phase.get("common_random_futures") is not False
        or phase.get("retained_action_keys") != []
        or phase.get("actions") != []
    ):
        raise ValueError("Attempt12 closed phase mapping/count contract changed")


def validate_attempt12_teacher_output(
    observation: ActorObservation,
    *,
    baseline_action_key: str,
    payload: Mapping[str, Any],
    config: Attempt12TeacherConfig,
) -> dict[str, Any]:
    """Recompute the bounded Attempt12 decision and fail closed on drift."""

    require_t1_second_root(observation)
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
    if any(
        token in encoded
        for token in ("opponent_private_discard", "opponent_hidden", '"particles"')
    ):
        raise ValueError("Attempt12 output contains hidden opponent information")
    if (
        payload.get("schema") != ATTEMPT12_TEACHER_SCHEMA
        or payload.get("solver_id") != ATTEMPT12_SOLVER_ID
        or payload.get("status") != "ok"
        or payload.get("policy_observation") != observation.to_dict()
        or payload.get("observation_fingerprint") != observation.fingerprint()
        or payload.get("seat") != "second"
        or payload.get("street") != "T1"
        or payload.get("to_act_order") != "second"
    ):
        raise ValueError("Attempt12 top-level identity changed")
    legal_actions = tuple(
        generate_turn_actions(observation.hero_board, observation.dealt_cards)
    )
    expected_mapping = _mapping_payload(legal_actions)
    if payload.get("legal_action_mapping") != expected_mapping:
        raise ValueError("Attempt12 complete legal action mapping changed")
    if payload.get("legal_action_mask") != [True] * len(legal_actions) or payload.get(
        "illegal_action_mask"
    ) != [False] * len(legal_actions):
        raise ValueError("Attempt12 illegal action masking changed")
    legal_by_token = {
        action_key(action).to_token(): action for action in legal_actions
    }
    if len(legal_by_token) != len(legal_actions):
        raise ValueError("Attempt12 complete legal action mapping is not unique")
    if baseline_action_key not in legal_by_token:
        raise ValueError("Attempt12 baseline ActionKey is not legal")
    baseline_index = next(
        index
        for index, action in enumerate(legal_actions)
        if action_key(action).to_token() == baseline_action_key
    )
    if (
        payload.get("baseline_action_key") != baseline_action_key
        or payload.get("baseline_original_legal_index") != baseline_index
    ):
        raise ValueError("Attempt12 baseline mapping changed")
    legal_rows = payload.get("legal_actions")
    if not isinstance(legal_rows, list) or len(legal_rows) != len(legal_actions):
        raise ValueError("Attempt12 legal action rows changed")
    for index, (row, action) in enumerate(zip(legal_rows, legal_actions, strict=True)):
        if (
            not isinstance(row, Mapping)
            or row.get("original_legal_index") != index
            or row.get("action_key") != action_key(action).to_token()
            or row.get("is_explicit_baseline") is not (index == baseline_index)
        ):
            raise ValueError("Attempt12 legal action index/order changed")
    generator = payload.get("frozen_candidate_generator")
    if (
        not isinstance(generator, Mapping)
        or generator.get("artifact_sha256") != config.frozen_model_sha256
        or generator.get("purpose") != "raw_risk_reserve_only_all_legal_R128"
        or generator.get("profile_runtime_feature") is not False
        or generator.get("runtime_authorized") is not False
    ):
        raise ValueError("Attempt12 frozen Lambda raw-risk boundary changed")

    expected_top_indices = sorted(
        (index for index in range(len(legal_actions)) if index != baseline_index),
        key=lambda index: action_key(legal_actions[index]).sort_key(),
    )
    expected_top_keys = [
        action_key(legal_actions[index]).to_token() for index in expected_top_indices
    ]
    if (
        payload.get("candidate_nonbaseline_count") != len(expected_top_indices)
        or payload.get("all_legal_nonbaseline_original_legal_indices")
        != expected_top_indices
        or payload.get("all_legal_nonbaseline_action_keys") != expected_top_keys
    ):
        raise ValueError("Attempt12 all-legal candidate index/order changed")
    proposal_keys = [*expected_top_keys, baseline_action_key]
    proposal_actions = tuple(legal_by_token[token] for token in proposal_keys)
    if payload.get("proposal_mapping") != _mapping_payload(proposal_actions):
        raise ValueError("Attempt12 proposal mapping changed")

    rerank = payload.get("rerank")
    shortlist = payload.get("shortlist")
    veto = payload.get("veto")
    stress = payload.get("stress")
    confirmation = payload.get("confirmation")
    pooled = payload.get("pooled")
    decision = payload.get("decision")
    evaluation = payload.get("evaluation")
    if not all(
        isinstance(value, Mapping)
        for value in (
            rerank,
            shortlist,
            veto,
            stress,
            confirmation,
            pooled,
            decision,
            evaluation,
        )
    ):
        raise ValueError("Attempt12 phase payload changed")
    _phase_raw_contract(
        rerank,
        sample_count=config.rerank_samples,
        baseline_token=baseline_action_key,
        legal_by_token=legal_by_token,
        expected_action_keys=proposal_keys,
    )
    if (
        rerank.get("sample_count") != config.rerank_samples
        or rerank.get("common_random_futures") is not True
    ):
        raise ValueError("Attempt12 R128 sample/count metadata changed")
    rerank_order = sorted(
        range(len(expected_top_indices)),
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
        raise ValueError("Attempt12 frozen R order changed")
    shortlist_size = min(config.shortlist_max, len(expected_top_indices))
    head_count = min(config.rerank_head_max, len(expected_top_indices))
    reserve_count = shortlist_size - head_count
    head_positions = rerank_order[:head_count]
    reserve_pool = rerank_order[head_count:]
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
    )[:reserve_count]
    shortlist_set = {*head_positions, *reserve_positions}
    shortlist_positions = [
        position for position in rerank_order if position in shortlist_set
    ]
    shortlist_keys = [proposal_keys[position] for position in shortlist_positions]
    if (
        shortlist.get("nonbaseline_count") != shortlist_size
        or shortlist.get("head_rerank_positions") != head_positions
        or shortlist.get("risk_reserve_rerank_positions") != reserve_positions
        or shortlist.get("veto_traversal_rerank_positions")
        != shortlist_positions
        or any(
            shortlist.get(name) != value
            for name, value in _mapping_payload(
                tuple(legal_by_token[token] for token in shortlist_keys)
            ).items()
        )
    ):
        raise ValueError("Attempt12 variable shortlist subset/order changed")

    veto_keys = [*shortlist_keys, baseline_action_key]
    _phase_raw_contract(
        veto,
        sample_count=config.veto_samples,
        baseline_token=baseline_action_key,
        legal_by_token=legal_by_token,
        expected_action_keys=veto_keys,
    )
    _open_phase_metadata(veto, sample_count=config.veto_samples)
    veto_expected = [
        position
        for position, row in enumerate(veto["actions"][:-1])
        if all(
            _checks(
                row["paired_delta_vs_baseline"],
                min_mean=config.coarse_min_mean,
                min_p05=config.coarse_min_p05,
                min_p01=config.coarse_min_p01,
            ).values()
        )
    ]
    veto_expected_keys = [veto_keys[position] for position in veto_expected]
    veto_min_diagnostics = [
        _diagnostic_min(
            row["paired_delta_vs_baseline"], reference=config.coarse_min_value
        )
        for row in veto["actions"][:-1]
    ]
    if (
        veto.get("retained_traversal_positions") != veto_expected
        or veto.get("retained_action_keys") != veto_expected_keys
        or veto.get("raw_min_diagnostics_by_traversal_position")
        != veto_min_diagnostics
        or veto.get("raw_min_can_filter_or_rank") is not False
        or veto.get("thresholds")
        != {
            "mean_strictly_greater_than": config.coarse_min_mean,
            "p05_at_least": config.coarse_min_p05,
            "p01_at_least": config.coarse_min_p01,
            "raw_min_reference_diagnostic_only": config.coarse_min_value,
        }
    ):
        raise ValueError("Attempt12 V256 retained set changed")
    opened = bool(veto_expected)
    if stress.get("opened") is not opened or confirmation.get("opened") is not opened:
        raise ValueError("Attempt12 X1024/C1024 conditional-open boundary changed")
    if pooled.get("opened") is not opened:
        raise ValueError("Attempt12 pooled2048 conditional-open boundary changed")

    selected = baseline_action_key
    if opened:
        pooled_scope_keys = [*veto_expected_keys, baseline_action_key]
        expected_positions = list(range(len(veto_expected_keys)))
        for phase, sample_count in (
            (stress, config.stress_samples),
            (confirmation, config.confirmation_samples),
        ):
            _phase_raw_contract(
                phase,
                sample_count=sample_count,
                baseline_token=baseline_action_key,
                legal_by_token=legal_by_token,
                expected_action_keys=pooled_scope_keys,
            )
            _open_phase_metadata(phase, sample_count=sample_count)
            if (
                phase.get("retained_positions") != expected_positions
                or phase.get("retained_action_keys") != veto_expected_keys
                or phase.get("filtering_allowed") is not False
                or phase.get("reranking_allowed") is not False
                or phase.get("selection_allowed") is not False
                or phase.get("pooled_decision_input") is not True
                or phase.get("scope")
                != "all_V256_survivors_in_frozen_R_order_plus_explicit_baseline"
            ):
                raise ValueError("Attempt12 X/C nonfiltering identical-scope contract changed")
        if stress.get("action_keys") != confirmation.get("action_keys"):
            raise ValueError("Attempt12 X/C action scopes differ")
        _phase_raw_contract(
            pooled,
            sample_count=ATTEMPT12_POOLED_SAMPLES,
            baseline_token=baseline_action_key,
            legal_by_token=legal_by_token,
            expected_action_keys=pooled_scope_keys,
        )
        if (
            pooled.get("sample_count") != ATTEMPT12_POOLED_SAMPLES
            or pooled.get("configured_sample_count") != ATTEMPT12_POOLED_SAMPLES
            or pooled.get("source_phases")
            != ["stress_x1024", "confirmation_c1024"]
            or pooled.get("source_phases_independent") is not True
            or pooled.get("common_random_futures_within_each_source_phase")
            is not True
            or pooled.get("raw_min_can_filter_or_rank") is not False
            or pooled.get("thresholds")
            != {
                "mean_strictly_greater_than": config.strict_min_mean,
                "p05_at_least": config.strict_min_p05,
                "p01_at_least": config.strict_min_p01,
                "raw_min_reference_diagnostic_only": config.strict_min_value,
            }
        ):
            raise ValueError("Attempt12 pooled2048 metadata changed")
        for position, pooled_row in enumerate(pooled["actions"]):
            x_row = stress["actions"][position]
            c_row = confirmation["actions"][position]
            expected_raw = [
                *x_row["raw_paired_deltas_vs_baseline"],
                *c_row["raw_paired_deltas_vs_baseline"],
            ]
            if (
                pooled_row.get("raw_paired_deltas_vs_baseline") != expected_raw
                or pooled_row.get("source_raw_paired_deltas_sha256")
                != {
                    "stress_x1024": x_row["raw_paired_deltas_sha256"],
                    "confirmation_c1024": c_row["raw_paired_deltas_sha256"],
                }
            ):
                raise ValueError("Attempt12 pooled X/C raw concatenation changed")
        pooled_checks = [
            _checks(
                row["paired_delta_vs_baseline"],
                min_mean=config.strict_min_mean,
                min_p05=config.strict_min_p05,
                min_p01=config.strict_min_p01,
            )
            for row in pooled["actions"][:-1]
        ]
        pooled_min_diagnostics = [
            _diagnostic_min(
                row["paired_delta_vs_baseline"], reference=config.strict_min_value
            )
            for row in pooled["actions"][:-1]
        ]
        eligible = [
            position
            for position, checks in enumerate(pooled_checks)
            if all(checks.values())
        ]
        eligible_keys = [pooled_scope_keys[position] for position in eligible]
        risk_by_position = {
            position: _observed_tail_risk(
                pooled["actions"][position]["paired_delta_vs_baseline"]
            )
            for position in eligible
        }
        selected_position = (
            min(
                eligible,
                key=lambda position: (
                    risk_by_position[position][0],
                    -float(
                        pooled["actions"][position]["paired_delta_vs_baseline"][
                            "mean"
                        ]
                    ),
                    position,
                    action_key(legal_by_token[pooled_scope_keys[position]]).sort_key(),
                ),
            )
            if eligible
            else None
        )
        expected_risk_payload = {
            str(position): {
                "score": risk_by_position[position][0],
                "components": risk_by_position[position][1],
            }
            for position in eligible
        }
        if (
            pooled.get("checks_by_position") != pooled_checks
            or pooled.get("raw_min_diagnostics_by_position")
            != pooled_min_diagnostics
            or pooled.get("eligible_positions") != eligible
            or pooled.get("eligible_action_keys") != eligible_keys
            or pooled.get("normalized_tail_risk_by_position")
            != expected_risk_payload
            or pooled.get("selected_position") != selected_position
            or pooled.get("selected_action_key")
            != (pooled_scope_keys[selected_position] if selected_position is not None else None)
        ):
            raise ValueError("Attempt12 pooled2048 eligibility/risk lock changed")
        if selected_position is not None:
            selected = pooled_scope_keys[selected_position]
    else:
        _closed_phase_contract(stress, configured_sample_count=config.stress_samples)
        _closed_phase_contract(
            confirmation, configured_sample_count=config.confirmation_samples
        )
        _closed_phase_contract(
            pooled, configured_sample_count=ATTEMPT12_POOLED_SAMPLES
        )

    if decision.get("final_selected_action_key") != selected:
        raise ValueError("Attempt12 locked final action changed")
    fired = selected != baseline_action_key
    expected_reason = None
    if not veto_expected:
        expected_reason = "no_v256_candidate_passed"
    elif selected == baseline_action_key:
        expected_reason = "no_pooled_x1024_c1024_candidate_passed"
    if (
        decision.get("override_fired") is not fired
        or decision.get("exact_baseline_fallback") is fired
        or decision.get("fallback_reason") != expected_reason
        or decision.get("candidate_fallback_after_V_or_pooled_allowed") is not True
        or decision.get("frozen_before_evaluation_namespace_open") is not True
    ):
        raise ValueError("Attempt12 decision fallback contract changed")
    if evaluation.get("opened"):
        _phase_raw_contract(
            evaluation,
            sample_count=config.evaluation_samples,
            baseline_token=baseline_action_key,
            legal_by_token=legal_by_token,
            expected_action_keys=[selected, baseline_action_key],
        )
        _open_phase_metadata(evaluation, sample_count=config.evaluation_samples)
    else:
        _closed_phase_contract(
            evaluation, configured_sample_count=config.evaluation_samples
        )
    if (
        evaluation.get("opened") is not fired
        or evaluation.get("locked_final_action_key") != selected
        or evaluation.get("diagnostics_only") is not True
        or evaluation.get("can_rerank_or_gate") is not False
        or evaluation.get("decision_frozen_before_namespace_open") is not True
    ):
        raise ValueError("Attempt12 E512 diagnostic lock changed")

    expected_search_config = {
        "all_legal_candidates": config.all_legal_candidates,
        "candidate_nonbaseline_count": len(expected_top_indices),
        "baseline_added_exactly_once": True,
        "rerank_samples": config.rerank_samples,
        "rerank_head_max": config.rerank_head_max,
        "shortlist_max": config.shortlist_max,
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
    }
    if payload.get("search_config") != expected_search_config:
        raise ValueError("Attempt12 variable search configuration changed")

    key_sets = [set(values) for values in payload.get("rng_key_digests", {}).values()]
    for left_index, left in enumerate(key_sets):
        for right in key_sets[left_index + 1 :]:
            if not left.isdisjoint(right):
                raise ValueError("Attempt12 phase RNG namespaces overlap")
    provenance = payload.get("seed_domain_provenance")
    if not isinstance(provenance, Mapping) or provenance.get("domain_order") != list(
        ATTEMPT12_RNG_DOMAINS
    ):
        raise ValueError("Attempt12 seven-domain seed provenance changed")
    expected_provenance = {
        "hand_external": config.hand_seed,
        "rerank_r128": config.rerank_seed,
        "veto_v256": config.veto_seed,
        "stress_x1024": config.stress_seed,
        "confirmation_c1024": config.confirmation_seed,
        "evaluation_e512": config.evaluation_seed,
        "child_policy": config.child_policy_seed,
    }
    if any(provenance.get(name) != value for name, value in expected_provenance.items()):
        raise ValueError("Attempt12 seed provenance changed")
    return {
        "schema": "hu_m43_attempt12_teacher_validation_v1",
        "selected_action_key": selected,
        "override_fired": fired,
        "exact_baseline_fallback": not fired,
        "opened_phases": list(payload.get("rng_key_digests", {})),
    }


__all__ = [
    "ATTEMPT12_COARSE_MIN_MEAN",
    "ATTEMPT12_COARSE_MIN_P01",
    "ATTEMPT12_COARSE_MIN_P05",
    "ATTEMPT12_COARSE_MIN_VALUE",
    "ATTEMPT12_CONFIRMATION_SAMPLES",
    "ATTEMPT12_EVALUATION_SAMPLES",
    "ATTEMPT12_FROZEN_MODEL_ID",
    "ATTEMPT12_FROZEN_MODEL_SHA256",
    "ATTEMPT12_ALL_LEGAL_CANDIDATES",
    "ATTEMPT12_RERANK_SAMPLES",
    "ATTEMPT12_POOLED_SAMPLES",
    "ATTEMPT12_RNG_DOMAINS",
    "ATTEMPT12_SOLVER_ID",
    "ATTEMPT12_STRESS_SAMPLES",
    "ATTEMPT12_STRICT_MIN_MEAN",
    "ATTEMPT12_STRICT_MIN_P01",
    "ATTEMPT12_STRICT_MIN_P05",
    "ATTEMPT12_STRICT_MIN_VALUE",
    "ATTEMPT12_TEACHER_SCHEMA",
    "ATTEMPT12_RERANK_HEAD_MAX",
    "ATTEMPT12_SHORTLIST_MAX",
    "ATTEMPT12_VETO_SAMPLES",
    "Attempt12RankScores",
    "Attempt12Ranker",
    "Attempt12TeacherConfig",
    "FrozenAttempt12LambdaRanker",
    "evaluate_attempt12_t1_second",
    "validate_attempt12_teacher_output",
]
