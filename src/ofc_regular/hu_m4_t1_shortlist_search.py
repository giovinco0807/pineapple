"""Fail-closed T1-second shortlist search backed by the M3 Rust engine.

The learned model supplies exactly four canonical ``ActionKey`` tokens.  This
module never calls the model and never sees an opponent policy id or private
discard.  It adds the baseline action, freezes that semantic action set, and
scores it on disjoint common-random candidate/evaluation particle batches.

T2 remains the explicitly injected fixed policy.  T3 and T4 child decisions
use the existing Rust batch entrypoint.  T4 decisions encountered inside this
T1 counterfactual search use a separately declared one-sample planning budget;
setting both direct and nested budgets to zero is the expensive exact-all
reference mode.  This module does not alter the real-game T4 exact solver or
its profile configuration.  Returned search values are runtime features only;
this module has no override threshold or fire gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .action_key import (
    ACTION_KEY_SCHEMA,
    action_key,
    canonical_descending_indices,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from .action_space import generate_turn_actions
from .hu_belief import sample_hidden_card_particles
from .hu_infoset import ActorObservation
from .hu_late_street_teacher import T4SearchConfig
from .hu_m4_t1_teacher import (
    _ActionScores,
    _ChildSelector,
    _score_actions,
    _score_actions_batched,
)
from .hu_m4_teacher_contract import (
    T1_SECOND_LIVE_SCHEDULE,
    require_disjoint_root_rng_keys,
    require_t1_second_root,
)
from .hu_turn3_joint_exact_teacher import JointExactConfig


M4_T1_SHORTLIST_SEARCH_SCHEMA = "hu_m4_t1_second_shortlist_search_v1"
M4_T1_SHORTLIST_SOLVER_ID = "python_root_m3_rust_batch_t3_t4_shortlist_v1"
M4_T1_SHORTLIST_TOP_K = 4
M4_T1_SHORTLIST_MC_BUDGETS = (4, 8)


@dataclass(frozen=True)
class M4T1ShortlistSearchConfig:
    """Frozen small-budget runtime search configuration.

    Candidate and evaluation budgets are deliberately limited to the two
    preregistered smoke variants.  Scalar execution exists only for parity
    checks; runtime callers should keep ``batch_child_selectors=True``.
    """

    candidate_samples: int = 4
    evaluation_samples: int = 8
    candidate_seed: int = 2026071501
    evaluation_seed: int = 2026071502
    child_policy_seed: int = 2026071503
    run_id: str = "hu-m4-t1-second-shortlist-search"
    t2_policy_id: str = "explicit-fixed-t2-policy"
    top_k: int = M4_T1_SHORTLIST_TOP_K
    t3_candidate_samples: int = 1
    t3_evaluation_samples: int = 1
    t3_downstream_t3_samples: int = 1
    t3_downstream_t4_samples: int = 1
    t4_candidate_samples: int = 1
    t4_evaluation_samples: int = 1
    batch_child_selectors: bool = True

    def __post_init__(self) -> None:
        if self.candidate_samples not in M4_T1_SHORTLIST_MC_BUDGETS:
            raise ValueError("candidate_samples must be one of 4 or 8")
        if self.evaluation_samples not in M4_T1_SHORTLIST_MC_BUDGETS:
            raise ValueError("evaluation_samples must be one of 4 or 8")
        if self.top_k != M4_T1_SHORTLIST_TOP_K:
            raise ValueError("Attempt05 shortlist top_k is frozen at 4")
        for name in (
            "t3_candidate_samples",
            "t3_evaluation_samples",
            "t3_downstream_t3_samples",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.t3_downstream_t4_samples not in (0, 1):
            raise ValueError(
                "t3_downstream_t4_samples must be 0 (exact reference) or 1"
            )
        if self.t4_candidate_samples not in (0, 1) or self.t4_evaluation_samples not in (0, 1):
            raise ValueError("T1-search direct T4 samples must be 0 or 1")
        if self.t4_candidate_samples != self.t4_evaluation_samples:
            raise ValueError("T1-search direct T4 candidate/evaluation budgets must match")
        seeds = (
            self.candidate_seed,
            self.evaluation_seed,
            self.child_policy_seed,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in seeds):
            raise TypeError("candidate/evaluation/child seeds must be integers")
        if len(set(seeds)) != len(seeds):
            raise ValueError("candidate/evaluation/child seed namespaces must be distinct")
        if not self.run_id or not self.t2_policy_id:
            raise ValueError("run_id and t2_policy_id must not be empty")
        if not isinstance(self.batch_child_selectors, bool):
            raise TypeError("batch_child_selectors must be a bool")


class _RustShortlistChildSelector(_ChildSelector):
    """M4 child selector with separately declared hypothetical T4 planning."""

    config: M4T1ShortlistSearchConfig

    def _t3_config(self, observation: ActorObservation) -> JointExactConfig:
        return JointExactConfig(
            candidate_samples=self.config.t3_candidate_samples,
            evaluation_samples=self.config.t3_evaluation_samples,
            downstream_t3_samples=self.config.t3_downstream_t3_samples,
            downstream_t4_samples=self.config.t3_downstream_t4_samples,
            seed=self.config.child_policy_seed,
            candidate_seed=self.config.child_policy_seed + 101,
            evaluation_seed=self.config.child_policy_seed + 102,
            run_id=f"m4-shortlist-child-t3:{self._t3_policy_id()}",
            seat=observation.seat,
            to_act_order=observation.to_act_order,
        )

    def _t4_config(self) -> T4SearchConfig:
        return T4SearchConfig(
            candidate_samples=self.config.t4_candidate_samples,
            evaluation_samples=self.config.t4_evaluation_samples,
            seed=self.config.child_policy_seed,
            candidate_seed=self.config.child_policy_seed + 201,
            evaluation_seed=self.config.child_policy_seed + 202,
            run_id=f"m4-shortlist-child-t4:{self._t4_policy_id()}",
        )

    def _t3_policy_id(self) -> str:
        return (
            "m3-t3-shortlist"
            f":c={self.config.t3_candidate_samples}"
            f":e={self.config.t3_evaluation_samples}"
            f":d3={self.config.t3_downstream_t3_samples}"
            f":d4={self.config.t3_downstream_t4_samples}"
            f":seed={self.config.child_policy_seed}"
        )

    def _t4_policy_id(self) -> str:
        return (
            "m3-t4-shortlist"
            f":c={self.config.t4_candidate_samples}"
            f":e={self.config.t4_evaluation_samples}"
            f":seed={self.config.child_policy_seed}"
        )


def evaluate_t1_second_shortlist_search(
    observation: ActorObservation,
    *,
    t2_policies: Mapping[str, object],
    baseline_action_key: str,
    learned_shortlist_action_keys: Sequence[str],
    config: M4T1ShortlistSearchConfig | None = None,
    library: Any | None = None,
) -> dict[str, Any]:
    """Score a frozen learned top-4 plus baseline without making a fire decision."""

    require_t1_second_root(observation)
    config = config or M4T1ShortlistSearchConfig()
    actions = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    if not actions:
        raise ValueError("T1-second observation has no legal actions")

    legal_tokens = tuple(action_key(value).to_token() for value in actions)
    token_to_index = {token: index for index, token in enumerate(legal_tokens)}
    if len(token_to_index) != len(actions):
        raise ValueError("legal T1 ActionKey mapping is not one-to-one")
    if not isinstance(baseline_action_key, str) or baseline_action_key not in token_to_index:
        raise ValueError("baseline ActionKey is not legal at the T1 root")
    shortlist = tuple(learned_shortlist_action_keys)
    if len(shortlist) != config.top_k:
        raise ValueError("learned shortlist must contain exactly four ActionKeys")
    if any(not isinstance(token, str) or not token for token in shortlist):
        raise TypeError("learned shortlist entries must be non-empty ActionKey strings")
    if len(set(shortlist)) != len(shortlist):
        raise ValueError("learned shortlist ActionKeys must be unique")
    unknown = [token for token in shortlist if token not in token_to_index]
    if unknown:
        raise ValueError(f"learned shortlist contains illegal ActionKey: {unknown[0]!r}")

    # The learned order is preserved for audit, but search tie-breaking is by
    # canonical ActionKey.  The baseline is appended exactly once if needed.
    fixed_tokens = (*shortlist, *(() if baseline_action_key in shortlist else (baseline_action_key,)))
    fixed_indices = tuple(token_to_index[token] for token in fixed_tokens)
    fixed_actions = tuple(actions[index] for index in fixed_indices)
    baseline_fixed_index = fixed_tokens.index(baseline_action_key)

    candidate_run_id = f"{config.run_id}:candidate_selection"
    evaluation_run_id = f"{config.run_id}:locked_evaluation"
    candidate = sample_hidden_card_particles(
        observation,
        base_seed=config.candidate_seed,
        run_id=candidate_run_id,
        sample_count=config.candidate_samples,
    )
    evaluation = sample_hidden_card_particles(
        observation,
        base_seed=config.evaluation_seed,
        run_id=evaluation_run_id,
        sample_count=config.evaluation_samples,
    )
    candidate.validate_against(observation)
    evaluation.validate_against(observation)
    if (
        candidate.base_seed != config.candidate_seed
        or candidate.run_id != candidate_run_id
        or candidate.start_index != 0
        or len(candidate.particles) != config.candidate_samples
    ):
        raise ValueError("candidate belief provenance disagrees with shortlist config")
    if (
        evaluation.base_seed != config.evaluation_seed
        or evaluation.run_id != evaluation_run_id
        or evaluation.start_index != 0
        or len(evaluation.particles) != config.evaluation_samples
    ):
        raise ValueError("evaluation belief provenance disagrees with shortlist config")
    candidate_keys = tuple(row.rng_key_digest for row in candidate.particles)
    evaluation_keys = tuple(row.rng_key_digest for row in evaluation.particles)
    require_disjoint_root_rng_keys(candidate_keys, evaluation_keys)

    selector = _RustShortlistChildSelector(
        t2_policies=t2_policies,
        config=config,  # type: ignore[arg-type]
        library=library,
    )
    scorer = _score_actions_batched if config.batch_child_selectors else _score_actions
    candidate_scores = scorer(observation, fixed_actions, candidate, selector)
    if len(candidate_scores) != len(fixed_actions):
        raise ValueError("candidate shortlist scorer returned the wrong action count")
    candidate_means = tuple(row.mean for row in candidate_scores)
    ranking = canonical_descending_indices(candidate_means, fixed_actions)
    proposed_fixed_index = ranking[0]
    runner_up_fixed_index = ranking[1] if len(ranking) > 1 else proposed_fixed_index

    # Freeze the semantic proposal before the independent evaluation batch is
    # opened.  Evaluation values are features only and cannot change ranking.
    evaluation_scores = scorer(observation, fixed_actions, evaluation, selector)
    if len(evaluation_scores) != len(fixed_actions):
        raise ValueError("evaluation shortlist scorer returned the wrong action count")
    evaluation_means = tuple(row.mean for row in evaluation_scores)
    baseline_candidate = candidate_scores[baseline_fixed_index]
    baseline_evaluation = evaluation_scores[baseline_fixed_index]

    learned_rank = {token: rank for rank, token in enumerate(shortlist)}
    rows = []
    for search_rank, fixed_index in enumerate(ranking):
        original_index = fixed_indices[fixed_index]
        token = fixed_tokens[fixed_index]
        action = fixed_actions[fixed_index]
        candidate_delta = _paired_mean_se(
            candidate_scores[fixed_index], baseline_candidate
        )
        evaluation_delta = _paired_mean_se(
            evaluation_scores[fixed_index], baseline_evaluation
        )
        rows.append(
            {
                "search_rank": search_rank,
                "fixed_search_index": fixed_index,
                "original_index": original_index,
                "action_key": token,
                "placements": [list(value) for value in action.placements],
                "discards": list(action.discards),
                "learned_rank": learned_rank.get(token),
                "is_baseline": token == baseline_action_key,
                "candidate_mean": candidate_means[fixed_index],
                "candidate_standard_error": candidate_scores[
                    fixed_index
                ].standard_error,
                "candidate_margin_vs_baseline": candidate_delta["mean"],
                "candidate_margin_standard_error": candidate_delta[
                    "standard_error"
                ],
                "evaluation_mean": evaluation_means[fixed_index],
                "evaluation_standard_error": evaluation_scores[
                    fixed_index
                ].standard_error,
                "evaluation_margin_vs_baseline": evaluation_delta["mean"],
                "evaluation_margin_standard_error": evaluation_delta[
                    "standard_error"
                ],
            }
        )

    proposed_token = fixed_tokens[proposed_fixed_index]
    return {
        "status": "ok",
        "schema": M4_T1_SHORTLIST_SEARCH_SCHEMA,
        "solver_id": M4_T1_SHORTLIST_SOLVER_ID,
        "observation_fingerprint": observation.fingerprint(),
        "seat": observation.seat,
        "street": observation.street,
        "to_act_order": observation.to_act_order,
        "action_key_schema": ACTION_KEY_SCHEMA,
        "legal_action_count": len(actions),
        "legal_action_set_digest": legal_action_set_digest(actions),
        "legal_action_order_digest": ordered_action_mapping_digest(actions),
        "ordered_legal_actions": [
            {
                "original_index": index,
                "action_key": token,
                "placements": [list(value) for value in action.placements],
                "discards": list(action.discards),
            }
            for index, (token, action) in enumerate(zip(legal_tokens, actions, strict=True))
        ],
        "baseline_original_index": token_to_index[baseline_action_key],
        "baseline_action_key": baseline_action_key,
        "learned_shortlist_action_keys": list(shortlist),
        "fixed_search_action_keys": list(fixed_tokens),
        "fixed_search_action_count": len(fixed_actions),
        "proposed_original_index": fixed_indices[proposed_fixed_index],
        "proposed_action_key": proposed_token,
        "proposal_is_baseline": proposed_token == baseline_action_key,
        "candidate_margin_vs_baseline": (
            candidate_means[proposed_fixed_index]
            - candidate_means[baseline_fixed_index]
        ),
        "candidate_margin_vs_runner_up": (
            candidate_means[proposed_fixed_index]
            - candidate_means[runner_up_fixed_index]
        ),
        "evaluation_margin_vs_baseline": (
            evaluation_means[proposed_fixed_index]
            - evaluation_means[baseline_fixed_index]
        ),
        "candidate_belief_digest": candidate.digest(),
        "evaluation_belief_digest": evaluation.digest(),
        "candidate_rng_key_digests": list(candidate_keys),
        "evaluation_rng_key_digests": list(evaluation_keys),
        "common_random_futures_across_fixed_actions": True,
        "candidate_evaluation_rng_disjoint": True,
        "fixed_candidates_before_sampling": True,
        "proposal_locked_before_evaluation": True,
        "seed_namespaces": {
            "candidate": {
                "seed": config.candidate_seed,
                "run_id": candidate_run_id,
            },
            "evaluation": {
                "seed": config.evaluation_seed,
                "run_id": evaluation_run_id,
            },
            "child_policy": {
                "seed": config.child_policy_seed,
                "policy_id": config.t2_policy_id,
            },
        },
        "search_config": {
            "top_k": config.top_k,
            "candidate_samples": config.candidate_samples,
            "evaluation_samples": config.evaluation_samples,
            "batch_child_selectors": config.batch_child_selectors,
            "t3_candidate_samples": config.t3_candidate_samples,
            "t3_evaluation_samples": config.t3_evaluation_samples,
            "t3_downstream_t3_samples": config.t3_downstream_t3_samples,
            "t3_downstream_t4_samples": config.t3_downstream_t4_samples,
            "t4_candidate_samples": config.t4_candidate_samples,
            "t4_evaluation_samples": config.t4_evaluation_samples,
        },
        "continuation_policy": {
            "t2_policy_id": config.t2_policy_id,
            "t3_selector": "m3_rust_evaluate_t3",
            "t4_selector": "m3_rust_evaluate_t4",
            "child_selector_execution": (
                "batched_infoset_locked_v1"
                if config.batch_child_selectors
                else "scalar_infoset_locked_v1"
            ),
            "outer_live_t4_exact_solver_unchanged": True,
            "t1_search_direct_t4_exact": config.t4_candidate_samples == 0,
            "t1_search_direct_t4_mode": (
                "exact_uniform_marginal"
                if config.t4_candidate_samples == 0
                else "counter_mc_1"
            ),
            "nested_t4_exact": config.t3_downstream_t4_samples == 0,
            "nested_t4_mode": (
                "exact_uniform_marginal"
                if config.t3_downstream_t4_samples == 0
                else "counter_mc_1"
            ),
            "outer_future_index_in_child_seed": False,
            "outer_action_index_in_child_seed": False,
        },
        "live_schedule": [
            {
                "seat": step.seat,
                "street": step.street,
                "draw_offset": step.draw_offset,
            }
            for step in T1_SECOND_LIVE_SCHEDULE
        ],
        "child_information_set_count": len(selector.cache),
        "actions": rows,
        "search_value_status": "runtime_feature_not_teacher_ev_or_runtime_gate",
        "runtime_gate_applied": False,
        "opponent_policy_identity_used": False,
        "opponent_private_discard_input_used": False,
    }


def _paired_mean_se(
    candidate: _ActionScores, baseline: _ActionScores
) -> dict[str, float | int]:
    if len(candidate.values) != len(baseline.values) or not candidate.values:
        raise ValueError("paired action scores require equal non-empty vectors")
    values = np.asarray(candidate.values, dtype=np.float64) - np.asarray(
        baseline.values, dtype=np.float64
    )
    if not np.isfinite(values).all():
        raise ValueError("paired search margins must be finite")
    count = int(values.size)
    standard_error = (
        float(np.std(values, ddof=1) / math.sqrt(count)) if count > 1 else 0.0
    )
    return {
        "count": count,
        "mean": float(np.mean(values)),
        "standard_error": standard_error,
    }


__all__ = [
    "M4_T1_SHORTLIST_MC_BUDGETS",
    "M4_T1_SHORTLIST_SEARCH_SCHEMA",
    "M4_T1_SHORTLIST_SOLVER_ID",
    "M4_T1_SHORTLIST_TOP_K",
    "M4T1ShortlistSearchConfig",
    "evaluate_t1_second_shortlist_search",
]
