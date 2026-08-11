"""Attempt07 development-only T1-second split search/confirm evaluator.

The frozen Attempt06 LambdaRank artifact is used only to form a top-eight
non-baseline proposal set.  Four disjoint common-random namespaces then have
strictly separated jobs:

* S8 screens all proposals and the explicit baseline into three proposals;
* R64 reranks that shortlist and the baseline (with R32 as a true prefix);
* V128 applies a paired safety veto only to locked rerank winners and the
  explicit baseline (with V64 as a true prefix); and
* A128 assesses every top-eight proposal and the baseline for diagnostics.

The assessment namespace is opened only after all four arm decisions have
been frozen and has no path back into selection.  A failed veto falls back to
the explicit baseline; it never promotes a second-best action.  Inputs are
limited to :class:`ActorObservation`, so opponent-private discards cannot enter
the ranker, particle prior, or continuation policy.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .action_key import (
    ACTION_KEY_SCHEMA,
    action_key,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from .action_space import Action, generate_turn_actions
from .hu_belief import sample_hidden_card_particles
from .hu_infoset import ActorObservation
from .hu_m43_attempt06_teacher import (
    ATTEMPT06_FROZEN_MODEL_SHA256,
    ATTEMPT06_T2_POLICY_ID,
    Attempt06Ranker,
    FrozenAttempt06LambdaRanker,
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


ATTEMPT07_TEACHER_SCHEMA = "hu_m43_attempt07_t1_second_s8_r64_v128_a128_v1"
ATTEMPT07_SOLVER_ID = "attempt06_lambda_top8_s8_top3_r64_v128_a128_m3_mc1_v1"
ATTEMPT07_TOP_K = 8
ATTEMPT07_SCREEN_SAMPLES = 8
ATTEMPT07_SHORTLIST_K = 3
ATTEMPT07_RERANK_SAMPLES = 64
ATTEMPT07_VETO_SAMPLES = 128
ATTEMPT07_ASSESSMENT_SAMPLES = 128
ATTEMPT07_VETO_MIN_MEAN = 0.0
ATTEMPT07_VETO_MIN_P05 = -25.0
ATTEMPT07_VETO_MIN_P01 = -40.0
ATTEMPT07_VETO_MIN_VALUE = -50.0
ATTEMPT07_ARM_SPECS: tuple[tuple[str, int, int], ...] = (
    ("R32_V64", 32, 64),
    ("R64_V64", 64, 64),
    ("R32_V128", 32, 128),
    ("R64_V128", 64, 128),
)


@dataclass(frozen=True)
class Attempt07TeacherConfig:
    """Hard-locked development contract for one Attempt07 root."""

    frozen_model_sha256: str
    screen_seed: int
    rerank_seed: int
    veto_seed: int
    assessment_seed: int
    child_policy_seed: int
    run_id: str
    candidate_top_k: int = ATTEMPT07_TOP_K
    screen_samples: int = ATTEMPT07_SCREEN_SAMPLES
    shortlist_k: int = ATTEMPT07_SHORTLIST_K
    rerank_samples: int = ATTEMPT07_RERANK_SAMPLES
    veto_samples: int = ATTEMPT07_VETO_SAMPLES
    assessment_samples: int = ATTEMPT07_ASSESSMENT_SAMPLES
    veto_min_mean: float = ATTEMPT07_VETO_MIN_MEAN
    veto_min_p05: float = ATTEMPT07_VETO_MIN_P05
    veto_min_p01: float = ATTEMPT07_VETO_MIN_P01
    veto_min_value: float = ATTEMPT07_VETO_MIN_VALUE
    t2_policy_id: str = ATTEMPT06_T2_POLICY_ID
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
        if self.frozen_model_sha256 != ATTEMPT06_FROZEN_MODEL_SHA256:
            raise ValueError("Attempt07 requires the frozen Attempt06 Lambda artifact")
        fixed_ints = {
            "candidate_top_k": ATTEMPT07_TOP_K,
            "screen_samples": ATTEMPT07_SCREEN_SAMPLES,
            "shortlist_k": ATTEMPT07_SHORTLIST_K,
            "rerank_samples": ATTEMPT07_RERANK_SAMPLES,
            "veto_samples": ATTEMPT07_VETO_SAMPLES,
            "assessment_samples": ATTEMPT07_ASSESSMENT_SAMPLES,
            "t3_candidate_samples": 1,
            "t3_evaluation_samples": 1,
            "t3_downstream_samples": 1,
            "t4_candidate_samples": 1,
            "t4_evaluation_samples": 1,
        }
        for name, expected in fixed_ints.items():
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value != expected
            ):
                raise ValueError(f"Attempt07 {name} is fixed at {expected}")
        fixed_floats = {
            "veto_min_mean": ATTEMPT07_VETO_MIN_MEAN,
            "veto_min_p05": ATTEMPT07_VETO_MIN_P05,
            "veto_min_p01": ATTEMPT07_VETO_MIN_P01,
            "veto_min_value": ATTEMPT07_VETO_MIN_VALUE,
        }
        for name, expected in fixed_floats.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or float(value) != expected:
                raise ValueError(f"Attempt07 {name} is fixed at {expected}")
        seed_names = (
            "screen_seed",
            "rerank_seed",
            "veto_seed",
            "assessment_seed",
            "child_policy_seed",
        )
        for name in seed_names:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if len({getattr(self, name) for name in seed_names}) != len(seed_names):
            raise ValueError("Attempt07 S/R/V/A/child seeds must all be distinct")
        if self.t2_policy_id != ATTEMPT06_T2_POLICY_ID:
            raise ValueError("Attempt07 T2 policy is fixed at stage9f_p2")
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("Attempt07 run_id must not be empty")
        if not isinstance(self.batch_child_selectors, bool):
            raise TypeError("batch_child_selectors must be a bool")

    def m4_config(self) -> M4T1TeacherConfig:
        """Build the unchanged M4 continuation selector configuration."""

        return M4T1TeacherConfig(
            candidate_samples=self.screen_samples,
            evaluation_samples=self.veto_samples,
            candidate_seed=self.screen_seed,
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


def _resolve_baseline(actions: Sequence[Action], token: str) -> tuple[int, str]:
    desired = token
    matching = [
        index
        for index, action in enumerate(actions)
        if action_key(action).to_token() == desired
    ]
    if len(matching) != 1:
        raise ValueError("Attempt07 explicit baseline ActionKey is not uniquely legal")
    return matching[0], desired


def _raw_digest(values: Sequence[float]) -> str:
    return hashlib.sha256(
        json.dumps(
            [float(value) for value in values],
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _score_prefix(score: _ActionScores, count: int) -> _ActionScores:
    if count <= 0 or count > len(score.values):
        raise ValueError("Attempt07 score prefix is outside the sampled vector")
    values = tuple(float(value) for value in score.values[:count])
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Attempt07 scorer emitted a non-finite value")
    return _ActionScores(values)


def _validate_scores(
    scores: Sequence[_ActionScores], *, action_count: int, sample_count: int, phase: str
) -> tuple[_ActionScores, ...]:
    if len(scores) != action_count:
        raise ValueError(f"Attempt07 {phase} scorer returned the wrong action count")
    checked = tuple(scores)
    for score in checked:
        if len(score.values) != sample_count:
            raise ValueError(
                f"Attempt07 {phase} scorer returned the wrong sample count"
            )
        if not all(math.isfinite(float(value)) for value in score.values):
            raise ValueError(f"Attempt07 {phase} scorer emitted a non-finite value")
    return checked


def _winner_position(
    scores: Sequence[_ActionScores],
    actions: Sequence[Action],
    *,
    baseline_position: int,
) -> int:
    """Choose the largest mean; the baseline wins an exact best tie."""

    if len(scores) != len(actions) or not 0 <= baseline_position < len(actions):
        raise ValueError("Attempt07 winner inputs disagree")
    means = tuple(score.mean for score in scores)
    best = max(means)
    tied = [index for index, value in enumerate(means) if value == best]
    if baseline_position in tied:
        return baseline_position
    return min(tied, key=lambda index: action_key(actions[index]).sort_key())


def _paired_row(
    candidate: _ActionScores,
    baseline: _ActionScores,
) -> dict[str, Any]:
    deltas = [
        float(candidate_value - baseline_value)
        for candidate_value, baseline_value in zip(
            candidate.values, baseline.values, strict=True
        )
    ]
    if not deltas or not all(math.isfinite(value) for value in deltas):
        raise ValueError("Attempt07 paired deltas must be finite and non-empty")
    return {
        "paired_delta_vs_baseline": _paired_delta_summary(candidate, baseline),
        "raw_paired_deltas_vs_baseline": deltas,
        "raw_paired_deltas_sha256": _raw_digest(deltas),
        "action_values_sha256": _raw_digest(candidate.values),
    }


def _phase_actions(
    actions: Sequence[Action],
    scores: Sequence[_ActionScores],
    *,
    baseline_position: int,
    prefix: int,
) -> list[dict[str, Any]]:
    prefixed = tuple(_score_prefix(score, prefix) for score in scores)
    baseline = prefixed[baseline_position]
    return [
        {
            "phase_position": position,
            "action_key": action_key(action).to_token(),
            "mean": score.mean,
            "standard_error": score.standard_error,
            "is_explicit_baseline": position == baseline_position,
            **_paired_row(score, baseline),
        }
        for position, (action, score) in enumerate(zip(actions, prefixed, strict=True))
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
        "mean_gt_0": float(summary["mean"]) > ATTEMPT07_VETO_MIN_MEAN,
        "p05_ge_neg25": float(summary["p05"]) >= ATTEMPT07_VETO_MIN_P05,
        "p01_ge_neg40": float(summary["p01"]) >= ATTEMPT07_VETO_MIN_P01,
        "min_ge_neg50": float(summary["min"]) >= ATTEMPT07_VETO_MIN_VALUE,
    }


def evaluate_attempt07_t1_second(
    observation: ActorObservation,
    *,
    baseline_action_key: str,
    ranker: Attempt06Ranker,
    t2_policies: Mapping[str, object],
    config: Attempt07TeacherConfig,
    library: Any | None = None,
) -> dict[str, Any]:
    """Evaluate four frozen split-search arms without activating a policy."""

    require_t1_second_root(observation)
    if ranker.artifact_sha256 != config.frozen_model_sha256:
        raise ValueError("Attempt07 ranker artifact hash disagrees with config")
    _require_stage9f_p2_policies(t2_policies)

    legal_actions = tuple(
        generate_turn_actions(observation.hero_board, observation.dealt_cards)
    )
    if not legal_actions:
        raise ValueError("Attempt07 root has no legal actions")
    baseline_index, baseline_token = _resolve_baseline(
        legal_actions, baseline_action_key
    )

    # Freeze the top-eight proposal set before opening any particle namespace.
    rank_scores = ranker.score_actions(
        observation, legal_actions, baseline_index=baseline_index
    )
    rank_scores.validate(len(legal_actions))
    nonbaseline_indices = [
        index for index in range(len(legal_actions)) if index != baseline_index
    ]
    if len(nonbaseline_indices) < config.candidate_top_k:
        raise ValueError("Attempt07 root has fewer than eight nonbaseline actions")
    nonbaseline_indices.sort(
        key=lambda index: (
            -rank_scores.mean[index],
            action_key(legal_actions[index]).sort_key(),
        )
    )
    top_indices = tuple(nonbaseline_indices[: config.candidate_top_k])
    proposal_indices = (*top_indices, baseline_index)
    proposal_actions = tuple(legal_actions[index] for index in proposal_indices)
    if len({action_key(action) for action in proposal_actions}) != 9:
        raise AssertionError("Attempt07 proposal set must be top8 plus baseline once")
    proposal_baseline_position = len(proposal_actions) - 1

    selector = _ChildSelector(
        t2_policies=t2_policies,
        config=config.m4_config(),
        library=library,
    )
    scorer = _score_actions_batched if config.batch_child_selectors else _score_actions

    batches: dict[str, Any] = {}
    rng_keys: dict[str, tuple[str, ...]] = {}

    def sample(phase: str, *, seed: int, count: int) -> Any:
        batch = sample_hidden_card_particles(
            observation,
            base_seed=seed,
            run_id=f"{config.run_id}:{phase}",
            sample_count=count,
        )
        batch.validate_against(observation)
        keys = tuple(particle.rng_key_digest for particle in batch.particles)
        if len(keys) != count or len(set(keys)) != count:
            raise ValueError(
                f"Attempt07 {phase} particle count or RNG-key uniqueness changed"
            )
        for prior_keys in rng_keys.values():
            require_disjoint_root_rng_keys(prior_keys, keys)
        batches[phase] = batch
        rng_keys[phase] = keys
        return batch

    screen_batch = sample(
        "screen_s8", seed=config.screen_seed, count=config.screen_samples
    )
    screen_scores = _validate_scores(
        scorer(observation, proposal_actions, screen_batch, selector),
        action_count=len(proposal_actions),
        sample_count=config.screen_samples,
        phase="screen",
    )

    # The explicit baseline is scored at S8 but cannot consume a shortlist slot.
    screened_nonbaseline_positions = list(range(config.candidate_top_k))
    screened_nonbaseline_positions.sort(
        key=lambda position: (
            -screen_scores[position].mean,
            action_key(proposal_actions[position]).sort_key(),
        )
    )
    shortlist_proposal_positions = tuple(
        screened_nonbaseline_positions[: config.shortlist_k]
    )
    shortlist_indices = tuple(
        proposal_indices[position] for position in shortlist_proposal_positions
    )
    shortlist_actions = tuple(legal_actions[index] for index in shortlist_indices)
    rerank_actions = (*shortlist_actions, legal_actions[baseline_index])
    rerank_baseline_position = len(rerank_actions) - 1

    rerank_batch = sample(
        "rerank_r64", seed=config.rerank_seed, count=config.rerank_samples
    )
    rerank_scores = _validate_scores(
        scorer(observation, rerank_actions, rerank_batch, selector),
        action_count=len(rerank_actions),
        sample_count=config.rerank_samples,
        phase="rerank",
    )
    rerank_prefixes: dict[str, dict[str, Any]] = {}
    rerank_winners: dict[int, int] = {}
    for prefix in (32, 64):
        prefix_scores = tuple(_score_prefix(score, prefix) for score in rerank_scores)
        winner_position = _winner_position(
            prefix_scores,
            rerank_actions,
            baseline_position=rerank_baseline_position,
        )
        rerank_winners[prefix] = winner_position
        rerank_prefixes[f"R{prefix}"] = {
            "sample_count": prefix,
            "winner_position": winner_position,
            "winner_action_key": action_key(rerank_actions[winner_position]).to_token(),
            "winner_is_explicit_baseline": winner_position
            == rerank_baseline_position,
            "actions": _phase_actions(
                rerank_actions,
                rerank_scores,
                baseline_position=rerank_baseline_position,
                prefix=prefix,
            ),
        }

    # V may inspect only actions locked by R32/R64 plus the explicit baseline.
    # Preserve R-prefix order while removing duplicate nonbaseline winners.
    veto_rerank_positions = tuple(
        dict.fromkeys(
            position
            for prefix in (32, 64)
            if (position := rerank_winners[prefix]) != rerank_baseline_position
        )
    )
    veto_actions = tuple(
        [
            *(rerank_actions[position] for position in veto_rerank_positions),
            rerank_actions[rerank_baseline_position],
        ]
    )
    veto_baseline_position = len(veto_actions) - 1
    veto_position_by_key = {
        action_key(action).to_token(): position
        for position, action in enumerate(veto_actions)
    }
    if len(veto_position_by_key) != len(veto_actions):
        raise AssertionError("Attempt07 veto action keys must be unique")

    veto_batch = sample(
        "veto_v128", seed=config.veto_seed, count=config.veto_samples
    )
    veto_scores = _validate_scores(
        scorer(observation, veto_actions, veto_batch, selector),
        action_count=len(veto_actions),
        sample_count=config.veto_samples,
        phase="veto",
    )
    veto_prefixes: dict[str, dict[str, Any]] = {}
    for prefix in (64, 128):
        veto_prefixes[f"V{prefix}"] = {
            "sample_count": prefix,
            "actions": _phase_actions(
                veto_actions,
                veto_scores,
                baseline_position=veto_baseline_position,
                prefix=prefix,
            ),
        }

    # Freeze all decisions before the independent assessment namespace opens.
    arms: dict[str, dict[str, Any]] = {}
    for arm_name, rerank_prefix, veto_prefix in ATTEMPT07_ARM_SPECS:
        winner_position = rerank_winners[rerank_prefix]
        winner_token = action_key(rerank_actions[winner_position]).to_token()
        winner_is_baseline = winner_position == rerank_baseline_position
        try:
            veto_position = veto_position_by_key[winner_token]
        except KeyError as exc:  # pragma: no cover - structural invariant
            raise AssertionError(
                "Attempt07 rerank winner is absent from veto scope"
            ) from exc
        veto_row = veto_prefixes[f"V{veto_prefix}"]["actions"][veto_position]
        veto_summary = veto_row["paired_delta_vs_baseline"]
        checks = _veto_checks(veto_summary)
        veto_pass = not winner_is_baseline and all(checks.values())
        fired = veto_pass
        if winner_is_baseline:
            fallback_reason = "rerank_winner_is_explicit_baseline"
        elif not veto_pass:
            fallback_reason = "paired_safety_veto_failed"
        else:
            fallback_reason = None
        arms[arm_name] = {
            "rerank_prefix": f"R{rerank_prefix}",
            "veto_prefix": f"V{veto_prefix}",
            "rerank_winner_action_key": winner_token,
            "rerank_winner_position": winner_position,
            "rerank_winner_is_baseline": winner_is_baseline,
            "veto_action_position": veto_position,
            "veto_paired_delta_vs_baseline": veto_summary,
            "veto_raw_paired_deltas_vs_baseline": veto_row[
                "raw_paired_deltas_vs_baseline"
            ],
            "veto_raw_paired_deltas_sha256": veto_row[
                "raw_paired_deltas_sha256"
            ],
            "veto_checks": checks,
            "veto_pass": veto_pass,
            "selected_action_key": winner_token if fired else baseline_token,
            "override_fired": fired,
            "exact_baseline_fallback": not fired,
            "fallback_reason": fallback_reason,
            "second_best_promotion_allowed": False,
        }

    assessment_batch = sample(
        "assessment_a128",
        seed=config.assessment_seed,
        count=config.assessment_samples,
    )
    assessment_scores = _validate_scores(
        scorer(observation, proposal_actions, assessment_batch, selector),
        action_count=len(proposal_actions),
        sample_count=config.assessment_samples,
        phase="assessment",
    )
    assessment_means = tuple(score.mean for score in assessment_scores)
    assessment_best = min(
        [
            position
            for position, value in enumerate(assessment_means)
            if value == max(assessment_means)
        ],
        key=lambda position: action_key(proposal_actions[position]).sort_key(),
    )

    legal_rows = [
        {
            "original_legal_index": index,
            "action_key": action_key(action).to_token(),
            "model_rank_score": rank_scores.mean[index],
            "model_rank_disagreement": rank_scores.standard_deviation[index],
            "in_learned_top8": index in set(top_indices),
            "is_explicit_baseline": index == baseline_index,
        }
        for index, action in enumerate(legal_actions)
    ]
    return {
        "status": "ok",
        "schema": ATTEMPT07_TEACHER_SCHEMA,
        "solver_id": ATTEMPT07_SOLVER_ID,
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
            "purpose": "candidate_generation_only",
            "runtime_authorized": False,
            "profile_runtime_feature": False,
        },
        "legal_action_mapping": _mapping_payload(legal_actions),
        "legal_actions": legal_rows,
        "baseline_action_key": baseline_token,
        "baseline_original_legal_index": baseline_index,
        "learned_top8_original_legal_indices": list(top_indices),
        "learned_top8_action_keys": [
            action_key(legal_actions[index]).to_token() for index in top_indices
        ],
        "proposal_mapping": _mapping_payload(proposal_actions),
        "shortlist_proposal_positions": list(shortlist_proposal_positions),
        "shortlist_original_legal_indices": list(shortlist_indices),
        "shortlist_action_keys": [
            action_key(action).to_token() for action in shortlist_actions
        ],
        "shortlist_mapping": _mapping_payload(shortlist_actions),
        "screen": {
            **_mapping_payload(proposal_actions),
            "sample_count": ATTEMPT07_SCREEN_SAMPLES,
            "actions": _phase_actions(
                proposal_actions,
                screen_scores,
                baseline_position=proposal_baseline_position,
                prefix=ATTEMPT07_SCREEN_SAMPLES,
            ),
        },
        "rerank": {
            **_mapping_payload(rerank_actions),
            "sample_count": ATTEMPT07_RERANK_SAMPLES,
            "prefix_semantics": "R32_is_first_32_rows_of_same_R64_batch",
            "baseline_preferred_on_exact_best_tie": True,
            "prefixes": rerank_prefixes,
        },
        "veto": {
            **_mapping_payload(veto_actions),
            "sample_count": ATTEMPT07_VETO_SAMPLES,
            "scope": "unique_nonbaseline_R32_R64_winners_plus_explicit_baseline",
            "locked_nonbaseline_rerank_positions": list(veto_rerank_positions),
            "prefix_semantics": "V64_is_first_64_rows_of_same_V128_batch",
            "thresholds": {
                "mean_strictly_greater_than": ATTEMPT07_VETO_MIN_MEAN,
                "p05_at_least": ATTEMPT07_VETO_MIN_P05,
                "p01_at_least": ATTEMPT07_VETO_MIN_P01,
                "min_at_least": ATTEMPT07_VETO_MIN_VALUE,
            },
            "prefixes": veto_prefixes,
        },
        "arm_order": [name for name, _, _ in ATTEMPT07_ARM_SPECS],
        "arms": arms,
        "assessment": {
            **_mapping_payload(proposal_actions),
            "sample_count": ATTEMPT07_ASSESSMENT_SAMPLES,
            "scope": "all_top8_plus_explicit_baseline",
            "diagnostics_only": True,
            "can_rerank_or_gate": False,
            "decision_frozen_before_namespace_open": True,
            "sample_best_action_key": action_key(
                proposal_actions[assessment_best]
            ).to_token(),
            "actions": _phase_actions(
                proposal_actions,
                assessment_scores,
                baseline_position=proposal_baseline_position,
                prefix=ATTEMPT07_ASSESSMENT_SAMPLES,
            ),
        },
        "belief_digests": {
            phase: batches[phase].digest() for phase in batches
        },
        "rng_key_digests": {
            phase: list(keys) for phase, keys in rng_keys.items()
        },
        "sample_independence": "pairwise_disjoint_S8_R64_V128_A128_particle_rng_keys",
        "root_selection_lock": (
            "top8_fixed_before_S8_then_top3_fixed_before_R64_then_arms_frozen_"
            "after_V128_before_diagnostic_A128"
        ),
        "search_config": {
            "learned_nonbaseline_top_k": config.candidate_top_k,
            "baseline_added_exactly_once": True,
            "screen_samples": config.screen_samples,
            "shortlist_nonbaseline_k": config.shortlist_k,
            "rerank_samples": config.rerank_samples,
            "veto_samples": config.veto_samples,
            "assessment_samples": config.assessment_samples,
            "screen_seed": config.screen_seed,
            "rerank_seed": config.rerank_seed,
            "veto_seed": config.veto_seed,
            "assessment_seed": config.assessment_seed,
            "child_policy_seed": config.child_policy_seed,
            "run_id": config.run_id,
            "batch_child_selectors": config.batch_child_selectors,
            "model_and_screen_tie_break": "ActionKey",
            "rerank_tie_break": "explicit_baseline_then_ActionKey",
        },
        "continuation_policy": {
            "t2_policy_id": ATTEMPT06_T2_POLICY_ID,
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
        },
        "live_schedule": [
            {"seat": step.seat, "street": step.street, "draw_offset": step.draw_offset}
            for step in T1_SECOND_LIVE_SCHEDULE
        ],
        "child_information_set_count": len(selector.cache),
        "teacher_value_status": "diagnostic_not_match_EV",
        "runtime_gate_allowed": False,
        "profile_activation_allowed": False,
        "current_profile_resolved": False,
        "development_only": True,
    }


__all__ = [
    "ATTEMPT07_ARM_SPECS",
    "ATTEMPT07_ASSESSMENT_SAMPLES",
    "ATTEMPT07_RERANK_SAMPLES",
    "ATTEMPT07_SCREEN_SAMPLES",
    "ATTEMPT07_SHORTLIST_K",
    "ATTEMPT07_TEACHER_SCHEMA",
    "ATTEMPT07_TOP_K",
    "ATTEMPT07_VETO_SAMPLES",
    "Attempt07TeacherConfig",
    "FrozenAttempt06LambdaRanker",
    "evaluate_attempt07_t1_second",
]
