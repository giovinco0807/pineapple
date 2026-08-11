"""HU-aware Turn2 teacher data with explicit Turn3 continuation selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np

from .action_space import Action, generate_turn_actions
from .action_key import (
    ACTION_KEY_SCHEMA,
    action_key,
    canonical_argmax_index,
    canonical_descending_indices,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from .ai_profiles import (
    DEFAULT_HU_TURN3_STAGE9D_GATE_MODEL,
    DEFAULT_HU_TURN3_STAGE9D_MIN_GATE_PROBABILITY,
    DEFAULT_HU_TURN3_STAGE9D_MIN_MARGIN,
    DEFAULT_HU_TURN3_STAGE9D_MIN_MODEL_SCORE,
    DEFAULT_HU_TURN3_STAGE9D_MIN_SUPPORT_MARGIN,
    DEFAULT_HU_TURN3_STAGE9D_MODEL,
    DEFAULT_HU_TURN3_STAGE9D_REFERENCE_MIN_MARGIN,
    DEFAULT_HU_TURN3_STAGE9D_SUPPORT_MODEL,
    DEFAULT_HU_TURN3_STAGE7_MODEL,
    DEFAULT_HU_TURN3_STAGE7_REFERENCE_MIN_MARGIN,
    DEFAULT_HU_TURN3_STAGE7_REFERENCE_MODEL,
    DEFAULT_HU_TURN3_STAGE7_MIN_MARGIN,
    DEFAULT_OPENING_MODEL,
    DEFAULT_TURN1_MODEL,
    DEFAULT_TURN2_MODEL,
    DEFAULT_TURN3_MODEL,
    ModelPaths,
    load_model_bundle,
)
from .cards import ALL_CARDS, create_deck, validate_cards
from .counter_rng import COUNTER_RNG_SCHEMA, policy_decision_seed
from .final_turn_decision_cache import (
    FinalTurnDecisionCache,
    decide_final_turn_exact,
    final_turn_slow_state_record,
)
from .hu_self_play_teacher_data import to_act_order_for
from .hu_belief import HiddenCardParticleBatch, sample_hidden_card_particles
from .hu_infoset import (
    ActorObservation,
    ReplayTruth,
    ScoringContext,
    actor_observation_from_record,
    replay_truth_from_record,
)
from .hu_late_street_teacher import T4SearchConfig, select_t4_action
from .hu_turn3_batch_continuation import (
    HuTurn3BatchModels,
    HuTurn3ActionCache,
    HuTurn3DecisionCache,
    Stage3StateFeatureCache,
    HuTurn3Stage3ReferenceCache,
    HuTurn3Stage7BatchConfig,
    HuTurn3State,
    Turn3Decision,
    decide_hu_turn3_stage7_batch,
)
from .hu_turn3_model import hu_policy_sample
from .play_ai import _prediction_thread_context, _visible_dead_cards_for
from .policy import RegularAiPolicy, action_to_json, board_to_json, policy_sample
from .state import Board
from .teacher import DEFAULT_FL_EV, terminal_score
from .evaluator import score_board
from .turn3_model import load_action_value_model, sample_to_matrix as self_sample_to_matrix

SourceBucket = Literal[
    "natural",
    "teacher_disagreement",
    "high_regret",
    "low_margin",
    "high_margin",
    "random_off_policy",
    "from_pool",
]

T3ContinuationMode = Literal["stage3_reference_default", "stage7_m5_r10", "stage9d_p07_relaxed_both"]

STAGE3_REFERENCE_CONTINUATION_NAME = "Stage3_HU_reference_default"
STAGE7_CONTINUATION_NAME = "Stage7_candidate_A_m5_r10"
STAGE9D_CONTINUATION_NAME = "Stage9d_second3seed_gate_p07_relaxed_both"
DEFAULT_T3_CONTINUATION: T3ContinuationMode = "stage3_reference_default"
BUCKET_PREFILTER_VERSION = "cheap_mc_v1"
NO_ROLLOUT_PREFILTER_VERSION = "cheap_no_rollout_v1"


def _visible_dead_cards_for_turn2_state(
    *,
    opponent_board: Board,
    dead_cards: Iterable[str],
    visible_dead_cards: Iterable[str] | None = None,
    hero_private_discards: Iterable[str] = (),
) -> tuple[str, ...]:
    if visible_dead_cards is not None:
        return tuple(visible_dead_cards)
    hero_private = tuple(hero_private_discards)
    fallback_dead = hero_private if hero_private else tuple(dead_cards)
    return (*opponent_board.all_cards(), *fallback_dead)


def hu_turn2_policy_sample(
    board: Board,
    dealt_cards: Iterable[str],
    actions: list[Action],
    *,
    opponent_board: Board,
    dead_cards: Iterable[str],
    seat: str,
    to_act_order: str,
) -> dict[str, Any]:
    sample = hu_policy_sample(
        board,
        dealt_cards,
        actions,
        opponent_board=opponent_board,
        dead_cards=dead_cards,
        seat=seat,
        to_act_order=to_act_order,
    )
    sample["schema"] = "hu_turn2_stage1"
    sample["phase"] = "hu_turn2_7card"
    sample["turn"] = "T2"
    return sample


def evaluate_hu_turn2_actions(
    *,
    board: Board,
    dealt_cards: Iterable[str],
    opponent_board: Board,
    hero_seat: str,
    continuation_policy: RegularAiPolicy,
    opponent_policy: RegularAiPolicy,
    continuation_policy_name: str = STAGE3_REFERENCE_CONTINUATION_NAME,
    continuation_metadata: dict[str, Any] | None = None,
    baseline_turn2_model: object,
    future_samples: int,
    future_rollout_seed: int,
    action_indices: Iterable[int] | None = None,
    use_batched_continuation: bool = False,
    batched_continuation_config: HuTurn3Stage7BatchConfig | None = None,
    batched_continuation_cache: HuTurn3DecisionCache | None = None,
    batched_reference_cache: HuTurn3Stage3ReferenceCache | None = None,
    batched_state_feature_cache: Stage3StateFeatureCache | None = None,
    batched_action_cache: HuTurn3ActionCache | None = None,
    batched_continuation_batch_size: int = 8192,
    final_turn_cache: FinalTurnDecisionCache | None = None,
    use_final_turn_cache: bool = True,
    t4_search_config: T4SearchConfig | None = None,
    observation: ActorObservation | None = None,
    belief_batch: HiddenCardParticleBatch | None = None,
) -> dict[str, Any] | None:
    if board.card_count() != 7:
        raise ValueError("HU Turn2 teacher requires a 7-card hero board")
    dealt = tuple(dealt_cards)
    if observation is None or belief_batch is None:
        raise ValueError(
            "T2 teacher requires ActorObservation and HiddenCardParticleBatch"
        )
    if (
        observation.hero_board != board
        or observation.opponent_public_board != opponent_board
        or observation.dealt_cards != dealt
        or observation.seat != hero_seat
        or observation.street != "T2"
    ):
        raise ValueError("belief root disagrees with the Turn2 decision state")
    belief_batch.validate_against(observation)
    if len(belief_batch.particles) != future_samples:
        raise ValueError("belief particle count must equal future_samples")
    hero_private = observation.hero_private_discards
    visible_dead = observation.legacy_dead_cards()
    rollout_dead_root = visible_dead
    sample_started_at = time.perf_counter()
    profile: dict[str, float] = {
        "state_collection_seconds": 0.0,
        "legal_action_enumeration_seconds": 0.0,
        "common_future_generation_seconds": 0.0,
        "baseline_prediction_seconds": 0.0,
        "rollout_seconds": 0.0,
        "continuation_decision_seconds": 0.0,
        "stage7_policy_latency_seconds": 0.0,
        "feature_generation_seconds": 0.0,
        "action_serialization_seconds": 0.0,
        "hand_evaluation_seconds": 0.0,
        "jsonl_writing_seconds": 0.0,
        "non_t3_continuation_decision_seconds": 0.0,
        "final_turn_decision_seconds": 0.0,
        "raw_t3_states": 0.0,
        "unique_t3_states": 0.0,
        "unique_raw_ratio": 0.0,
        "t3_continuation_total_seconds": 0.0,
        "t3_dedup_seconds": 0.0,
        "t3_action_generation_seconds": 0.0,
        "stage3_fallback_batch_seconds": 0.0,
        "stage3_fallback_feature_seconds": 0.0,
        "stage3_fallback_model_inference_seconds": 0.0,
        "stage3_reference_batch_seconds": 0.0,
        "stage3_reference_feature_seconds": 0.0,
        "stage3_reference_model_inference_seconds": 0.0,
        "stage3_reference_computed_count": 0.0,
        "stage3_fallback_recomputed_count": 0.0,
        "stage3_fallback_reuse_count": 0.0,
        "duplicate_stage3_compute_avoided_count": 0.0,
        "stage3_feature_rows": 0.0,
        "stage3_feature_generation_time": 0.0,
        "stage3_hgb_predict_calls": 0.0,
        "stage3_hgb_predict_time": 0.0,
        "stage3_feature_rows_per_second": 0.0,
        "stage3_predict_rows_per_second": 0.0,
        "stage3_feature_mode": "scalar",
        "feature_dtype": "",
        "feature_column_count": 0.0,
        "memory_peak_mb": 0.0,
        "stage3_reference_cache_hit": 0.0,
        "stage3_reference_cache_miss": 0.0,
        "stage3_reference_cache_hit_rate": 0.0,
        "stage7_feature_batch_seconds": 0.0,
        "stage7_model_inference_seconds": 0.0,
        "t3_postprocess_gate_seconds": 0.0,
        "cache_hits": 0.0,
        "cache_misses": 0.0,
        "skipped_by_reference_margin": 0.0,
        "stage7_model_called_count": 0.0,
        "ms_per_raw_t3_continuation_decision": 0.0,
        "ms_per_unique_t3_state": 0.0,
        "peak_memory_mb": 0.0,
        "opponent_policy_decision_time": 0.0,
        "hero_non_t3_policy_decision_time": 0.0,
        "non_t3_feature_generation_time": 0.0,
        "non_t3_model_inference_time": 0.0,
        "non_t3_legal_action_time": 0.0,
        "non_t3_fallback_time": 0.0,
        "non_t3_scoring_time": 0.0,
        "non_t3_other_time": 0.0,
        "non_t3_turn_T0_seconds": 0.0,
        "non_t3_turn_T1_seconds": 0.0,
        "non_t3_turn_T2_seconds": 0.0,
        "non_t3_turn_T3_seconds": 0.0,
        "non_t3_turn_T4plus_seconds": 0.0,
        "non_t3_opponent_decision_seconds": 0.0,
        "raw_non_t3_decisions": 0.0,
        "unique_non_t3_decisions": 0.0,
        "non_t3_decision_cache_hit": 0.0,
        "non_t3_decision_cache_miss": 0.0,
        "non_t3_decision_cache_hit_rate": 0.0,
        "t3_action_generation_cache_hit": 0.0,
        "t3_action_generation_cache_miss": 0.0,
        "t3_action_generation_cache_hit_rate": 0.0,
        "stage3_reference_action_generation_seconds": 0.0,
        "stage3_reference_sample_generation_seconds": 0.0,
        "stage7_sample_generation_seconds": 0.0,
        "stage3_context_generation_time": 0.0,
        "stage3_row_global_summary_time": 0.0,
        "stage3_candidate_delta_feature_time": 0.0,
        "stage3_encoder_matrix_build_seconds": 0.0,
        "stage3_after_board_construction_seconds": 0.0,
        "stage3_row_summary_seconds": 0.0,
        "stage3_global_summary_seconds": 0.0,
        "stage3_action_delta_seconds": 0.0,
        "stage3_scalar_fallback_seconds": 0.0,
        "stage3_cache_lookup_update_seconds": 0.0,
        "stage3_column_validation_seconds": 0.0,
        "stage3_numpy_allocation_seconds": 0.0,
        "stage3_hgb_input_preparation_seconds": 0.0,
        "stage3_non_encoder_overhead_seconds": 0.0,
        "direct_column_count": 0.0,
        "scalar_fallback_column_count": 0.0,
        "direct_column_coverage_ratio": 0.0,
        "row_summary_cache_hit": 0.0,
        "row_summary_cache_miss": 0.0,
        "row_summary_cache_hit_rate": 0.0,
        "top_summary_cache_hit": 0.0,
        "top_summary_cache_miss": 0.0,
        "top_summary_cache_hit_rate": 0.0,
        "complete_row_summary_cache_hit": 0.0,
        "complete_row_summary_cache_miss": 0.0,
        "complete_row_summary_cache_hit_rate": 0.0,
        "final_turn_state_count": 0.0,
        "final_turn_unique_state_count": 0.0,
        "final_turn_exact_enumeration_time": 0.0,
        "final_turn_exact_scoring_time": 0.0,
        "final_turn_policy_decision_time": 0.0,
        "final_turn_legal_action_generation_seconds": 0.0,
        "final_turn_hand_eval_seconds": 0.0,
        "final_turn_royalty_scoring_seconds": 0.0,
        "final_turn_foul_check_seconds": 0.0,
        "final_turn_cache_lookup_seconds": 0.0,
        "final_turn_cache_update_seconds": 0.0,
        "final_turn_other_seconds": 0.0,
        "final_turn_cache_hit": 0.0,
        "final_turn_cache_miss": 0.0,
        "final_turn_cache_hit_rate": 0.0,
        "final_turn_unique_raw_ratio": 0.0,
        "final_turn_cache_memory_estimate": 0.0,
        "final_turn_hero_state_count": 0.0,
        "final_turn_opponent_state_count": 0.0,
        "final_turn_first_position_state_count": 0.0,
        "final_turn_second_position_state_count": 0.0,
        "final_turn_seconds_per_state": 0.0,
    }
    if final_turn_cache is None and use_final_turn_cache:
        final_turn_cache = FinalTurnDecisionCache()
    action_enumeration_started_at = time.perf_counter()
    actions = generate_turn_actions(board, dealt)
    profile["legal_action_enumeration_seconds"] += time.perf_counter() - action_enumeration_started_at
    if not actions:
        return None
    profile["legal_actions"] = float(len(actions))
    if action_indices is None:
        action_index_set = set(range(len(actions)))
    else:
        action_index_set = {int(index) for index in action_indices if 0 <= int(index) < len(actions)}
        if not action_index_set:
            return None
    profile["evaluated_action_count"] = float(len(action_index_set))
    future_started_at = time.perf_counter()
    future_rollouts = [particle.unseen_deck for particle in belief_batch.particles]
    opponent_private_by_rollout = [
        particle.opponent_private_discards for particle in belief_batch.particles
    ]
    profile["common_future_generation_seconds"] += time.perf_counter() - future_started_at
    if not future_rollouts:
        return None
    future_digest = _future_rollout_digest(future_rollouts)

    baseline_started_at = time.perf_counter()
    baseline_sample = policy_sample(board, dealt, actions)
    baseline_predictions = baseline_turn2_model.predict_sample(baseline_sample)
    baseline_index = canonical_argmax_index(baseline_predictions, actions)
    baseline_margin = _prediction_margin(baseline_predictions, baseline_index)
    profile["baseline_prediction_seconds"] += time.perf_counter() - baseline_started_at
    hero_log_start = _prepare_policy_log(continuation_policy)
    opponent_log_start = _prepare_policy_log(opponent_policy)
    if use_batched_continuation:
        rollout_by_action = _rollouts_after_hero_t2_actions_batched(
            board=board,
            actions=actions,
            action_indices=action_index_set,
            opponent_board=opponent_board,
            dead_cards=rollout_dead_root,
            hero_private_discards=hero_private,
            opponent_private_discards=(),
            opponent_private_discards_by_rollout=opponent_private_by_rollout,
            hero_seat=hero_seat,
            future_rollouts=future_rollouts,
            future_rollout_seed=future_rollout_seed,
            root_fingerprint=future_digest,
            hero_policy=continuation_policy,
            opponent_policy=opponent_policy,
            profile=profile,
            config=batched_continuation_config or HuTurn3Stage7BatchConfig(),
            cache=batched_continuation_cache,
            reference_cache=batched_reference_cache,
            state_feature_cache=batched_state_feature_cache,
            action_cache=batched_action_cache,
            batch_size=batched_continuation_batch_size,
            final_turn_cache=final_turn_cache,
            use_final_turn_cache=use_final_turn_cache,
            t4_search_config=t4_search_config,
        )
    else:
        rollout_by_action: dict[int, _ActionRolloutAggregate] = {}
        for action_index, action in enumerate(actions):
            if action_index not in action_index_set:
                continue
            scores: list[float] = []
            hero_after_action = board.place(action.placements)
            action_rollout_started_at = time.perf_counter()
            for future_index, (future_cards, rollout_opponent_private) in enumerate(
                zip(future_rollouts, opponent_private_by_rollout)
            ):
                score = _rollout_after_hero_t2_action(
                    hero_board=hero_after_action,
                    opponent_board=opponent_board,
                    dead_cards=(*rollout_dead_root, *action.discards),
                    hero_private_discards=(*hero_private, *action.discards),
                    opponent_private_discards=rollout_opponent_private,
                    hero_seat=hero_seat,
                    future_cards=list(future_cards),
                    future_index=future_index,
                    future_rollout_seed=future_rollout_seed,
                    root_fingerprint=future_digest,
                    hero_policy=continuation_policy,
                    opponent_policy=opponent_policy,
                    profile=profile,
                    final_turn_cache=final_turn_cache,
                    use_final_turn_cache=use_final_turn_cache,
                    t4_search_config=t4_search_config,
                )
                if score is not None:
                    scores.append(score)
            profile["rollout_seconds"] += time.perf_counter() - action_rollout_started_at
            rollout_by_action[action_index] = _ActionRolloutAggregate(scores=scores)

    action_records: list[dict[str, Any]] = []
    for action_index, action in enumerate(actions):
        aggregate = rollout_by_action.get(action_index, _ActionRolloutAggregate())
        scores = aggregate.scores
        if not scores:
            continue
        ev = float(sum(scores) / len(scores))
        se = _standard_error(scores)
        serialize_started_at = time.perf_counter()
        payload = action_to_json(board, action)
        payload.update(
            {
                "score": ev,
                "raw_score": ev,
                "ev": ev,
                "ev_standard_error": se,
                "standard_error": se,
                "rollout_count": len(scores),
                "future_count": len(scores),
                "original_index": action_index,
                "canonical_action_key": action_key(action).to_token(),
                "baseline_model_score": float(baseline_predictions[action_index]),
                "common_random_future_digest": future_digest,
                "stage7_t3_eval_count": aggregate.t3_eval_count,
                "stage7_t3_fired_count": aggregate.t3_fired_count,
                "stage7_t3_override_rate": (
                    aggregate.t3_fired_count / aggregate.t3_eval_count
                    if aggregate.t3_eval_count
                    else 0.0
                ),
                "no_override_reason_counts": dict(aggregate.no_override_reason_counts),
            }
        )
        profile["action_serialization_seconds"] += time.perf_counter() - serialize_started_at
        action_records.append(payload)

    if not action_records:
        return None
    action_records.sort(
        key=lambda item: (-float(item["score"]), str(item["canonical_action_key"]))
    )
    best = action_records[0]
    second = action_records[1] if len(action_records) > 1 else None
    baseline = _find_sorted_action(action_records, baseline_index)
    if baseline is None:
        return None

    best_ev = float(best["score"])
    second_ev = float(second["score"]) if second is not None else best_ev
    baseline_ev = float(baseline["score"])
    baseline_se = float(baseline["ev_standard_error"])
    best_se = float(best["ev_standard_error"])
    se_delta = math.sqrt(best_se * best_se + baseline_se * baseline_se)
    reference = baseline
    fallback = baseline
    paired_delta_stats = _paired_delta_stats(rollout_by_action, baseline_index, action_index_set)

    for action in action_records:
        action["delta_vs_baseline"] = float(action["score"]) - baseline_ev
        action["delta_vs_reference"] = float(action["score"]) - float(reference["score"])
        action["is_teacher_best"] = action is best
        action["is_baseline_action"] = int(action["original_index"]) == baseline_index

    feature_started_at = time.perf_counter()
    sample = hu_turn2_policy_sample(
        board,
        dealt,
        actions,
        opponent_board=opponent_board,
        dead_cards=visible_dead,
        seat=hero_seat,
        to_act_order=to_act_order_for(board, opponent_board),
    )
    profile["feature_generation_seconds"] += time.perf_counter() - feature_started_at
    if use_batched_continuation:
        stage7_stats = _stage7_aggregate_stats(rollout_by_action)
        profile["stage7_policy_latency_seconds"] = float(profile.get("stage7_model_inference_seconds", 0.0))
    else:
        stage7_stats = _stage7_log_stats(
            continuation_policy,
            opponent_policy,
            hero_log_start,
            opponent_log_start,
        )
        profile["stage7_policy_latency_seconds"] = stage7_stats["stage7_policy_latency_ms"] / 1000.0
    raw_t3_states = float(profile.get("raw_t3_states", 0.0))
    unique_t3_states = float(profile.get("unique_t3_states", 0.0))
    t3_seconds = float(profile.get("t3_continuation_total_seconds", 0.0))
    profile["unique_raw_ratio"] = unique_t3_states / raw_t3_states if raw_t3_states else 0.0
    profile["ms_per_raw_t3_continuation_decision"] = (
        t3_seconds * 1000.0 / raw_t3_states if raw_t3_states else 0.0
    )
    profile["ms_per_raw_t3_decision"] = profile["ms_per_raw_t3_continuation_decision"]
    profile["ms_per_unique_t3_state"] = (
        t3_seconds * 1000.0 / unique_t3_states if unique_t3_states else 0.0
    )
    stage3_rows = float(profile.get("stage3_feature_rows", 0.0))
    stage3_feature_seconds = float(profile.get("stage3_feature_generation_time", 0.0))
    stage3_predict_seconds = float(profile.get("stage3_hgb_predict_time", 0.0))
    profile["stage3_feature_rows_per_second"] = (
        stage3_rows / stage3_feature_seconds if stage3_feature_seconds else 0.0
    )
    profile["stage3_predict_rows_per_second"] = (
        stage3_rows / stage3_predict_seconds if stage3_predict_seconds else 0.0
    )
    stage3_cache_hit = float(profile.get("stage3_reference_cache_hit", 0.0))
    stage3_cache_miss = float(profile.get("stage3_reference_cache_miss", 0.0))
    profile["stage3_reference_cache_hit_rate"] = (
        stage3_cache_hit / (stage3_cache_hit + stage3_cache_miss)
        if stage3_cache_hit + stage3_cache_miss
        else 0.0
    )
    state_feature_hit = float(profile.get("stage3_state_feature_cache_hit", 0.0))
    state_feature_miss = float(profile.get("stage3_state_feature_cache_miss", 0.0))
    profile["stage3_state_feature_cache_hit_rate"] = (
        state_feature_hit / (state_feature_hit + state_feature_miss)
        if state_feature_hit + state_feature_miss
        else 0.0
    )
    known_non_t3 = (
        float(profile.get("non_t3_feature_generation_time", 0.0))
        + float(profile.get("non_t3_model_inference_time", 0.0))
        + float(profile.get("non_t3_legal_action_time", 0.0))
        + float(profile.get("non_t3_fallback_time", 0.0))
        + float(profile.get("non_t3_scoring_time", 0.0))
    )
    total_non_t3 = float(profile.get("non_t3_continuation_decision_seconds", 0.0))
    profile["non_t3_other_time"] = max(0.0, total_non_t3 - known_non_t3)
    non_t3_hit = float(profile.get("non_t3_decision_cache_hit", 0.0))
    non_t3_miss = float(profile.get("non_t3_decision_cache_miss", 0.0))
    profile["non_t3_decision_cache_hit_rate"] = (
        non_t3_hit / (non_t3_hit + non_t3_miss) if non_t3_hit + non_t3_miss else 0.0
    )
    t3_action_hit = float(profile.get("t3_action_generation_cache_hit", 0.0))
    t3_action_miss = float(profile.get("t3_action_generation_cache_miss", 0.0))
    profile["t3_action_generation_cache_hit_rate"] = (
        t3_action_hit / (t3_action_hit + t3_action_miss) if t3_action_hit + t3_action_miss else 0.0
    )
    final_count = float(profile.get("final_turn_state_count", 0.0))
    final_seconds = float(profile.get("final_turn_decision_seconds", 0.0))
    profile["final_turn_seconds_per_state"] = final_seconds / final_count if final_count else 0.0
    final_hit = float(profile.get("final_turn_cache_hit", 0.0))
    final_miss = float(profile.get("final_turn_cache_miss", 0.0))
    profile["final_turn_cache_hit_rate"] = (
        final_hit / (final_hit + final_miss) if final_hit + final_miss else 0.0
    )
    profile["final_turn_unique_state_count"] = final_miss
    profile["final_turn_unique_raw_ratio"] = final_miss / final_count if final_count else 0.0
    if final_turn_cache is not None:
        profile["final_turn_cache_memory_estimate"] = float(final_turn_cache.memory_estimate_bytes())
    profile["seconds_total"] = time.perf_counter() - sample_started_at
    sample.update(
        {
            "source": "hu_turn2_self_play_rollout",
            "continuation_policy_T3": continuation_policy_name,
            "t4_rollout_selector": (
                "infoset_safe_sequential"
                if t4_search_config is not None
                else "legacy_exact_self_board_first_seat"
            ),
            "t4_search_config": (
                {
                    "candidate_samples": t4_search_config.candidate_samples,
                    "evaluation_samples": t4_search_config.evaluation_samples,
                    "seed": t4_search_config.seed,
                    "candidate_seed": (
                        t4_search_config.seed
                        if t4_search_config.candidate_seed is None
                        else t4_search_config.candidate_seed
                    ),
                    "evaluation_seed": (
                        t4_search_config.seed
                        if t4_search_config.evaluation_seed is None
                        else t4_search_config.evaluation_seed
                    ),
                    "run_id": t4_search_config.run_id,
                }
                if t4_search_config is not None
                else None
            ),
            "continuation": continuation_metadata
            or _t3_continuation_metadata(DEFAULT_T3_CONTINUATION),
            "common_random_futures": True,
            "future_rollout_seed": future_rollout_seed,
            "common_random_future_digest": future_digest,
            "action_key_schema": ACTION_KEY_SCHEMA,
            "rng_schema": COUNTER_RNG_SCHEMA,
            "legal_action_set_digest": legal_action_set_digest(actions),
            "legal_action_order_digest": ordered_action_mapping_digest(actions),
            "rollout_count": len(future_rollouts),
            "actions": action_records,
            "legal_actions": action_records,
            "best_action": 0,
            "second_best_action": _action_summary(second) if second is not None else None,
            "best_margin": best_ev - second_ev,
            "score_gap": best_ev - second_ev,
            "baseline_action": _action_summary(baseline),
            "baseline_EV": baseline_ev,
            "baseline_model_margin": baseline_margin,
            "reference_action": _action_summary(reference),
            "reference_EV": float(reference["score"]),
            "fallback_action": _action_summary(fallback),
            "fallback_EV": float(fallback["score"]),
            "delta_best_vs_baseline": best_ev - baseline_ev,
            "delta_candidate_vs_baseline": best_ev - baseline_ev,
            "delta_best_vs_reference": best_ev - float(reference["score"]),
            "delta_candidate_vs_reference": best_ev - float(reference["score"]),
            "SE_delta_best_vs_baseline": se_delta,
            **paired_delta_stats,
            "opponent_policy_version": continuation_policy_name,
            "scoring_version": "regular_ofc_v1",
            "royalty_version": "regular_ofc_v1",
            "fl_ev_14": float(DEFAULT_FL_EV[14]),
            "fl_ev": {str(key): float(value) for key, value in DEFAULT_FL_EV.items()},
            "missing": 0,
            "teacher_label": _gate_label(best_ev - baseline_ev, se_delta),
            "teacher_distribution_metrics": {
                "best_original_index": int(best["original_index"]),
                "baseline_original_index": baseline_index,
                "baseline_disagreement": int(best["original_index"]) != baseline_index,
            },
            "downstream_features": {
                "t3_continuation_decision_count": stage7_stats["stage7_decision_count"],
                "t3_continuation_override_count": stage7_stats["stage7_override_count"],
                "t3_continuation_override_rate": stage7_stats["stage7_override_rate"],
                "t3_stage7_decision_count": stage7_stats["stage7_decision_count"],
                "t3_stage7_override_count": stage7_stats["stage7_override_count"],
                "t3_stage7_override_rate": stage7_stats["stage7_override_rate"],
            },
            "profiling": profile,
        }
    )
    sample.update(
        {
            "belief_conditioned": True,
            "observation_fingerprint": observation.fingerprint(),
            "belief_batch_digest": belief_batch.digest(),
            "belief_schema": belief_batch.to_dict()["belief_schema"],
            "belief_prior": belief_batch.prior,
        }
    )
    return sample


def _paired_delta_stats(
    rollout_by_action: dict[int, _ActionRolloutAggregate],
    baseline_index: int,
    action_indices: set[int],
) -> dict[str, Any]:
    """Return paired candidate-baseline delta stats for rolled-out actions."""
    if baseline_index not in action_indices:
        return {}
    summaries = _paired_delta_by_action(rollout_by_action, baseline_index, action_indices)
    if not summaries:
        return {}
    if len(action_indices) == 2:
        summary = summaries[0]
    else:
        summary = max(summaries, key=lambda item: float(item.get("mean", 0.0)))
    return {
        "paired_delta_candidate_index": int(summary["candidate_index"]),
        "paired_delta_baseline_index": int(summary["baseline_index"]),
        "paired_delta_count": int(summary["count"]),
        "paired_delta_mean": float(summary["mean"]),
        "paired_delta_standard_error": float(summary["standard_error"]),
        "paired_delta_std": float(summary["std"]),
        "paired_delta_min": float(summary["min"]),
        "paired_delta_p01": float(summary["p01"]),
        "paired_delta_p05": float(summary["p05"]),
        "paired_delta_p25": float(summary["p25"]),
        "paired_delta_p50": float(summary["p50"]),
        "paired_delta_p75": float(summary["p75"]),
        "paired_delta_p95": float(summary["p95"]),
        "paired_delta_p99": float(summary["p99"]),
        "paired_delta_max": float(summary["max"]),
        "paired_delta_lt0_rate": float(summary.get("lt0_rate", 0.0)),
        "paired_delta_le_neg6_rate": float(summary.get("le_neg6_rate", 0.0)),
        "paired_delta_le_neg12_rate": float(summary.get("le_neg12_rate", 0.0)),
        "paired_delta_le_neg20_rate": float(summary.get("le_neg20_rate", 0.0)),
        "paired_delta_by_action": summaries,
    }


def _paired_delta_by_action(
    rollout_by_action: dict[int, _ActionRolloutAggregate],
    baseline_index: int,
    action_indices: set[int],
) -> list[dict[str, Any]]:
    candidate_indices = [index for index in action_indices if index != baseline_index]
    baseline_aggregate = rollout_by_action.get(baseline_index, _ActionRolloutAggregate())
    baseline_scores = baseline_aggregate.scores
    summaries: list[dict[str, Any]] = []
    for candidate_index in sorted(candidate_indices):
        candidate_aggregate = rollout_by_action.get(candidate_index, _ActionRolloutAggregate())
        candidate_scores = candidate_aggregate.scores
        deltas = [
            float(candidate) - float(baseline)
            for candidate, baseline in zip(candidate_scores, baseline_scores)
        ]
        if deltas:
            summary = _paired_delta_summary(deltas)
            component_summaries = _paired_component_delta_summaries(
                candidate_aggregate.component_values,
                baseline_aggregate.component_values,
            )
            if component_summaries:
                summary["component_delta_summaries"] = component_summaries
            summary.update(
                {
                    "candidate_index": int(candidate_index),
                    "baseline_index": int(baseline_index),
                }
            )
            summaries.append(summary)
    return summaries


def _paired_delta_summary(deltas: list[float]) -> dict[str, Any]:
    if not deltas:
        return {}
    values = np.asarray(deltas, dtype=np.float64)
    return {
        "count": len(deltas),
        "mean": float(np.mean(values)),
        "paired_delta_standard_error": _standard_error(deltas),
        "standard_error": _standard_error(deltas),
        "std": float(np.std(values, ddof=1)) if len(deltas) > 1 else 0.0,
        "min": float(np.min(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
        "lt0_rate": float(np.mean(values < 0.0)),
        "le_neg6_rate": float(np.mean(values <= -6.0)),
        "le_neg12_rate": float(np.mean(values <= -12.0)),
        "le_neg20_rate": float(np.mean(values <= -20.0)),
    }


def _paired_component_delta_summaries(
    candidate_components: dict[str, list[float]],
    baseline_components: dict[str, list[float]],
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for key in sorted(set(candidate_components) & set(baseline_components)):
        candidate_values = candidate_components.get(key) or []
        baseline_values = baseline_components.get(key) or []
        deltas = [
            float(candidate) - float(baseline)
            for candidate, baseline in zip(candidate_values, baseline_values)
        ]
        if deltas:
            summaries[key] = _paired_delta_summary(deltas)
    return summaries


def build_hu_turn2_sample(
    *,
    sample_id: int,
    state_id: int,
    seed: int,
    hand_seed: int,
    hand_index: int,
    player: int,
    board: Board,
    dealt_cards: Iterable[str],
    opponent_board: Board,
    dead_cards: Iterable[str],
    visible_dead_cards: Iterable[str] | None = None,
    hero_private_discards: Iterable[str] = (),
    opponent_private_discards: Iterable[str] = (),
    hero_seat: str,
    continuation_policy: RegularAiPolicy,
    opponent_policy: RegularAiPolicy,
    continuation_policy_name: str = STAGE3_REFERENCE_CONTINUATION_NAME,
    continuation_metadata: dict[str, Any] | None = None,
    baseline_turn2_model: object,
    future_samples: int,
    future_rollout_seed: int,
    source_bucket: SourceBucket,
    use_batched_continuation: bool = False,
    batched_continuation_config: HuTurn3Stage7BatchConfig | None = None,
    batched_continuation_cache: HuTurn3DecisionCache | None = None,
    batched_reference_cache: HuTurn3Stage3ReferenceCache | None = None,
    batched_state_feature_cache: Stage3StateFeatureCache | None = None,
    batched_action_cache: HuTurn3ActionCache | None = None,
    batched_continuation_batch_size: int = 8192,
    final_turn_cache: FinalTurnDecisionCache | None = None,
    use_final_turn_cache: bool = True,
    t4_search_config: T4SearchConfig | None = None,
) -> dict[str, Any] | None:
    dealt_tuple = tuple(dealt_cards)
    if visible_dead_cards is None and not tuple(hero_private_discards):
        raise ValueError(
            "T2 teacher samples require explicit hero_private_discards or "
            "visible_dead_cards; ambiguous dead_cards are replay truth only"
        )
    visible_dead = _visible_dead_cards_for_turn2_state(
        opponent_board=opponent_board,
        dead_cards=dead_cards,
        visible_dead_cards=visible_dead_cards,
        hero_private_discards=hero_private_discards,
    )
    opponent_public_cards = set(opponent_board.all_cards())
    policy_hero_private = tuple(
        card for card in visible_dead if card not in opponent_public_cards
    )
    supplied_hero_private = tuple(hero_private_discards)
    if supplied_hero_private and set(supplied_hero_private) != set(
        policy_hero_private
    ):
        raise ValueError(
            "T2 visible dead cards disagree with explicit hero private discards"
        )
    truth_hero_private = supplied_hero_private or policy_hero_private
    replay_truth = ReplayTruth(
        true_dead_cards=tuple(dead_cards),
        visible_dead_cards=visible_dead,
        hero_private_discards=truth_hero_private,
        opponent_private_discards=tuple(opponent_private_discards),
    )
    policy_observation = ActorObservation(
        hero_board=board,
        opponent_public_board=opponent_board,
        dealt_cards=dealt_tuple,
        hero_private_discards=policy_hero_private,
        seat=hero_seat,  # type: ignore[arg-type]
        street="T2",
        to_act_order=to_act_order_for(board, opponent_board),
        scoring=ScoringContext(),
    )
    if set(policy_observation.legacy_dead_cards()) != set(visible_dead):
        raise ValueError("T2 policy observation disagrees with visible dead cards")
    belief_batch = sample_hidden_card_particles(
        policy_observation,
        base_seed=future_rollout_seed,
        run_id=f"hu_turn2_teacher|state={state_id}|sample={sample_id}",
        sample_count=future_samples,
    )
    sample = evaluate_hu_turn2_actions(
        board=board,
        dealt_cards=dealt_tuple,
        opponent_board=opponent_board,
        hero_seat=hero_seat,
        continuation_policy=continuation_policy,
        opponent_policy=opponent_policy,
        continuation_policy_name=continuation_policy_name,
        continuation_metadata=continuation_metadata,
        baseline_turn2_model=baseline_turn2_model,
        future_samples=future_samples,
        future_rollout_seed=future_rollout_seed,
        use_batched_continuation=use_batched_continuation,
        batched_continuation_config=batched_continuation_config,
        batched_continuation_cache=batched_continuation_cache,
        batched_reference_cache=batched_reference_cache,
        batched_state_feature_cache=batched_state_feature_cache,
        batched_action_cache=batched_action_cache,
        batched_continuation_batch_size=batched_continuation_batch_size,
        final_turn_cache=final_turn_cache,
        use_final_turn_cache=use_final_turn_cache,
        t4_search_config=t4_search_config,
        observation=policy_observation,
        belief_batch=belief_batch,
    )
    if sample is None:
        return None
    sample.update(
        {
            "sample_id": sample_id,
            "state_id": state_id,
            "hand_id": hand_seed,
            "seed": seed,
            "hand_seed": hand_seed,
            "hand_index": hand_index,
            "player": player,
            "source_bucket": source_bucket,
            "source_bucket_requested": source_bucket,
            "source_bucket_actual": source_bucket,
            "turn": "T2",
            "cards_to_place": list(dealt_tuple),
            "dead_cards": list(visible_dead),
            "visible_dead_cards": list(visible_dead),
            "hero_private_discards": list(policy_hero_private),
            "true_dead_cards": list(dead_cards),
            "true_hero_private_discards": list(replay_truth.hero_private_discards),
            "true_opponent_private_discards": list(
                replay_truth.opponent_private_discards
            ),
            "policy_observation": policy_observation.to_dict(),
            "artifact_visibility_schema": "actor_observation_plus_replay_truth_v1",
            "replay_truth": replay_truth.to_dict(),
            "replay_ready": True,
        }
    )
    return sample


def collect_hu_turn2_dataset(
    *,
    output: Path,
    samples: int,
    seed: int,
    policy_bundle: object,
    future_samples: int,
    opening_lookahead_samples: int,
    max_hands: int,
    source_bucket: SourceBucket,
    progress_every: int = 0,
    use_batched_continuation: bool = False,
    batched_continuation_batch_size: int = 8192,
    disable_continuation_cache: bool = False,
    continuation_cache_size: int = 200_000,
    disable_stage3_feature_fast_path: bool = False,
    dump_stage3_feature_replay: Path | None = None,
    stage3_feature_replay_sample_limit: int = 0,
    stage3_feature_encoder_mode: str = "scalar_fast",
    disable_final_turn_cache: bool = False,
    final_turn_cache_size: int = 200_000,
    prefilter_future_samples: int = 0,
    candidate_pool_output: Path | None = None,
    build_candidate_pool: bool = False,
    candidate_pool_size: int = 0,
    candidate_pool_prefilter: str = "cheap_mc_v1",
    candidate_pool_target_bucket: str = "",
    candidate_pool_max_attempts: int = 0,
    candidate_pool_balance_position: bool = False,
    candidate_pool_balance_source: bool = False,
    t3_continuation: T3ContinuationMode = DEFAULT_T3_CONTINUATION,
    t4_search_config: T4SearchConfig | None = None,
) -> dict[str, Any]:
    if samples <= 0:
        raise ValueError("samples must be positive")
    if max_hands <= 0:
        raise ValueError("max_hands must be positive")

    candidate_pool_target_size = candidate_pool_size if candidate_pool_size > 0 else samples
    output.parent.mkdir(parents=True, exist_ok=True)
    collected = 0
    attempts = 0
    hands = 0
    skipped_bucket = 0
    started_at = time.time()
    timing_totals: dict[str, float] = {}
    attempt_profile = _new_attempt_profile(source_bucket)
    candidate_pool_handle = None
    candidate_pool_written = 0
    candidate_pool_position_counts = {"first": 0, "second": 0}
    state_profile = "random_exact_final" if source_bucket == "random_off_policy" else "current"
    replay_source = _stage3_feature_replay_source(future_samples, samples)
    continuation_name = _t3_continuation_policy_name(t3_continuation)
    continuation_metadata = _t3_continuation_metadata(t3_continuation)
    teacher_run_hash = _stage3_feature_teacher_run_hash(
        seed=seed,
        samples=samples,
        future_samples=future_samples,
        source_bucket=source_bucket,
        stage3_feature_encoder_mode=stage3_feature_encoder_mode,
        disable_stage3_feature_fast_path=disable_stage3_feature_fast_path,
        t3_continuation=t3_continuation,
    )
    batched_config = HuTurn3Stage7BatchConfig(
        stage7_enabled=t3_continuation != "stage3_reference_default",
        hu_turn3_min_margin=_t3_continuation_min_margin(t3_continuation),
        hu_turn3_reference_min_margin=_t3_continuation_reference_min_margin(t3_continuation),
        hu_turn3_min_support_margin=_t3_continuation_min_support_margin(t3_continuation),
        hu_turn3_min_model_score=_t3_continuation_min_model_score(t3_continuation),
        hu_turn3_min_gate_probability=_t3_continuation_min_gate_probability(t3_continuation),
        batch_size=batched_continuation_batch_size,
        use_cache=not disable_continuation_cache,
        use_stage3_feature_fast_path=not disable_stage3_feature_fast_path,
        stage3_feature_encoder_mode=stage3_feature_encoder_mode,
        dump_stage3_feature_replay=str(dump_stage3_feature_replay) if dump_stage3_feature_replay else None,
        stage3_feature_replay_sample_limit=0,
        stage3_feature_replay_model_path=str(DEFAULT_HU_TURN3_STAGE7_REFERENCE_MODEL),
        stage3_feature_replay_stage7_model_path=_t3_continuation_model_path(t3_continuation),
        stage3_feature_replay_source=replay_source,
        stage3_feature_replay_teacher_run_hash=teacher_run_hash,
    )
    batched_cache = (
        None
        if disable_continuation_cache
        else HuTurn3DecisionCache(max_size=continuation_cache_size)
    )
    batched_reference_cache = (
        None
        if disable_continuation_cache
        else HuTurn3Stage3ReferenceCache(max_size=continuation_cache_size)
    )
    batched_action_cache = (
        None
        if disable_continuation_cache
        else HuTurn3ActionCache(max_size=continuation_cache_size)
    )
    batched_state_feature_cache = (
        None
        if disable_continuation_cache
        else Stage3StateFeatureCache(max_size=continuation_cache_size)
    )
    final_turn_cache = (
        None
        if disable_final_turn_cache
        else FinalTurnDecisionCache(max_size=final_turn_cache_size)
    )
    sample_profiles: list[dict[str, Any]] = []
    if candidate_pool_output is not None:
        candidate_pool_output.parent.mkdir(parents=True, exist_ok=True)
        candidate_pool_handle = candidate_pool_output.open("w", encoding="utf-8")
    with output.open("w", encoding="utf-8") as handle:
        try:
            while collected < samples and hands < max_hands:
                if build_candidate_pool and candidate_pool_written >= candidate_pool_target_size:
                    break
                if build_candidate_pool and candidate_pool_max_attempts > 0 and attempts >= candidate_pool_max_attempts:
                    break
                hand_index = hands
                hand_seed = seed + hand_index
                hands += 1
                deck = create_deck(shuffle=True, rng=random.Random(hand_seed))
                cursor = 0
                boards = [Board.from_rows(), Board.from_rows()]
                dead_cards: list[str] = []
                private_discards: list[list[str]] = [[], []]
                state_policies = [
                    _build_policy_for_profile(
                        state_profile,
                        policy_bundle,
                        seed=hand_seed * 4,
                        seat="first",
                        opening_lookahead_samples=opening_lookahead_samples,
                    ),
                    _build_policy_for_profile(
                        state_profile,
                        policy_bundle,
                        seed=hand_seed * 4 + 1,
                        seat="second",
                        opening_lookahead_samples=opening_lookahead_samples,
                    ),
                ]
                continuation_policies = [
                    _build_t3_continuation_policy(
                        t3_continuation,
                        policy_bundle,
                        seed=hand_seed * 4 + 2,
                        seat="first",
                        opening_lookahead_samples=opening_lookahead_samples,
                    ),
                    _build_t3_continuation_policy(
                        t3_continuation,
                        policy_bundle,
                        seed=hand_seed * 4 + 3,
                        seat="second",
                        opening_lookahead_samples=opening_lookahead_samples,
                    ),
                ]

                for player in (0, 1):
                    dealt = tuple(deck[cursor : cursor + 5])
                    cursor += 5
                    action = state_policies[player].choose_action(
                        boards[player],
                        dealt,
                        dead_cards=_visible_dead_cards_for(player, boards, private_discards),
                        opponent_board=boards[1 - player],
                    )
                    boards[player] = boards[player].place(action.placements)
                    dead_cards.extend(action.discards)
                    private_discards[player].extend(action.discards)

                for _round in range(1, 5):
                    for player in (0, 1):
                        dealt = tuple(deck[cursor : cursor + 3])
                        cursor += 3
                        if boards[player].card_count() == 7 and collected < samples:
                            if build_candidate_pool and candidate_pool_written >= candidate_pool_target_size:
                                continue
                            if build_candidate_pool and candidate_pool_max_attempts > 0 and attempts >= candidate_pool_max_attempts:
                                continue
                            attempts += 1
                            attempt_started_at = time.perf_counter()
                            _attempt_profile_count(attempt_profile, "attempt_count")
                            future_rollout_seed = _future_seed(seed, hand_seed, attempts, player)
                            if build_candidate_pool and candidate_pool_prefilter == NO_ROLLOUT_PREFILTER_VERSION:
                                proxy_started_at = time.perf_counter()
                                proxy = _cheap_no_rollout_proxy(
                                    board=boards[player],
                                    dealt_cards=dealt,
                                    opponent_board=boards[1 - player],
                                    dead_cards=private_discards[player],
                                    visible_dead_cards=_visible_dead_cards_for(player, boards, private_discards),
                                    hero_private_discards=private_discards[player],
                                    hero_seat="first" if player == 0 else "second",
                                    baseline_turn2_model=policy_bundle.turn2,
                                )
                                _attempt_profile_add(
                                    attempt_profile,
                                    "cheap_score_time",
                                    time.perf_counter() - proxy_started_at,
                                )
                                target_bucket = candidate_pool_target_bucket or _default_predicted_bucket(source_bucket)
                                predicate_started_at = time.perf_counter()
                                prefilter_pass, prefilter_reason = _passes_no_rollout_pool(
                                    proxy,
                                    target_bucket,
                                )
                                _attempt_profile_add(
                                    attempt_profile,
                                    "bucket_predicate_time",
                                    time.perf_counter() - predicate_started_at,
                                )
                                hero_seat = "first" if player == 0 else "second"
                                if (
                                    prefilter_pass
                                    and candidate_pool_balance_position
                                    and not _pool_position_quota_allows(
                                        candidate_pool_position_counts,
                                        hero_seat,
                                        candidate_pool_target_size,
                                    )
                                ):
                                    prefilter_pass = False
                                    prefilter_reason = "position_quota_full"
                                if prefilter_pass and candidate_pool_handle is not None:
                                    pool_record = _candidate_pool_record(
                                        seed=seed,
                                        hand_seed=hand_seed,
                                        hand_index=hand_index,
                                        state_id=attempts - 1,
                                        player=player,
                                        board=boards[player],
                                        dealt_cards=dealt,
                                        opponent_board=boards[1 - player],
                                        dead_cards=dead_cards,
                                        visible_dead_cards=_visible_dead_cards_for(player, boards, private_discards),
                                        hero_private_discards=private_discards[player],
                                        opponent_private_discards=private_discards[1 - player],
                                        hero_seat=hero_seat,
                                        source_bucket=_actual_bucket_for_predicted(target_bucket),
                                        cheap_sample=proxy,
                                        accept_reason=prefilter_reason,
                                        predicted_bucket=target_bucket,
                                        prefilter_version=NO_ROLLOUT_PREFILTER_VERSION,
                                    )
                                    candidate_pool_handle.write(
                                        json.dumps(pool_record, ensure_ascii=False, separators=(",", ":")) + "\n"
                                    )
                                    candidate_pool_handle.flush()
                                    candidate_pool_written += 1
                                    candidate_pool_position_counts[hero_seat] = (
                                        candidate_pool_position_counts.get(hero_seat, 0) + 1
                                    )
                                    _attempt_profile_count(attempt_profile, "accepted_count")
                                else:
                                    skipped_bucket += 1
                                    _attempt_profile_count(attempt_profile, "skipped_count")
                                    _attempt_profile_count(attempt_profile, "skip_before_rollout_count")
                                    _attempt_profile_reason(attempt_profile, prefilter_reason)
                                _attempt_profile_add(
                                    attempt_profile,
                                    "attempt_wall_seconds",
                                    time.perf_counter() - attempt_started_at,
                                )
                                continue
                            replay_enabled_for_sample = (
                                dump_stage3_feature_replay is not None
                                and (
                                stage3_feature_replay_sample_limit <= 0
                                    or collected < stage3_feature_replay_sample_limit
                                )
                            )
                            sample_batched_config = (
                                replace(
                                    batched_config,
                                    dump_stage3_feature_replay=(
                                        str(dump_stage3_feature_replay)
                                        if replay_enabled_for_sample
                                        else None
                                    ),
                                )
                                if dump_stage3_feature_replay is not None
                                else batched_config
                            )
                            cheap_sample: dict[str, Any] | None = None
                            prefilter_pass = True
                            prefilter_reason = "not_used"
                            if _uses_bucket_prefilter(source_bucket, prefilter_future_samples):
                                cheap_started_at = time.perf_counter()
                                cheap_sample = build_hu_turn2_sample(
                                    sample_id=collected,
                                    state_id=attempts - 1,
                                    seed=seed,
                                    hand_seed=hand_seed,
                                    hand_index=hand_index,
                                    player=player,
                                    board=boards[player],
                                    dealt_cards=dealt,
                                    opponent_board=boards[1 - player],
                                    dead_cards=dead_cards,
                                    visible_dead_cards=_visible_dead_cards_for(player, boards, private_discards),
                                    hero_private_discards=private_discards[player],
                                    opponent_private_discards=private_discards[1 - player],
                                    hero_seat="first" if player == 0 else "second",
                                    continuation_policy=continuation_policies[player],
                                    opponent_policy=continuation_policies[1 - player],
                                    continuation_policy_name=continuation_name,
                                    continuation_metadata=continuation_metadata,
                                    baseline_turn2_model=policy_bundle.turn2,
                                    future_samples=prefilter_future_samples,
                                    future_rollout_seed=_cheap_future_seed(seed, hand_seed, attempts, player),
                                    source_bucket=source_bucket,
                                    use_batched_continuation=use_batched_continuation,
                                    batched_continuation_config=sample_batched_config,
                                    batched_continuation_cache=batched_cache,
                                    batched_reference_cache=batched_reference_cache,
                                    batched_state_feature_cache=batched_state_feature_cache,
                                    batched_action_cache=batched_action_cache,
                                    batched_continuation_batch_size=batched_continuation_batch_size,
                                    final_turn_cache=final_turn_cache,
                                    use_final_turn_cache=not disable_final_turn_cache,
                                    t4_search_config=t4_search_config,
                                )
                                cheap_elapsed = time.perf_counter() - cheap_started_at
                                _attempt_profile_add(attempt_profile, "cheap_score_time", cheap_elapsed)
                                if cheap_sample is None:
                                    prefilter_pass = False
                                    prefilter_reason = "cheap_sample_none"
                                else:
                                    predicate_started_at = time.perf_counter()
                                    prefilter_pass, prefilter_reason = _passes_bucket_prefilter(
                                        cheap_sample,
                                        source_bucket,
                                    )
                                    _attempt_profile_add(
                                        attempt_profile,
                                        "bucket_predicate_time",
                                        time.perf_counter() - predicate_started_at,
                                    )
                                if not prefilter_pass:
                                    skipped_bucket += 1
                                    _attempt_profile_count(attempt_profile, "skipped_count")
                                    _attempt_profile_count(attempt_profile, "skip_before_rollout_count")
                                    _attempt_profile_reason(attempt_profile, prefilter_reason)
                                    _attempt_profile_add(
                                        attempt_profile,
                                        "attempt_wall_seconds",
                                        time.perf_counter() - attempt_started_at,
                                    )
                                    continue

                            if candidate_pool_handle is not None:
                                hero_seat = "first" if player == 0 else "second"
                                if (
                                    build_candidate_pool
                                    and candidate_pool_balance_position
                                    and not _pool_position_quota_allows(
                                        candidate_pool_position_counts,
                                        hero_seat,
                                        candidate_pool_target_size,
                                    )
                                ):
                                    skipped_bucket += 1
                                    _attempt_profile_count(attempt_profile, "skipped_count")
                                    _attempt_profile_count(attempt_profile, "skip_before_rollout_count")
                                    _attempt_profile_reason(attempt_profile, "position_quota_full")
                                    _attempt_profile_add(
                                        attempt_profile,
                                        "attempt_wall_seconds",
                                        time.perf_counter() - attempt_started_at,
                                    )
                                    continue
                                pool_record = _candidate_pool_record(
                                    seed=seed,
                                    hand_seed=hand_seed,
                                    hand_index=hand_index,
                                    state_id=attempts - 1,
                                    player=player,
                                    board=boards[player],
                                    dealt_cards=dealt,
                                    opponent_board=boards[1 - player],
                                    dead_cards=dead_cards,
                                    visible_dead_cards=_visible_dead_cards_for(player, boards, private_discards),
                                    hero_private_discards=private_discards[player],
                                    opponent_private_discards=private_discards[1 - player],
                                    hero_seat=hero_seat,
                                    source_bucket=source_bucket,
                                    cheap_sample=cheap_sample,
                                    accept_reason=prefilter_reason,
                                    predicted_bucket=candidate_pool_target_bucket or _default_predicted_bucket(source_bucket),
                                    prefilter_version=BUCKET_PREFILTER_VERSION,
                                )
                                candidate_pool_handle.write(
                                    json.dumps(pool_record, ensure_ascii=False, separators=(",", ":")) + "\n"
                                )
                                candidate_pool_handle.flush()
                                candidate_pool_written += 1
                                candidate_pool_position_counts[hero_seat] = (
                                    candidate_pool_position_counts.get(hero_seat, 0) + 1
                                )
                                if build_candidate_pool:
                                    _attempt_profile_count(attempt_profile, "accepted_count")
                                    _attempt_profile_add(
                                        attempt_profile,
                                        "attempt_wall_seconds",
                                        time.perf_counter() - attempt_started_at,
                                    )
                                    continue

                            full_started_at = time.perf_counter()
                            sample = build_hu_turn2_sample(
                                sample_id=collected,
                                state_id=attempts - 1,
                                seed=seed,
                                hand_seed=hand_seed,
                                hand_index=hand_index,
                                player=player,
                                board=boards[player],
                                dealt_cards=dealt,
                                opponent_board=boards[1 - player],
                                dead_cards=dead_cards,
                                visible_dead_cards=_visible_dead_cards_for(player, boards, private_discards),
                                hero_private_discards=private_discards[player],
                                opponent_private_discards=private_discards[1 - player],
                                hero_seat="first" if player == 0 else "second",
                                continuation_policy=continuation_policies[player],
                                opponent_policy=continuation_policies[1 - player],
                                continuation_policy_name=continuation_name,
                                continuation_metadata=continuation_metadata,
                                baseline_turn2_model=policy_bundle.turn2,
                                future_samples=future_samples,
                                future_rollout_seed=future_rollout_seed,
                                source_bucket=source_bucket,
                                use_batched_continuation=use_batched_continuation,
                                batched_continuation_config=sample_batched_config,
                                batched_continuation_cache=batched_cache,
                                batched_reference_cache=batched_reference_cache,
                                batched_state_feature_cache=batched_state_feature_cache,
                                batched_action_cache=batched_action_cache,
                                batched_continuation_batch_size=batched_continuation_batch_size,
                                final_turn_cache=final_turn_cache,
                                use_final_turn_cache=not disable_final_turn_cache,
                                t4_search_config=t4_search_config,
                            )
                            full_elapsed = time.perf_counter() - full_started_at
                            _attempt_profile_add(attempt_profile, "full_teacher_rollout_time", full_elapsed)
                            if sample is not None:
                                _attach_broad_bucket_metadata(
                                    sample,
                                    source_bucket_requested=source_bucket,
                                    source_bucket_actual=source_bucket,
                                )
                                _attempt_profile_add(
                                    attempt_profile,
                                    "baseline_reference_policy_time",
                                    float(sample.get("profiling", {}).get("baseline_prediction_seconds", 0.0)),
                                )
                            predicate_started_at = time.perf_counter()
                            full_pass = sample is not None and _passes_source_bucket(sample, source_bucket)
                            _attempt_profile_add(
                                attempt_profile,
                                "bucket_predicate_time",
                                time.perf_counter() - predicate_started_at,
                            )
                            if full_pass:
                                line = json.dumps(sample, ensure_ascii=False, separators=(",", ":"))
                                write_started_at = time.perf_counter()
                                handle.write(line + "\n")
                                handle.flush()
                                write_elapsed = time.perf_counter() - write_started_at
                                sample_profile = sample.get("profiling", {})
                                sample_profile["jsonl_writing_seconds"] = (
                                    float(sample_profile.get("jsonl_writing_seconds", 0.0)) + write_elapsed
                                )
                                sample_profiles.append(dict(sample_profile))
                                collected += 1
                                _attempt_profile_count(attempt_profile, "accepted_count")
                            else:
                                skipped_bucket += 1
                                _attempt_profile_count(attempt_profile, "skipped_count")
                                _attempt_profile_count(attempt_profile, "skip_after_rollout_count")
                                _attempt_profile_reason(
                                    attempt_profile,
                                    "full_bucket_reject" if sample is not None else "full_sample_none",
                                )
                            _attempt_profile_add(
                                attempt_profile,
                                "attempt_wall_seconds",
                                time.perf_counter() - attempt_started_at,
                            )

                        action = state_policies[player].choose_action(
                            boards[player],
                            dealt,
                            dead_cards=_visible_dead_cards_for(player, boards, private_discards),
                            opponent_board=boards[1 - player],
                        )
                        boards[player] = boards[player].place(action.placements)
                        dead_cards.extend(action.discards)
                        private_discards[player].extend(action.discards)

                if progress_every > 0 and hands % progress_every == 0:
                    print(
                        json.dumps(
                            {
                                "event": "progress",
                                "samples": collected,
                                "candidate_pool_written": candidate_pool_written,
                                "hands": hands,
                                "attempts": attempts,
                                "skipped_bucket": skipped_bucket,
                                "source_bucket": source_bucket,
                                "source_bucket_requested": source_bucket,
                                "source_bucket_actual": source_bucket,
                                "elapsed_seconds": time.time() - started_at,
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        flush=True,
                    )
        finally:
            if candidate_pool_handle is not None:
                candidate_pool_handle.close()

    if build_candidate_pool:
        collected = candidate_pool_written
    if collected < samples and not build_candidate_pool:
        raise RuntimeError(f"collected {collected}/{samples} samples after {hands} hands")
    timing_totals = aggregate_hu_turn2_profiles(sample_profiles)
    elapsed_seconds = time.time() - started_at
    state_profile_output = _default_state_profile_output(output)
    write_hu_turn2_state_profile_csv(sample_profiles, state_profile_output)
    final_turn_profile_output = _default_final_turn_profile_output(output)
    final_turn_slow_output = _default_final_turn_slow_states_output(output)
    final_turn_cache_output = _default_final_turn_cache_stats_output(output)
    attempt_profile_output = _default_attempt_profile_output(output)
    finalized_attempt_profile = _finalize_attempt_profile(attempt_profile, sample_profiles)
    write_hu_turn2_final_turn_profile_csv(sample_profiles, final_turn_profile_output)
    write_hu_turn2_final_turn_slow_states(sample_profiles, final_turn_slow_output)
    write_hu_turn2_final_turn_cache_stats(
        sample_profiles,
        final_turn_cache_output,
        cache=final_turn_cache,
        cache_enabled=not disable_final_turn_cache,
    )
    write_hu_turn2_attempt_profile_csv(finalized_attempt_profile, attempt_profile_output)
    return {
        "output": str(output),
        "samples": collected,
        "hands": hands,
        "attempts": attempts,
        "skipped_bucket": skipped_bucket,
        "candidate_pool_written": candidate_pool_written,
        "seed": seed,
        "future_samples": future_samples,
        "prefilter_future_samples": prefilter_future_samples,
        "bucket_prefilter_version": (
            candidate_pool_prefilter
            if build_candidate_pool
            else BUCKET_PREFILTER_VERSION if prefilter_future_samples > 0 else ""
        ),
        "build_candidate_pool": build_candidate_pool,
        "candidate_pool_output": str(candidate_pool_output) if candidate_pool_output else "",
        "candidate_pool_size": candidate_pool_target_size if build_candidate_pool else candidate_pool_size,
        "candidate_pool_prefilter": candidate_pool_prefilter,
        "candidate_pool_target_bucket": candidate_pool_target_bucket or _default_predicted_bucket(source_bucket),
        "candidate_pool_max_attempts": candidate_pool_max_attempts,
        "candidate_pool_balance_position": candidate_pool_balance_position,
        "candidate_pool_balance_source": candidate_pool_balance_source,
        "candidate_pool_position_counts": candidate_pool_position_counts,
        "source_bucket": source_bucket,
        "source_bucket_requested": source_bucket,
        "source_bucket_actual": source_bucket,
        "t3_continuation": t3_continuation,
        "continuation_policy_T3": continuation_name,
        "continuation": continuation_metadata,
        "scoring_objective": {
            "fl_ev_14": float(DEFAULT_FL_EV[14]),
            "fl_ev": {str(key): float(value) for key, value in DEFAULT_FL_EV.items()},
        },
        "use_batched_continuation": use_batched_continuation,
        "batched_continuation_batch_size": batched_continuation_batch_size,
        "continuation_cache_enabled": not disable_continuation_cache,
        "continuation_cache_size": continuation_cache_size if not disable_continuation_cache else 0,
        "final_turn_cache_enabled": not disable_final_turn_cache,
        "final_turn_cache_size": final_turn_cache_size if not disable_final_turn_cache else 0,
        "stage3_feature_fast_path_enabled": not disable_stage3_feature_fast_path,
        "stage3_feature_mode": "scalar" if disable_stage3_feature_fast_path else stage3_feature_encoder_mode,
        "stage3_feature_encoder_mode": stage3_feature_encoder_mode,
        "stage3_feature_replay_source": replay_source if dump_stage3_feature_replay else "",
        "source_teacher_run_hash": teacher_run_hash,
        "feature_dtype": str(timing_totals.get("feature_dtype", "float32" if not disable_stage3_feature_fast_path else "")),
        "feature_column_count": int(timing_totals.get("feature_column_count", 0)),
        "elapsed_seconds": elapsed_seconds,
        "timing_totals": timing_totals,
        "profile_distributions": summarize_hu_turn2_profile_distributions(sample_profiles),
        "state_profile_csv": str(state_profile_output),
        "final_turn_profile_csv": str(final_turn_profile_output),
        "final_turn_slow_states_top50_jsonl": str(final_turn_slow_output),
        "final_turn_cache_stats_json": str(final_turn_cache_output),
        "attempt_profile_csv": str(attempt_profile_output),
        "attempt_profile": finalized_attempt_profile,
        "seconds_per_state": elapsed_seconds / max(collected, 1),
    }


def collect_hu_turn2_dataset_from_candidate_pool(
    *,
    candidate_pool_input: Path,
    output: Path,
    samples: int,
    seed: int,
    policy_bundle: object,
    future_samples: int,
    opening_lookahead_samples: int,
    source_bucket: SourceBucket,
    use_batched_continuation: bool = False,
    batched_continuation_batch_size: int = 8192,
    disable_continuation_cache: bool = False,
    continuation_cache_size: int = 200_000,
    disable_stage3_feature_fast_path: bool = False,
    stage3_feature_encoder_mode: str = "scalar_fast",
    disable_final_turn_cache: bool = False,
    final_turn_cache_size: int = 200_000,
    t3_continuation: T3ContinuationMode = DEFAULT_T3_CONTINUATION,
    t4_search_config: T4SearchConfig | None = None,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    collected = 0
    attempts = 0
    skipped_bucket = 0
    sample_profiles: list[dict[str, Any]] = []
    attempt_profile = _new_attempt_profile(source_bucket)
    replay_source = _stage3_feature_replay_source(future_samples, samples)
    continuation_name = _t3_continuation_policy_name(t3_continuation)
    continuation_metadata = _t3_continuation_metadata(t3_continuation)
    teacher_run_hash = _stage3_feature_teacher_run_hash(
        seed=seed,
        samples=samples,
        future_samples=future_samples,
        source_bucket=source_bucket,
        stage3_feature_encoder_mode=stage3_feature_encoder_mode,
        disable_stage3_feature_fast_path=disable_stage3_feature_fast_path,
        t3_continuation=t3_continuation,
    )
    batched_config = HuTurn3Stage7BatchConfig(
        stage7_enabled=t3_continuation != "stage3_reference_default",
        hu_turn3_min_margin=_t3_continuation_min_margin(t3_continuation),
        hu_turn3_reference_min_margin=_t3_continuation_reference_min_margin(t3_continuation),
        hu_turn3_min_support_margin=_t3_continuation_min_support_margin(t3_continuation),
        hu_turn3_min_model_score=_t3_continuation_min_model_score(t3_continuation),
        hu_turn3_min_gate_probability=_t3_continuation_min_gate_probability(t3_continuation),
        batch_size=batched_continuation_batch_size,
        use_cache=not disable_continuation_cache,
        use_stage3_feature_fast_path=not disable_stage3_feature_fast_path,
        stage3_feature_encoder_mode=stage3_feature_encoder_mode,
        stage3_feature_replay_model_path=str(DEFAULT_HU_TURN3_STAGE7_REFERENCE_MODEL),
        stage3_feature_replay_stage7_model_path=_t3_continuation_model_path(t3_continuation),
        stage3_feature_replay_source=replay_source,
        stage3_feature_replay_teacher_run_hash=teacher_run_hash,
    )
    batched_cache = None if disable_continuation_cache else HuTurn3DecisionCache(max_size=continuation_cache_size)
    batched_reference_cache = None if disable_continuation_cache else HuTurn3Stage3ReferenceCache(max_size=continuation_cache_size)
    batched_action_cache = None if disable_continuation_cache else HuTurn3ActionCache(max_size=continuation_cache_size)
    batched_state_feature_cache = None if disable_continuation_cache else Stage3StateFeatureCache(max_size=continuation_cache_size)
    final_turn_cache = None if disable_final_turn_cache else FinalTurnDecisionCache(max_size=final_turn_cache_size)

    with candidate_pool_input.open(encoding="utf-8") as pool_handle, output.open("w", encoding="utf-8") as output_handle:
        for line in pool_handle:
            if collected >= samples:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            if source_bucket != "from_pool" and record.get("source_bucket_requested") != source_bucket:
                continue
            attempts += 1
            _attempt_profile_count(attempt_profile, "attempt_count")
            attempt_started_at = time.perf_counter()
            hand_seed = int(record.get("hand_seed", seed + attempts))
            hand_index = int(record.get("hand_index", attempts - 1))
            player = int(record.get("player", 0))
            hero_seat = str(record.get("seat", "first" if player == 0 else "second"))
            board = _board_from_json(record["board"])
            opponent_board = _board_from_json(record["opponent_board"])
            dealt = tuple(record.get("cards_to_place") or record.get("dealt") or ())
            pool_observation = actor_observation_from_record(record)
            pool_truth = replay_truth_from_record(record)
            continuation_policies = [
                _build_t3_continuation_policy(
                    t3_continuation,
                    policy_bundle,
                    seed=hand_seed * 4 + 2,
                    seat="first",
                    opening_lookahead_samples=opening_lookahead_samples,
                ),
                _build_t3_continuation_policy(
                    t3_continuation,
                    policy_bundle,
                    seed=hand_seed * 4 + 3,
                    seat="second",
                    opening_lookahead_samples=opening_lookahead_samples,
                ),
            ]
            full_started_at = time.perf_counter()
            sample = build_hu_turn2_sample(
                sample_id=collected,
                state_id=int(record.get("state_id", attempts - 1)),
                seed=seed,
                hand_seed=hand_seed,
                hand_index=hand_index,
                player=player,
                board=board,
                dealt_cards=dealt,
                opponent_board=opponent_board,
                dead_cards=pool_truth.true_dead_cards,
                visible_dead_cards=pool_observation.legacy_dead_cards(),
                hero_private_discards=pool_truth.hero_private_discards,
                opponent_private_discards=pool_truth.opponent_private_discards,
                hero_seat=hero_seat,
                continuation_policy=continuation_policies[player],
                opponent_policy=continuation_policies[1 - player],
                continuation_policy_name=continuation_name,
                continuation_metadata=continuation_metadata,
                baseline_turn2_model=policy_bundle.turn2,
                future_samples=future_samples,
                future_rollout_seed=_future_seed(seed, hand_seed, attempts, player),
                source_bucket=source_bucket,
                use_batched_continuation=use_batched_continuation,
                batched_continuation_config=batched_config,
                batched_continuation_cache=batched_cache,
                batched_reference_cache=batched_reference_cache,
                batched_state_feature_cache=batched_state_feature_cache,
                batched_action_cache=batched_action_cache,
                batched_continuation_batch_size=batched_continuation_batch_size,
                final_turn_cache=final_turn_cache,
                use_final_turn_cache=not disable_final_turn_cache,
                t4_search_config=t4_search_config,
            )
            _attempt_profile_add(attempt_profile, "full_teacher_rollout_time", time.perf_counter() - full_started_at)
            if sample is not None:
                _attach_broad_bucket_metadata(
                    sample,
                    source_bucket_requested=str(record.get("source_bucket_requested", source_bucket)),
                    source_bucket_actual=str(record.get("source_bucket_actual", source_bucket)),
                    pool_record=record,
                )
                _attempt_profile_add(
                    attempt_profile,
                    "baseline_reference_policy_time",
                    float(sample.get("profiling", {}).get("baseline_prediction_seconds", 0.0)),
                )
            predicate_started_at = time.perf_counter()
            full_pass = sample is not None and _passes_source_bucket(sample, source_bucket)
            _attempt_profile_add(attempt_profile, "bucket_predicate_time", time.perf_counter() - predicate_started_at)
            if full_pass:
                write_started_at = time.perf_counter()
                output_handle.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
                output_handle.flush()
                write_elapsed = time.perf_counter() - write_started_at
                sample_profile = sample.get("profiling", {})
                sample_profile["jsonl_writing_seconds"] = (
                    float(sample_profile.get("jsonl_writing_seconds", 0.0)) + write_elapsed
                )
                sample_profiles.append(dict(sample_profile))
                collected += 1
                _attempt_profile_count(attempt_profile, "accepted_count")
            else:
                skipped_bucket += 1
                _attempt_profile_count(attempt_profile, "skipped_count")
                _attempt_profile_count(attempt_profile, "skip_after_rollout_count")
                _attempt_profile_reason(
                    attempt_profile,
                    "pool_full_bucket_reject" if sample is not None else "pool_full_sample_none",
                )
            _attempt_profile_add(attempt_profile, "attempt_wall_seconds", time.perf_counter() - attempt_started_at)

    if collected < samples:
        raise RuntimeError(
            f"collected {collected}/{samples} samples from pool {candidate_pool_input}"
        )
    timing_totals = aggregate_hu_turn2_profiles(sample_profiles)
    elapsed_seconds = time.time() - started_at
    state_profile_output = _default_state_profile_output(output)
    final_turn_profile_output = _default_final_turn_profile_output(output)
    final_turn_slow_output = _default_final_turn_slow_states_output(output)
    final_turn_cache_output = _default_final_turn_cache_stats_output(output)
    attempt_profile_output = _default_attempt_profile_output(output)
    finalized_attempt_profile = _finalize_attempt_profile(attempt_profile, sample_profiles)
    write_hu_turn2_state_profile_csv(sample_profiles, state_profile_output)
    write_hu_turn2_final_turn_profile_csv(sample_profiles, final_turn_profile_output)
    write_hu_turn2_final_turn_slow_states(sample_profiles, final_turn_slow_output)
    write_hu_turn2_final_turn_cache_stats(
        sample_profiles,
        final_turn_cache_output,
        cache=final_turn_cache,
        cache_enabled=not disable_final_turn_cache,
    )
    write_hu_turn2_attempt_profile_csv(finalized_attempt_profile, attempt_profile_output)
    return {
        "output": str(output),
        "samples": collected,
        "attempts": attempts,
        "skipped_bucket": skipped_bucket,
        "candidate_pool_input": str(candidate_pool_input),
        "seed": seed,
        "future_samples": future_samples,
        "source_bucket": source_bucket,
        "source_bucket_requested": source_bucket,
        "source_bucket_actual": source_bucket,
        "t3_continuation": t3_continuation,
        "continuation_policy_T3": continuation_name,
        "continuation": continuation_metadata,
        "scoring_objective": {
            "fl_ev_14": float(DEFAULT_FL_EV[14]),
            "fl_ev": {str(key): float(value) for key, value in DEFAULT_FL_EV.items()},
        },
        "use_batched_continuation": use_batched_continuation,
        "batched_continuation_batch_size": batched_continuation_batch_size,
        "continuation_cache_enabled": not disable_continuation_cache,
        "continuation_cache_size": continuation_cache_size if not disable_continuation_cache else 0,
        "final_turn_cache_enabled": not disable_final_turn_cache,
        "final_turn_cache_size": final_turn_cache_size if not disable_final_turn_cache else 0,
        "stage3_feature_fast_path_enabled": not disable_stage3_feature_fast_path,
        "stage3_feature_mode": "scalar" if disable_stage3_feature_fast_path else stage3_feature_encoder_mode,
        "stage3_feature_encoder_mode": stage3_feature_encoder_mode,
        "source_teacher_run_hash": teacher_run_hash,
        "feature_dtype": str(timing_totals.get("feature_dtype", "float32" if not disable_stage3_feature_fast_path else "")),
        "feature_column_count": int(timing_totals.get("feature_column_count", 0)),
        "elapsed_seconds": elapsed_seconds,
        "timing_totals": timing_totals,
        "profile_distributions": summarize_hu_turn2_profile_distributions(sample_profiles),
        "state_profile_csv": str(state_profile_output),
        "final_turn_profile_csv": str(final_turn_profile_output),
        "final_turn_slow_states_top50_jsonl": str(final_turn_slow_output),
        "final_turn_cache_stats_json": str(final_turn_cache_output),
        "attempt_profile_csv": str(attempt_profile_output),
        "attempt_profile": finalized_attempt_profile,
        "seconds_per_state": elapsed_seconds / max(collected, 1),
    }


SUMMARY_SUM_KEYS = {
    "seconds_total",
    "state_collection_seconds",
    "legal_action_enumeration_seconds",
    "common_future_generation_seconds",
    "baseline_prediction_seconds",
    "rollout_seconds",
    "continuation_decision_seconds",
    "stage7_policy_latency_seconds",
    "feature_generation_seconds",
    "action_serialization_seconds",
    "hand_evaluation_seconds",
    "jsonl_writing_seconds",
    "non_t3_continuation_decision_seconds",
    "final_turn_decision_seconds",
    "t3_continuation_total_seconds",
    "t3_dedup_seconds",
    "t3_action_generation_seconds",
    "stage3_fallback_batch_seconds",
    "stage3_fallback_feature_seconds",
    "stage3_fallback_model_inference_seconds",
    "stage3_reference_batch_seconds",
    "stage3_reference_feature_seconds",
    "stage3_reference_model_inference_seconds",
    "stage3_feature_generation_time",
    "stage3_hgb_predict_time",
    "stage3_feature_generation_total",
    "stage3_state_feature_time",
    "stage3_action_feature_time",
    "stage3_matrix_assembly_time",
    "stage7_feature_batch_seconds",
    "stage7_model_inference_seconds",
    "stage7_sample_generation_seconds",
    "support_feature_batch_seconds",
    "support_model_inference_seconds",
    "gate_self_feature_batch_seconds",
    "gate_self_model_inference_seconds",
    "gate_probability_seconds",
    "t3_postprocess_gate_seconds",
    "opponent_policy_decision_time",
    "hero_non_t3_policy_decision_time",
    "non_t3_feature_generation_time",
    "non_t3_model_inference_time",
    "non_t3_legal_action_time",
    "non_t3_fallback_time",
    "non_t3_scoring_time",
    "non_t3_other_time",
    "non_t3_turn_T0_seconds",
    "non_t3_turn_T1_seconds",
    "non_t3_turn_T2_seconds",
    "non_t3_turn_T3_seconds",
    "non_t3_turn_T4plus_seconds",
    "non_t3_opponent_decision_seconds",
    "stage3_reference_action_generation_seconds",
    "stage3_reference_sample_generation_seconds",
    "stage3_context_generation_time",
    "stage3_row_global_summary_time",
    "stage3_candidate_delta_feature_time",
    "stage3_encoder_matrix_build_seconds",
    "stage3_after_board_construction_seconds",
    "stage3_row_summary_seconds",
    "stage3_global_summary_seconds",
    "stage3_action_delta_seconds",
    "stage3_scalar_fallback_seconds",
    "stage3_cache_lookup_update_seconds",
    "stage3_column_validation_seconds",
    "stage3_numpy_allocation_seconds",
    "stage3_hgb_input_preparation_seconds",
    "stage3_non_encoder_overhead_seconds",
    "final_turn_exact_enumeration_time",
    "final_turn_exact_scoring_time",
    "final_turn_policy_decision_time",
    "final_turn_legal_action_generation_seconds",
    "final_turn_hand_eval_seconds",
    "final_turn_royalty_scoring_seconds",
    "final_turn_foul_check_seconds",
    "final_turn_cache_lookup_seconds",
    "final_turn_cache_update_seconds",
    "final_turn_other_seconds",
}

SUMMARY_COUNT_SUM_KEYS = {
    "raw_t3_states",
    "unique_t3_states",
    "stage3_reference_computed_count",
    "stage3_fallback_recomputed_count",
    "stage3_fallback_reuse_count",
    "duplicate_stage3_compute_avoided_count",
    "stage3_feature_rows",
    "stage3_hgb_predict_calls",
    "stage3_reference_cache_hit",
    "stage3_reference_cache_miss",
    "stage3_state_feature_cache_hit",
    "stage3_state_feature_cache_miss",
    "cache_hits",
    "cache_misses",
    "skipped_by_reference_margin",
    "stage7_model_called_count",
    "support_model_called_count",
    "gate_model_called_count",
    "raw_non_t3_decisions",
    "unique_non_t3_decisions",
    "non_t3_decision_cache_hit",
    "non_t3_decision_cache_miss",
    "stage3_unique_states",
    "stage3_total_legal_actions",
    "t3_action_generation_cache_hit",
    "t3_action_generation_cache_miss",
    "row_summary_cache_hit",
    "row_summary_cache_miss",
    "top_summary_cache_hit",
    "top_summary_cache_miss",
    "complete_row_summary_cache_hit",
    "complete_row_summary_cache_miss",
    "final_turn_state_count",
    "final_turn_unique_state_count",
    "final_turn_cache_hit",
    "final_turn_cache_miss",
    "final_turn_hero_state_count",
    "final_turn_opponent_state_count",
    "final_turn_first_position_state_count",
    "final_turn_second_position_state_count",
}

SUMMARY_MAX_KEYS = {
    "memory_peak_mb",
    "peak_memory_mb",
    "legal_actions",
    "raw_t3_states",
    "unique_t3_states",
}

SUMMARY_CONSTANT_KEYS = {
    "feature_column_count",
    "feature_dtype",
    "stage3_feature_mode",
    "feature_schema_version",
    "direct_column_count",
    "scalar_fallback_column_count",
    "direct_column_coverage_ratio",
}

SUMMARY_RATE_KEYS = {
    "unique_raw_ratio",
    "ms_per_raw_t3_continuation_decision",
    "ms_per_raw_t3_decision",
    "ms_per_unique_t3_state",
    "stage3_feature_rows_per_second",
    "stage3_predict_rows_per_second",
    "stage3_reference_cache_hit_rate",
    "stage3_state_feature_cache_hit_rate",
    "non_t3_decision_cache_hit_rate",
    "t3_action_generation_cache_hit_rate",
    "row_summary_cache_hit_rate",
    "top_summary_cache_hit_rate",
    "complete_row_summary_cache_hit_rate",
    "final_turn_seconds_per_state",
    "final_turn_cache_hit_rate",
    "final_turn_unique_raw_ratio",
}

STATE_PROFILE_COLUMNS = [
    "state_index",
    "seconds",
    "legal_actions",
    "raw_t3_states",
    "unique_t3_states",
    "unique_raw_ratio",
    "stage3_feature_generation_seconds",
    "stage3_state_feature_seconds",
    "stage3_action_feature_seconds",
    "stage3_matrix_assembly_seconds",
    "stage3_model_inference_seconds",
    "stage3_encoder_matrix_build_seconds",
    "stage3_after_board_construction_seconds",
    "stage3_row_summary_seconds",
    "stage3_global_summary_seconds",
    "stage3_action_delta_seconds",
    "stage3_scalar_fallback_seconds",
    "stage3_non_encoder_overhead_seconds",
    "stage7_inference_seconds",
    "support_inference_seconds",
    "gate_self_inference_seconds",
    "gate_probability_seconds",
    "t3_action_generation_seconds",
    "final_turn_decision_seconds",
    "final_turn_exact_enumeration_seconds",
    "final_turn_exact_scoring_seconds",
    "final_turn_policy_decision_seconds",
    "final_turn_legal_action_generation_seconds",
    "final_turn_hand_eval_seconds",
    "final_turn_royalty_scoring_seconds",
    "final_turn_foul_check_seconds",
    "final_turn_cache_lookup_seconds",
    "final_turn_cache_update_seconds",
    "final_turn_other_seconds",
    "final_turn_state_count",
    "final_turn_unique_state_count",
    "final_turn_cache_hit",
    "final_turn_cache_miss",
    "final_turn_cache_hit_rate",
    "final_turn_unique_raw_ratio",
    "final_turn_cache_memory_estimate",
    "final_turn_hero_state_count",
    "final_turn_opponent_state_count",
    "final_turn_first_position_state_count",
    "final_turn_second_position_state_count",
    "non_t3_continuation_seconds",
    "memory_peak_mb",
    "stage7_model_called",
    "support_model_called",
    "gate_model_called",
    "skipped_by_reference_margin",
    "fallback_recompute",
    "fallback_reuse",
    "stage3_feature_rows",
    "feature_columns",
    "direct_column_count",
    "scalar_fallback_column_count",
    "direct_column_coverage_ratio",
]


def aggregate_hu_turn2_profiles(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, Any] = {}
    for key in SUMMARY_SUM_KEYS | SUMMARY_COUNT_SUM_KEYS:
        totals[key] = sum(_as_float(profile.get(key, 0.0)) for profile in profiles)

    totals["memory_peak_mb"] = max((_as_float(profile.get("memory_peak_mb", 0.0)) for profile in profiles), default=0.0)
    totals["peak_memory_mb"] = max((_as_float(profile.get("peak_memory_mb", 0.0)) for profile in profiles), default=0.0)
    totals["max_legal_actions"] = max((_as_float(profile.get("legal_actions", 0.0)) for profile in profiles), default=0.0)
    totals["max_raw_t3_states_per_state"] = max((_as_float(profile.get("raw_t3_states", 0.0)) for profile in profiles), default=0.0)
    totals["max_unique_t3_states_per_state"] = max((_as_float(profile.get("unique_t3_states", 0.0)) for profile in profiles), default=0.0)
    totals["max_seconds_per_state"] = max((_as_float(profile.get("seconds_total", 0.0)) for profile in profiles), default=0.0)

    constant_mismatches: dict[str, list[Any]] = {}
    for key in SUMMARY_CONSTANT_KEYS:
        values = [profile.get(key) for profile in profiles if profile.get(key) not in (None, "", 0, 0.0)]
        unique_values = []
        for value in values:
            if value not in unique_values:
                unique_values.append(value)
        if unique_values:
            totals[key] = unique_values[0]
            if len(unique_values) > 1:
                constant_mismatches[key] = unique_values
    if constant_mismatches:
        totals["constant_mismatches"] = constant_mismatches

    raw_t3 = _as_float(totals.get("raw_t3_states", 0.0))
    unique_t3 = _as_float(totals.get("unique_t3_states", 0.0))
    t3_seconds = _as_float(totals.get("t3_continuation_total_seconds", 0.0))
    totals["unique_raw_ratio"] = unique_t3 / raw_t3 if raw_t3 else 0.0
    totals["ms_per_raw_t3_continuation_decision"] = t3_seconds * 1000.0 / raw_t3 if raw_t3 else 0.0
    totals["ms_per_raw_t3_decision"] = totals["ms_per_raw_t3_continuation_decision"]
    totals["ms_per_unique_t3_state"] = t3_seconds * 1000.0 / unique_t3 if unique_t3 else 0.0
    totals["seconds_per_state"] = _as_float(totals.get("seconds_total", 0.0)) / len(profiles) if profiles else 0.0

    rows = _as_float(totals.get("stage3_feature_rows", 0.0))
    feature_seconds = _as_float(totals.get("stage3_feature_generation_time", 0.0))
    predict_seconds = _as_float(totals.get("stage3_hgb_predict_time", 0.0))
    totals["stage3_feature_rows_per_second"] = rows / feature_seconds if feature_seconds else 0.0
    totals["stage3_predict_rows_per_second"] = rows / predict_seconds if predict_seconds else 0.0
    _put_rate(totals, "stage3_reference_cache_hit_rate", "stage3_reference_cache_hit", "stage3_reference_cache_miss")
    _put_rate(totals, "stage3_state_feature_cache_hit_rate", "stage3_state_feature_cache_hit", "stage3_state_feature_cache_miss")
    _put_rate(totals, "non_t3_decision_cache_hit_rate", "non_t3_decision_cache_hit", "non_t3_decision_cache_miss")
    _put_rate(totals, "t3_action_generation_cache_hit_rate", "t3_action_generation_cache_hit", "t3_action_generation_cache_miss")
    _put_rate(totals, "row_summary_cache_hit_rate", "row_summary_cache_hit", "row_summary_cache_miss")
    _put_rate(totals, "top_summary_cache_hit_rate", "top_summary_cache_hit", "top_summary_cache_miss")
    _put_rate(totals, "complete_row_summary_cache_hit_rate", "complete_row_summary_cache_hit", "complete_row_summary_cache_miss")
    final_count = _as_float(totals.get("final_turn_state_count", 0.0))
    totals["final_turn_seconds_per_state"] = (
        _as_float(totals.get("final_turn_decision_seconds", 0.0)) / final_count if final_count else 0.0
    )
    _put_rate(totals, "final_turn_cache_hit_rate", "final_turn_cache_hit", "final_turn_cache_miss")
    final_unique = _as_float(totals.get("final_turn_cache_miss", 0.0))
    totals["final_turn_unique_state_count"] = final_unique
    totals["final_turn_unique_raw_ratio"] = final_unique / final_count if final_count else 0.0
    totals["final_turn_cache_memory_estimate"] = max(
        (_as_float(profile.get("final_turn_cache_memory_estimate", 0.0)) for profile in profiles),
        default=0.0,
    )
    _put_feature_attribution_shares(totals)
    totals["summary_aggregation_version"] = "hu_turn2_batch6_5_v1"
    return totals


def summarize_hu_turn2_profile_distributions(profiles: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {
        "seconds_per_state": _distribution_stats([_as_float(profile.get("seconds_total", 0.0)) for profile in profiles]),
        "raw_t3_states_per_state": _distribution_stats([_as_float(profile.get("raw_t3_states", 0.0)) for profile in profiles]),
        "legal_actions_per_state": _distribution_stats([_as_float(profile.get("legal_actions", 0.0)) for profile in profiles]),
        "stage3_feature_generation_per_state": _distribution_stats(
            [_as_float(profile.get("stage3_feature_generation_time", 0.0)) for profile in profiles]
        ),
        "t3_action_generation_per_state": _distribution_stats(
            [_as_float(profile.get("t3_action_generation_seconds", 0.0)) for profile in profiles]
        ),
        "final_turn_decision_per_state": _distribution_stats(
            [_as_float(profile.get("final_turn_decision_seconds", 0.0)) for profile in profiles]
        ),
        "final_turn_exact_scoring_per_state": _distribution_stats(
            [_as_float(profile.get("final_turn_exact_scoring_time", 0.0)) for profile in profiles]
        ),
        "final_turn_legal_action_generation_per_state": _distribution_stats(
            [_as_float(profile.get("final_turn_legal_action_generation_seconds", 0.0)) for profile in profiles]
        ),
    }


def write_hu_turn2_state_profile_csv(profiles: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATE_PROFILE_COLUMNS)
        writer.writeheader()
        for index, profile in enumerate(profiles):
            writer.writerow(_state_profile_csv_row(index, profile))


def _default_state_profile_output(output: Path) -> Path:
    return output.with_name(f"{output.stem}_state_profile.csv")


FINAL_TURN_PROFILE_COLUMNS = [
    "state_index",
    "final_turn_state_count",
    "final_turn_unique_state_count",
    "final_turn_decision_seconds",
    "final_turn_exact_enumeration_seconds",
    "final_turn_exact_scoring_seconds",
    "final_turn_policy_decision_seconds",
    "final_turn_legal_action_generation_seconds",
    "final_turn_hand_eval_seconds",
    "final_turn_royalty_scoring_seconds",
    "final_turn_foul_check_seconds",
    "final_turn_cache_lookup_seconds",
    "final_turn_cache_update_seconds",
    "final_turn_other_seconds",
    "final_turn_cache_hit",
    "final_turn_cache_miss",
    "final_turn_cache_hit_rate",
    "final_turn_unique_raw_ratio",
    "final_turn_cache_memory_estimate",
    "final_turn_hero_state_count",
    "final_turn_opponent_state_count",
    "final_turn_first_position_state_count",
    "final_turn_second_position_state_count",
]


def write_hu_turn2_final_turn_profile_csv(profiles: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FINAL_TURN_PROFILE_COLUMNS)
        writer.writeheader()
        for index, profile in enumerate(profiles):
            writer.writerow(_final_turn_profile_csv_row(index, profile))


def write_hu_turn2_final_turn_slow_states(
    profiles: list[dict[str, Any]],
    output: Path,
    *,
    limit: int = 50,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for state_index, profile in enumerate(profiles):
        for record in profile.get("_final_turn_slow_states", []) or []:
            if isinstance(record, dict):
                payload = dict(record)
                payload["state_index"] = state_index
                records.append(payload)
    records.sort(key=lambda item: float(item.get("seconds", 0.0)), reverse=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records[:limit]:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_hu_turn2_final_turn_cache_stats(
    profiles: list[dict[str, Any]],
    output: Path,
    *,
    cache: FinalTurnDecisionCache | None,
    cache_enabled: bool,
) -> None:
    totals = aggregate_hu_turn2_profiles(profiles)
    payload = {
        "cache_enabled": cache_enabled,
        "cache_size": len(cache) if cache is not None else 0,
        "cache_memory_estimate_bytes": cache.memory_estimate_bytes() if cache is not None else 0,
        "final_turn_state_count": totals.get("final_turn_state_count", 0.0),
        "final_turn_unique_state_count": totals.get("final_turn_unique_state_count", 0.0),
        "final_turn_cache_hit": totals.get("final_turn_cache_hit", 0.0),
        "final_turn_cache_miss": totals.get("final_turn_cache_miss", 0.0),
        "final_turn_cache_hit_rate": totals.get("final_turn_cache_hit_rate", 0.0),
        "final_turn_unique_raw_ratio": totals.get("final_turn_unique_raw_ratio", 0.0),
        "final_turn_decision_seconds": totals.get("final_turn_decision_seconds", 0.0),
        "final_turn_exact_enumeration_seconds": totals.get("final_turn_exact_enumeration_time", 0.0),
        "final_turn_exact_scoring_seconds": totals.get("final_turn_exact_scoring_time", 0.0),
        "final_turn_legal_action_generation_seconds": totals.get("final_turn_legal_action_generation_seconds", 0.0),
        "final_turn_hand_eval_seconds": totals.get("final_turn_hand_eval_seconds", 0.0),
        "final_turn_other_seconds": totals.get("final_turn_other_seconds", 0.0),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _default_final_turn_profile_output(output: Path) -> Path:
    return output.with_name(f"{output.stem}_final_turn_profile.csv")


def _default_final_turn_slow_states_output(output: Path) -> Path:
    return output.with_name(f"{output.stem}_final_turn_slow_states_top50.jsonl")


def _default_final_turn_cache_stats_output(output: Path) -> Path:
    return output.with_name(f"{output.stem}_final_turn_cache_stats.json")


def _state_profile_csv_row(index: int, profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "state_index": index,
        "seconds": _as_float(profile.get("seconds_total", 0.0)),
        "legal_actions": _as_float(profile.get("legal_actions", 0.0)),
        "raw_t3_states": _as_float(profile.get("raw_t3_states", 0.0)),
        "unique_t3_states": _as_float(profile.get("unique_t3_states", 0.0)),
        "unique_raw_ratio": _as_float(profile.get("unique_raw_ratio", 0.0)),
        "stage3_feature_generation_seconds": _as_float(profile.get("stage3_feature_generation_time", 0.0)),
        "stage3_state_feature_seconds": _as_float(profile.get("stage3_state_feature_time", 0.0)),
        "stage3_action_feature_seconds": _as_float(profile.get("stage3_action_feature_time", 0.0)),
        "stage3_matrix_assembly_seconds": _as_float(profile.get("stage3_matrix_assembly_time", 0.0)),
        "stage3_model_inference_seconds": _as_float(profile.get("stage3_hgb_predict_time", 0.0)),
        "stage3_encoder_matrix_build_seconds": _as_float(profile.get("stage3_encoder_matrix_build_seconds", 0.0)),
        "stage3_after_board_construction_seconds": _as_float(profile.get("stage3_after_board_construction_seconds", 0.0)),
        "stage3_row_summary_seconds": _as_float(profile.get("stage3_row_summary_seconds", 0.0)),
        "stage3_global_summary_seconds": _as_float(profile.get("stage3_global_summary_seconds", 0.0)),
        "stage3_action_delta_seconds": _as_float(profile.get("stage3_action_delta_seconds", 0.0)),
        "stage3_scalar_fallback_seconds": _as_float(profile.get("stage3_scalar_fallback_seconds", 0.0)),
        "stage3_non_encoder_overhead_seconds": _as_float(profile.get("stage3_non_encoder_overhead_seconds", 0.0)),
        "stage7_inference_seconds": _as_float(profile.get("stage7_model_inference_seconds", 0.0)),
        "support_inference_seconds": _as_float(profile.get("support_model_inference_seconds", 0.0)),
        "gate_self_inference_seconds": _as_float(profile.get("gate_self_model_inference_seconds", 0.0)),
        "gate_probability_seconds": _as_float(profile.get("gate_probability_seconds", 0.0)),
        "t3_action_generation_seconds": _as_float(profile.get("t3_action_generation_seconds", 0.0)),
        "final_turn_decision_seconds": _as_float(profile.get("final_turn_decision_seconds", 0.0)),
        "final_turn_exact_enumeration_seconds": _as_float(profile.get("final_turn_exact_enumeration_time", 0.0)),
        "final_turn_exact_scoring_seconds": _as_float(profile.get("final_turn_exact_scoring_time", 0.0)),
        "final_turn_policy_decision_seconds": _as_float(profile.get("final_turn_policy_decision_time", 0.0)),
        "final_turn_legal_action_generation_seconds": _as_float(profile.get("final_turn_legal_action_generation_seconds", 0.0)),
        "final_turn_hand_eval_seconds": _as_float(profile.get("final_turn_hand_eval_seconds", 0.0)),
        "final_turn_royalty_scoring_seconds": _as_float(profile.get("final_turn_royalty_scoring_seconds", 0.0)),
        "final_turn_foul_check_seconds": _as_float(profile.get("final_turn_foul_check_seconds", 0.0)),
        "final_turn_cache_lookup_seconds": _as_float(profile.get("final_turn_cache_lookup_seconds", 0.0)),
        "final_turn_cache_update_seconds": _as_float(profile.get("final_turn_cache_update_seconds", 0.0)),
        "final_turn_other_seconds": _as_float(profile.get("final_turn_other_seconds", 0.0)),
        "final_turn_state_count": _as_float(profile.get("final_turn_state_count", 0.0)),
        "final_turn_unique_state_count": _as_float(profile.get("final_turn_unique_state_count", 0.0)),
        "final_turn_cache_hit": _as_float(profile.get("final_turn_cache_hit", 0.0)),
        "final_turn_cache_miss": _as_float(profile.get("final_turn_cache_miss", 0.0)),
        "final_turn_cache_hit_rate": _as_float(profile.get("final_turn_cache_hit_rate", 0.0)),
        "final_turn_unique_raw_ratio": _as_float(profile.get("final_turn_unique_raw_ratio", 0.0)),
        "final_turn_cache_memory_estimate": _as_float(profile.get("final_turn_cache_memory_estimate", 0.0)),
        "final_turn_hero_state_count": _as_float(profile.get("final_turn_hero_state_count", 0.0)),
        "final_turn_opponent_state_count": _as_float(profile.get("final_turn_opponent_state_count", 0.0)),
        "final_turn_first_position_state_count": _as_float(profile.get("final_turn_first_position_state_count", 0.0)),
        "final_turn_second_position_state_count": _as_float(profile.get("final_turn_second_position_state_count", 0.0)),
        "non_t3_continuation_seconds": _as_float(profile.get("non_t3_continuation_decision_seconds", 0.0)),
        "memory_peak_mb": _as_float(profile.get("memory_peak_mb", 0.0)),
        "stage7_model_called": _as_float(profile.get("stage7_model_called_count", 0.0)),
        "support_model_called": _as_float(profile.get("support_model_called_count", 0.0)),
        "gate_model_called": _as_float(profile.get("gate_model_called_count", 0.0)),
        "skipped_by_reference_margin": _as_float(profile.get("skipped_by_reference_margin", 0.0)),
        "fallback_recompute": _as_float(profile.get("stage3_fallback_recomputed_count", 0.0)),
        "fallback_reuse": _as_float(profile.get("stage3_fallback_reuse_count", 0.0)),
        "stage3_feature_rows": _as_float(profile.get("stage3_feature_rows", 0.0)),
        "feature_columns": _as_float(profile.get("feature_column_count", 0.0)),
        "direct_column_count": _as_float(profile.get("direct_column_count", 0.0)),
        "scalar_fallback_column_count": _as_float(profile.get("scalar_fallback_column_count", 0.0)),
        "direct_column_coverage_ratio": _as_float(profile.get("direct_column_coverage_ratio", 0.0)),
    }


def _final_turn_profile_csv_row(index: int, profile: dict[str, Any]) -> dict[str, Any]:
    final_count = _as_float(profile.get("final_turn_state_count", 0.0))
    final_hit = _as_float(profile.get("final_turn_cache_hit", 0.0))
    final_miss = _as_float(profile.get("final_turn_cache_miss", 0.0))
    return {
        "state_index": index,
        "final_turn_state_count": final_count,
        "final_turn_unique_state_count": _as_float(profile.get("final_turn_unique_state_count", final_miss)),
        "final_turn_decision_seconds": _as_float(profile.get("final_turn_decision_seconds", 0.0)),
        "final_turn_exact_enumeration_seconds": _as_float(profile.get("final_turn_exact_enumeration_time", 0.0)),
        "final_turn_exact_scoring_seconds": _as_float(profile.get("final_turn_exact_scoring_time", 0.0)),
        "final_turn_policy_decision_seconds": _as_float(profile.get("final_turn_policy_decision_time", 0.0)),
        "final_turn_legal_action_generation_seconds": _as_float(profile.get("final_turn_legal_action_generation_seconds", 0.0)),
        "final_turn_hand_eval_seconds": _as_float(profile.get("final_turn_hand_eval_seconds", 0.0)),
        "final_turn_royalty_scoring_seconds": _as_float(profile.get("final_turn_royalty_scoring_seconds", 0.0)),
        "final_turn_foul_check_seconds": _as_float(profile.get("final_turn_foul_check_seconds", 0.0)),
        "final_turn_cache_lookup_seconds": _as_float(profile.get("final_turn_cache_lookup_seconds", 0.0)),
        "final_turn_cache_update_seconds": _as_float(profile.get("final_turn_cache_update_seconds", 0.0)),
        "final_turn_other_seconds": _as_float(profile.get("final_turn_other_seconds", 0.0)),
        "final_turn_cache_hit": final_hit,
        "final_turn_cache_miss": final_miss,
        "final_turn_cache_hit_rate": final_hit / (final_hit + final_miss) if final_hit + final_miss else 0.0,
        "final_turn_unique_raw_ratio": final_miss / final_count if final_count else 0.0,
        "final_turn_cache_memory_estimate": _as_float(profile.get("final_turn_cache_memory_estimate", 0.0)),
        "final_turn_hero_state_count": _as_float(profile.get("final_turn_hero_state_count", 0.0)),
        "final_turn_opponent_state_count": _as_float(profile.get("final_turn_opponent_state_count", 0.0)),
        "final_turn_first_position_state_count": _as_float(profile.get("final_turn_first_position_state_count", 0.0)),
        "final_turn_second_position_state_count": _as_float(profile.get("final_turn_second_position_state_count", 0.0)),
    }


def _distribution_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(values)
    return {
        "mean": float(sum(ordered) / len(ordered)),
        "median": _percentile(ordered, 50.0),
        "p90": _percentile(ordered, 90.0),
        "p95": _percentile(ordered, 95.0),
        "max": float(ordered[-1]),
    }


def _percentile(ordered: list[float], percentile: float) -> float:
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))


def _put_rate(totals: dict[str, Any], output_key: str, hit_key: str, miss_key: str) -> None:
    hits = _as_float(totals.get(hit_key, 0.0))
    misses = _as_float(totals.get(miss_key, 0.0))
    totals[output_key] = hits / (hits + misses) if hits + misses else 0.0


def _put_feature_attribution_shares(totals: dict[str, Any]) -> None:
    denominator = _as_float(totals.get("stage3_feature_generation_time", 0.0))
    share_keys = {
        "encoder_matrix_build_share": "stage3_encoder_matrix_build_seconds",
        "after_board_share": "stage3_after_board_construction_seconds",
        "row_summary_share": "stage3_row_summary_seconds",
        "global_summary_share": "stage3_global_summary_seconds",
        "action_delta_share": "stage3_action_delta_seconds",
        "scalar_fallback_share": "stage3_scalar_fallback_seconds",
        "cache_lookup_update_share": "stage3_cache_lookup_update_seconds",
        "column_validation_share": "stage3_column_validation_seconds",
        "numpy_allocation_share": "stage3_numpy_allocation_seconds",
        "hgb_input_preparation_share": "stage3_hgb_input_preparation_seconds",
        "non_encoder_overhead_share": "stage3_non_encoder_overhead_seconds",
    }
    for output_key, source_key in share_keys.items():
        totals[output_key] = _as_float(totals.get(source_key, 0.0)) / denominator if denominator else 0.0


def _as_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _build_stage7_continuation_policy(
    bundle: object,
    *,
    seed: int,
    seat: str,
    opening_lookahead_samples: int,
) -> RegularAiPolicy:
    return RegularAiPolicy(
        opening_model=bundle.opening,
        turn1_model=bundle.turn1,
        turn2_model=bundle.turn2,
        turn3_model=bundle.turn3,
        hu_turn3_model=getattr(bundle, "hu_turn3_stage7", None),
        hu_turn3_reference_model=getattr(bundle, "hu_turn3_stage7_reference", None),
        hu_turn3_min_margin=DEFAULT_HU_TURN3_STAGE7_MIN_MARGIN,
        hu_turn3_reference_min_margin=DEFAULT_HU_TURN3_STAGE7_REFERENCE_MIN_MARGIN,
        seed=seed,
        seat=seat,
        opening_lookahead_samples=opening_lookahead_samples,
    )


def _build_stage9d_continuation_policy(
    bundle: object,
    *,
    seed: int,
    seat: str,
    opening_lookahead_samples: int,
) -> RegularAiPolicy:
    return RegularAiPolicy(
        opening_model=bundle.opening,
        turn1_model=bundle.turn1,
        turn2_model=bundle.turn2,
        turn3_model=bundle.turn3,
        hu_turn3_model=getattr(bundle, "hu_turn3_stage9d", None),
        hu_turn3_reference_model=getattr(bundle, "hu_turn3_stage7_reference", None),
        hu_turn3_support_model=getattr(bundle, "hu_turn3_stage9d_support", None),
        hu_turn3_gate_model=getattr(bundle, "hu_turn3_stage9d_gate", None),
        hu_turn3_min_margin=DEFAULT_HU_TURN3_STAGE9D_MIN_MARGIN,
        hu_turn3_reference_min_margin=DEFAULT_HU_TURN3_STAGE9D_REFERENCE_MIN_MARGIN,
        hu_turn3_min_support_margin=DEFAULT_HU_TURN3_STAGE9D_MIN_SUPPORT_MARGIN,
        hu_turn3_min_model_score=DEFAULT_HU_TURN3_STAGE9D_MIN_MODEL_SCORE,
        hu_turn3_min_gate_probability=DEFAULT_HU_TURN3_STAGE9D_MIN_GATE_PROBABILITY,
        seed=seed,
        seat=seat,
        opening_lookahead_samples=opening_lookahead_samples,
    )


def _build_stage3_reference_continuation_policy(
    bundle: object,
    *,
    seed: int,
    seat: str,
    opening_lookahead_samples: int,
) -> RegularAiPolicy:
    return RegularAiPolicy(
        opening_model=bundle.opening,
        turn1_model=bundle.turn1,
        turn2_model=bundle.turn2,
        turn3_model=bundle.turn3,
        hu_turn3_model=None,
        hu_turn3_reference_model=getattr(bundle, "hu_turn3_stage7_reference", None),
        hu_turn3_min_margin=0.0,
        hu_turn3_reference_min_margin=0.0,
        hu_turn3_stage7_enabled=False,
        seed=seed,
        seat=seat,
        opening_lookahead_samples=opening_lookahead_samples,
    )


def _build_t3_continuation_policy(
    mode: T3ContinuationMode,
    bundle: object,
    *,
    seed: int,
    seat: str,
    opening_lookahead_samples: int,
    ) -> RegularAiPolicy:
    if mode == "stage7_m5_r10":
        return _build_stage7_continuation_policy(
            bundle,
            seed=seed,
            seat=seat,
            opening_lookahead_samples=opening_lookahead_samples,
        )
    if mode == "stage9d_p07_relaxed_both":
        return _build_stage9d_continuation_policy(
            bundle,
            seed=seed,
            seat=seat,
            opening_lookahead_samples=opening_lookahead_samples,
        )
    if mode == "stage3_reference_default":
        return _build_stage3_reference_continuation_policy(
            bundle,
            seed=seed,
            seat=seat,
            opening_lookahead_samples=opening_lookahead_samples,
        )
    raise ValueError(f"unknown T3 continuation mode: {mode}")


def _t3_continuation_policy_name(mode: T3ContinuationMode) -> str:
    if mode == "stage7_m5_r10":
        return STAGE7_CONTINUATION_NAME
    if mode == "stage9d_p07_relaxed_both":
        return STAGE9D_CONTINUATION_NAME
    if mode == "stage3_reference_default":
        return STAGE3_REFERENCE_CONTINUATION_NAME
    raise ValueError(f"unknown T3 continuation mode: {mode}")


def _t3_continuation_model_path(mode: T3ContinuationMode) -> str:
    if mode == "stage7_m5_r10":
        return str(DEFAULT_HU_TURN3_STAGE7_MODEL)
    if mode == "stage9d_p07_relaxed_both":
        return str(DEFAULT_HU_TURN3_STAGE9D_MODEL)
    if mode == "stage3_reference_default":
        return ""
    raise ValueError(f"unknown T3 continuation mode: {mode}")


def _t3_continuation_min_margin(mode: T3ContinuationMode) -> float:
    if mode == "stage7_m5_r10":
        return DEFAULT_HU_TURN3_STAGE7_MIN_MARGIN
    if mode == "stage9d_p07_relaxed_both":
        return DEFAULT_HU_TURN3_STAGE9D_MIN_MARGIN
    if mode == "stage3_reference_default":
        return 0.0
    raise ValueError(f"unknown T3 continuation mode: {mode}")


def _t3_continuation_reference_min_margin(mode: T3ContinuationMode) -> float:
    if mode == "stage7_m5_r10":
        return DEFAULT_HU_TURN3_STAGE7_REFERENCE_MIN_MARGIN
    if mode == "stage9d_p07_relaxed_both":
        return DEFAULT_HU_TURN3_STAGE9D_REFERENCE_MIN_MARGIN
    if mode == "stage3_reference_default":
        return 0.0
    raise ValueError(f"unknown T3 continuation mode: {mode}")


def _t3_continuation_min_support_margin(mode: T3ContinuationMode) -> float:
    if mode == "stage9d_p07_relaxed_both":
        return DEFAULT_HU_TURN3_STAGE9D_MIN_SUPPORT_MARGIN
    if mode in {"stage3_reference_default", "stage7_m5_r10"}:
        return 0.0
    raise ValueError(f"unknown T3 continuation mode: {mode}")


def _t3_continuation_min_model_score(mode: T3ContinuationMode) -> float | None:
    if mode == "stage9d_p07_relaxed_both":
        return DEFAULT_HU_TURN3_STAGE9D_MIN_MODEL_SCORE
    if mode in {"stage3_reference_default", "stage7_m5_r10"}:
        return None
    raise ValueError(f"unknown T3 continuation mode: {mode}")


def _t3_continuation_min_gate_probability(mode: T3ContinuationMode) -> float:
    if mode == "stage9d_p07_relaxed_both":
        return DEFAULT_HU_TURN3_STAGE9D_MIN_GATE_PROBABILITY
    if mode in {"stage3_reference_default", "stage7_m5_r10"}:
        return 0.0
    raise ValueError(f"unknown T3 continuation mode: {mode}")


def _t3_continuation_metadata(mode: T3ContinuationMode) -> dict[str, Any]:
    if mode == "stage7_m5_r10":
        return {
            "policy": STAGE7_CONTINUATION_NAME,
            "mode": mode,
            "stage7_enabled": True,
            "stage7_model_path": str(DEFAULT_HU_TURN3_STAGE7_MODEL),
            "stage7_hu_turn3_min_margin": DEFAULT_HU_TURN3_STAGE7_MIN_MARGIN,
            "stage7_hu_turn3_reference_min_margin": DEFAULT_HU_TURN3_STAGE7_REFERENCE_MIN_MARGIN,
            "stage3_reference_model_path": str(DEFAULT_HU_TURN3_STAGE7_REFERENCE_MODEL),
        }
    if mode == "stage9d_p07_relaxed_both":
        return {
            "policy": STAGE9D_CONTINUATION_NAME,
            "mode": mode,
            "stage7_enabled": True,
            "stage7_model_path": str(DEFAULT_HU_TURN3_STAGE9D_MODEL),
            "stage7_hu_turn3_min_margin": DEFAULT_HU_TURN3_STAGE9D_MIN_MARGIN,
            "stage7_hu_turn3_reference_min_margin": DEFAULT_HU_TURN3_STAGE9D_REFERENCE_MIN_MARGIN,
            "stage3_reference_model_path": str(DEFAULT_HU_TURN3_STAGE7_REFERENCE_MODEL),
            "support_model_path": str(DEFAULT_HU_TURN3_STAGE9D_SUPPORT_MODEL),
            "gate_model_path": str(DEFAULT_HU_TURN3_STAGE9D_GATE_MODEL),
            "hu_turn3_min_support_margin": DEFAULT_HU_TURN3_STAGE9D_MIN_SUPPORT_MARGIN,
            "hu_turn3_min_model_score": DEFAULT_HU_TURN3_STAGE9D_MIN_MODEL_SCORE,
            "hu_turn3_min_gate_probability": DEFAULT_HU_TURN3_STAGE9D_MIN_GATE_PROBABILITY,
        }
    if mode == "stage3_reference_default":
        return {
            "policy": STAGE3_REFERENCE_CONTINUATION_NAME,
            "mode": mode,
            "stage7_enabled": False,
            "stage7_model_path": "",
            "stage7_hu_turn3_min_margin": 0.0,
            "stage7_hu_turn3_reference_min_margin": 0.0,
            "stage3_reference_model_path": str(DEFAULT_HU_TURN3_STAGE7_REFERENCE_MODEL),
            "note": "Hidden-discard default: use the HU Stage3 reference action directly; Stage7 m5_r10 must be opted in explicitly.",
        }
    raise ValueError(f"unknown T3 continuation mode: {mode}")


def _build_policy_for_profile(
    profile: str,
    bundle: object,
    *,
    seed: int,
    seat: str,
    opening_lookahead_samples: int,
) -> RegularAiPolicy:
    if profile == "random_exact_final":
        return RegularAiPolicy(seed=seed, seat=seat)
    return _build_stage9d_continuation_policy(
        bundle,
        seed=seed,
        seat=seat,
        opening_lookahead_samples=opening_lookahead_samples,
    )


@dataclass
class _ActionRolloutAggregate:
    scores: list[float] = field(default_factory=list)
    component_values: dict[str, list[float]] = field(default_factory=dict)
    t3_eval_count: int = 0
    t3_fired_count: int = 0
    no_override_reason_counts: Counter[str] = field(default_factory=Counter)


@dataclass
class _BatchedRollout:
    action_index: int
    hero: Board
    opponent: Board
    dead: list[str]
    hero_private_discards: list[str]
    opponent_private_discards: list[str]
    future_cards: list[str]
    future_index: int
    future_rollout_seed: int
    root_fingerprint: str
    cursor: int = 0
    valid: bool = True
    t3_eval_count: int = 0
    t3_fired_count: int = 0
    no_override_reason_counts: Counter[str] = field(default_factory=Counter)


@dataclass
class _PendingNonT3Decision:
    key: tuple[Any, ...]
    item: _BatchedRollout
    actor: str
    board: Board
    opponent_board: Board
    dealt: tuple[str, str, str]
    policy: RegularAiPolicy
    actions: list[Action]
    sample: dict[str, Any]
    model: object


def _item_visible_dead_cards(item: _BatchedRollout, actor: str) -> tuple[str, ...]:
    if actor == "hero":
        return (*item.opponent.all_cards(), *tuple(item.hero_private_discards))
    if actor == "opponent":
        return (*item.hero.all_cards(), *tuple(item.opponent_private_discards))
    raise ValueError(f"unknown actor: {actor}")


def _item_policy_decision_seed(
    item: _BatchedRollout, actor: str, card_count: int
) -> int:
    street = {5: "T1", 7: "T2", 9: "T3", 11: "T4"}.get(card_count, "T4")
    return policy_decision_seed(
        base_seed=item.future_rollout_seed,
        run_id="hu_turn2_teacher_rollout",
        root_fingerprint=item.root_fingerprint,
        future_index=item.future_index,
        actor=actor,  # type: ignore[arg-type]
        street=street,
        decision_ordinal=card_count * 2 + (0 if actor == "hero" else 1),
    )


def _rollouts_after_hero_t2_actions_batched(
    *,
    board: Board,
    actions: list[Action],
    action_indices: Iterable[int] | None = None,
    opponent_board: Board,
    dead_cards: Iterable[str],
    hero_private_discards: Iterable[str],
    opponent_private_discards: Iterable[str],
    hero_seat: str,
    future_rollouts: list[tuple[str, ...]],
    future_rollout_seed: int,
    root_fingerprint: str,
    hero_policy: RegularAiPolicy,
    opponent_policy: RegularAiPolicy,
    profile: dict[str, float],
    config: HuTurn3Stage7BatchConfig,
    cache: HuTurn3DecisionCache | None,
    reference_cache: HuTurn3Stage3ReferenceCache | None,
    state_feature_cache: Stage3StateFeatureCache | None,
    action_cache: HuTurn3ActionCache | None,
    batch_size: int,
    final_turn_cache: FinalTurnDecisionCache | None,
    use_final_turn_cache: bool,
    t4_search_config: T4SearchConfig | None = None,
    opponent_private_discards_by_rollout: Iterable[Iterable[str]] | None = None,
) -> dict[int, _ActionRolloutAggregate]:
    rollout_started_at = time.perf_counter()
    if action_indices is None:
        selected_indices = list(range(len(actions)))
    else:
        selected_indices = sorted({int(index) for index in action_indices if 0 <= int(index) < len(actions)})
    aggregates = {index: _ActionRolloutAggregate() for index in selected_indices}
    items: list[_BatchedRollout] = []
    for action_index in selected_indices:
        action = actions[action_index]
        hero_after_action = board.place(action.placements)
        initial_dead = [*tuple(dead_cards), *action.discards]
        initial_hero_private_discards = [*tuple(hero_private_discards), *action.discards]
        rollout_opponent_discards = (
            [tuple(cards) for cards in opponent_private_discards_by_rollout]
            if opponent_private_discards_by_rollout is not None
            else [tuple(opponent_private_discards)] * len(future_rollouts)
        )
        if len(rollout_opponent_discards) != len(future_rollouts):
            raise ValueError("opponent discard roots must align with future rollouts")
        for future_index, (future_cards, opponent_discards) in enumerate(
            zip(future_rollouts, rollout_opponent_discards)
        ):
            items.append(
                _BatchedRollout(
                    action_index=action_index,
                    hero=hero_after_action,
                    opponent=opponent_board,
                    dead=list(initial_dead),
                    hero_private_discards=list(initial_hero_private_discards),
                    opponent_private_discards=list(opponent_discards),
                    future_cards=list(future_cards),
                    future_index=future_index,
                    future_rollout_seed=future_rollout_seed,
                    root_fingerprint=root_fingerprint,
                )
            )

    _advance_items_to_turn3_pair_batched(
        items,
        hero_seat=hero_seat,
        hero_policy=hero_policy,
        opponent_policy=opponent_policy,
        profile=profile,
        batch_size=batch_size,
    )

    _apply_t3_batch_wave(
        items,
        hero_seat=hero_seat,
        hero_policy=hero_policy,
        opponent_policy=opponent_policy,
        profile=profile,
        config=config,
        cache=cache,
        reference_cache=reference_cache,
        state_feature_cache=state_feature_cache,
        action_cache=action_cache,
        batch_size=batch_size,
    )
    _apply_t3_batch_wave(
        items,
        hero_seat=hero_seat,
        hero_policy=hero_policy,
        opponent_policy=opponent_policy,
        profile=profile,
        config=config,
        cache=cache,
        reference_cache=reference_cache,
        state_feature_cache=state_feature_cache,
        action_cache=action_cache,
        batch_size=batch_size,
    )

    for item in items:
        if not item.valid:
            continue
        rollout_result = _finish_rollout_after_t3(
            item,
            hero_seat=hero_seat,
            hero_policy=hero_policy,
            opponent_policy=opponent_policy,
            profile=profile,
            final_turn_cache=final_turn_cache,
            use_final_turn_cache=use_final_turn_cache,
            t4_search_config=t4_search_config,
        )
        if rollout_result is None:
            continue
        score, components = rollout_result
        aggregate = aggregates[item.action_index]
        aggregate.scores.append(score)
        for key, value in components.items():
            aggregate.component_values.setdefault(key, []).append(float(value))
        aggregate.t3_eval_count += item.t3_eval_count
        aggregate.t3_fired_count += item.t3_fired_count
        aggregate.no_override_reason_counts.update(item.no_override_reason_counts)
    profile["rollout_seconds"] += time.perf_counter() - rollout_started_at
    return aggregates


def _advance_items_to_turn3_pair_batched(
    items: list[_BatchedRollout],
    *,
    hero_seat: str,
    hero_policy: RegularAiPolicy,
    opponent_policy: RegularAiPolicy,
    profile: dict[str, float],
    batch_size: int,
) -> None:
    decision_cache: dict[tuple[Any, ...], Action] = {}
    while True:
        pending: list[_PendingNonT3Decision] = []
        pending_by_key: dict[tuple[Any, ...], int] = {}
        duplicate_waiters: dict[tuple[Any, ...], list[tuple[_BatchedRollout, str, int]]] = {}
        scalar_items: list[tuple[_BatchedRollout, str, tuple[str, str, str]]] = []
        legal_started_at = time.perf_counter()
        for item in items:
            if not item.valid:
                continue
            if item.hero.card_count() >= 9 and item.opponent.card_count() >= 9:
                continue
            actor = _next_actor(item.hero, item.opponent, hero_seat)
            dealt = _draw3_from_item(item)
            if dealt is None:
                item.valid = False
                continue
            board = item.hero if actor == "hero" else item.opponent
            opponent_board = item.opponent if actor == "hero" else item.hero
            policy = hero_policy if actor == "hero" else opponent_policy
            model = _non_t3_model_for_policy(policy, board.card_count())
            if model is None:
                scalar_items.append((item, actor, dealt))
                continue
            key = _non_t3_decision_key(model, board, dealt)
            profile["raw_non_t3_decisions"] += 1.0
            cached_action = decision_cache.get(key)
            if cached_action is not None:
                profile["non_t3_decision_cache_hit"] += 1.0
                _add_non_t3_actor_turn_only(
                    profile,
                    actor=actor,
                    card_count=board.card_count(),
                    elapsed=0.0,
                )
                _apply_non_t3_action(item, actor, cached_action)
                continue
            if key in pending_by_key:
                profile["non_t3_decision_cache_hit"] += 1.0
                duplicate_waiters.setdefault(key, []).append((item, actor, board.card_count()))
                continue
            profile["non_t3_decision_cache_miss"] += 1.0
            try:
                actions = generate_turn_actions(board, dealt)
                sample = policy_sample(board, dealt, actions)
            except Exception:
                scalar_items.append((item, actor, dealt))
                continue
            if not actions:
                item.valid = False
                continue
            pending_by_key[key] = len(pending)
            pending.append(
                _PendingNonT3Decision(
                    key=key,
                    item=item,
                    actor=actor,
                    board=board,
                    opponent_board=opponent_board,
                    dealt=dealt,
                    policy=policy,
                    actions=actions,
                    sample=sample,
                    model=model,
                )
            )
        legal_elapsed = time.perf_counter() - legal_started_at
        if not pending and not scalar_items:
            return
        profile["non_t3_legal_action_time"] += legal_elapsed

        batch_started_at = time.perf_counter()
        _apply_pending_non_t3_decisions_batched(
            pending,
            profile=profile,
            batch_size=batch_size,
            decision_cache=decision_cache,
        )
        for key, waiters in duplicate_waiters.items():
            action = decision_cache.get(key)
            if action is None:
                for item, actor, _card_count in waiters:
                    item.valid = False
                continue
            for item, actor, _card_count in waiters:
                _apply_non_t3_action(item, actor, action)
        batch_elapsed = time.perf_counter() - batch_started_at
        scalar_elapsed = _apply_pending_non_t3_decisions_scalar(
            scalar_items,
            hero_policy=hero_policy,
            opponent_policy=opponent_policy,
            profile=profile,
        )
        total_elapsed = legal_elapsed + batch_elapsed + scalar_elapsed
        profile["non_t3_continuation_decision_seconds"] += total_elapsed
        _add_non_t3_batch_actor_turn_profile(
            profile,
            pending=pending,
            scalar_items=scalar_items,
            duplicate_waiters=duplicate_waiters,
            elapsed=total_elapsed,
        )


def _apply_pending_non_t3_decisions_batched(
    pending: list[_PendingNonT3Decision],
    *,
    profile: dict[str, float],
    batch_size: int,
    decision_cache: dict[tuple[Any, ...], Action],
) -> None:
    if not pending:
        return
    feature_started_at = time.perf_counter()
    grouped: dict[int, list[int]] = {}
    for index, decision in enumerate(pending):
        grouped.setdefault(id(decision.model), []).append(index)
    feature_elapsed = 0.0
    inference_elapsed = 0.0
    for indices in grouped.values():
        outputs = _predict_non_t3_group_batched(
            [pending[index] for index in indices],
            batch_size=batch_size,
        )
        feature_elapsed += outputs["feature_seconds"]
        inference_elapsed += outputs["inference_seconds"]
        for local_index, prediction in enumerate(outputs["predictions"]):
            decision = pending[indices[local_index]]
            action_index = _safe_prediction_index(prediction, decision.actions)
            if action_index is None:
                action = _apply_non_t3_scalar_fallback(decision, profile)
                if action is not None:
                    decision_cache[decision.key] = action
                continue
            action = decision.actions[action_index]
            decision_cache[decision.key] = action
            _apply_non_t3_action(decision.item, decision.actor, action)
    profile["non_t3_feature_generation_time"] += feature_elapsed
    profile["non_t3_model_inference_time"] += inference_elapsed
    profile["non_t3_other_time"] += max(0.0, time.perf_counter() - feature_started_at - feature_elapsed - inference_elapsed)


def _predict_non_t3_group_batched(
    pending: list[_PendingNonT3Decision],
    *,
    batch_size: int,
) -> dict[str, Any]:
    if not pending:
        return {"predictions": [], "feature_seconds": 0.0, "inference_seconds": 0.0}
    model = pending[0].model
    if not hasattr(model, "predict_matrix"):
        inference_started_at = time.perf_counter()
        predictions = [_predict_non_t3_single(model, decision.sample) for decision in pending]
        return {
            "predictions": predictions,
            "feature_seconds": 0.0,
            "inference_seconds": time.perf_counter() - inference_started_at,
        }

    predictions: list[np.ndarray | None] = [None] * len(pending)
    feature_blocks: list[np.ndarray] = []
    output_indices: list[int] = []
    block_sizes: list[int] = []
    row_count = 0
    feature_seconds = 0.0
    inference_seconds = 0.0

    def flush() -> None:
        nonlocal feature_blocks, output_indices, block_sizes, row_count, inference_seconds
        if not feature_blocks:
            return
        inference_started_at = time.perf_counter()
        try:
            raw = np.asarray(model.predict_matrix(np.vstack(feature_blocks)), dtype=np.float64)
        except Exception:
            inference_seconds += time.perf_counter() - inference_started_at
            for output_index in output_indices:
                predictions[output_index] = _predict_non_t3_single(
                    model,
                    pending[output_index].sample,
                )
            feature_blocks = []
            output_indices = []
            block_sizes = []
            row_count = 0
            return
        inference_seconds += time.perf_counter() - inference_started_at
        cursor = 0
        for output_index, block_size in zip(output_indices, block_sizes):
            predictions[output_index] = raw[cursor : cursor + block_size]
            cursor += block_size
        feature_blocks = []
        output_indices = []
        block_sizes = []
        row_count = 0

    for index, decision in enumerate(pending):
        feature_started_at = time.perf_counter()
        features, _targets = self_sample_to_matrix(decision.sample)
        feature_seconds += time.perf_counter() - feature_started_at
        if row_count and row_count + features.shape[0] > batch_size:
            flush()
        feature_blocks.append(features)
        output_indices.append(index)
        block_sizes.append(features.shape[0])
        row_count += int(features.shape[0])
        if row_count >= batch_size:
            flush()
    flush()
    return {
        "predictions": predictions,
        "feature_seconds": feature_seconds,
        "inference_seconds": inference_seconds,
    }


def _apply_pending_non_t3_decisions_scalar(
    scalar_items: list[tuple[_BatchedRollout, str, tuple[str, str, str]]],
    *,
    hero_policy: RegularAiPolicy,
    opponent_policy: RegularAiPolicy,
    profile: dict[str, float],
) -> float:
    started_at = time.perf_counter()
    for item, actor, dealt in scalar_items:
        if not item.valid:
            continue
        policy = hero_policy if actor == "hero" else opponent_policy
        board = item.hero if actor == "hero" else item.opponent
        opponent = item.opponent if actor == "hero" else item.hero
        action = policy.choose_action(
            board,
            dealt,
            dead_cards=_item_visible_dead_cards(item, actor),
            opponent_board=opponent,
            decision_seed=_item_policy_decision_seed(item, actor, board.card_count()),
            street={5: "T1", 7: "T2", 9: "T3"}.get(board.card_count()),
        )
        _apply_non_t3_action(item, actor, action)
    elapsed = time.perf_counter() - started_at
    profile["non_t3_fallback_time"] += elapsed
    return elapsed


def _apply_non_t3_scalar_fallback(
    decision: _PendingNonT3Decision,
    profile: dict[str, float],
) -> Action | None:
    started_at = time.perf_counter()
    try:
        action = decision.policy.choose_action(
            decision.board,
            decision.dealt,
            dead_cards=_item_visible_dead_cards(decision.item, decision.actor),
            opponent_board=decision.opponent_board,
            decision_seed=_item_policy_decision_seed(
                decision.item, decision.actor, decision.board.card_count()
            ),
            street={5: "T1", 7: "T2", 9: "T3"}.get(
                decision.board.card_count()
            ),
        )
    except Exception:
        decision.item.valid = False
        profile["non_t3_fallback_time"] += time.perf_counter() - started_at
        return None
    profile["non_t3_fallback_time"] += time.perf_counter() - started_at
    _apply_non_t3_action(decision.item, decision.actor, action)
    return action


def _apply_non_t3_action(item: _BatchedRollout, actor: str, action: Action) -> None:
    if actor == "hero":
        item.hero = item.hero.place(action.placements)
        item.hero_private_discards.extend(action.discards)
    else:
        item.opponent = item.opponent.place(action.placements)
        item.opponent_private_discards.extend(action.discards)
    item.dead.extend(action.discards)


def _non_t3_model_for_policy(policy: RegularAiPolicy, card_count: int) -> object | None:
    if card_count == 5:
        return policy.turn1_model
    if card_count == 7:
        return policy.turn2_model
    if card_count == 9:
        return policy.turn3_model
    return None


def _non_t3_decision_key(
    model: object,
    board: Board,
    dealt: tuple[str, str, str],
) -> tuple[Any, ...]:
    return (
        id(model),
        board.card_count(),
        tuple(sorted(board.top, key=_card_sort_index)),
        tuple(sorted(board.middle, key=_card_sort_index)),
        tuple(sorted(board.bottom, key=_card_sort_index)),
        tuple(sorted(dealt, key=_card_sort_index)),
    )


def _card_sort_index(card: str) -> int:
    try:
        return ALL_CARDS.index(card)
    except ValueError:
        return 999


def _predict_non_t3_single(model: object, sample: dict[str, Any]) -> np.ndarray | None:
    if not hasattr(model, "predict_sample"):
        return None
    try:
        return np.asarray(model.predict_sample(sample), dtype=np.float64)
    except Exception:
        return None


def _safe_prediction_index(
    prediction: np.ndarray | None, actions: list[Action]
) -> int | None:
    if prediction is None or len(prediction) != len(actions):
        return None
    values = [float(value) for value in prediction]
    if not values or any(not math.isfinite(value) for value in values):
        return None
    return canonical_argmax_index(values, actions)


def _add_non_t3_batch_actor_turn_profile(
    profile: dict[str, float],
    *,
    pending: list[_PendingNonT3Decision],
    scalar_items: list[tuple[_BatchedRollout, str, tuple[str, str, str]]],
    duplicate_waiters: dict[tuple[Any, ...], list[tuple[_BatchedRollout, str, int]]],
    elapsed: float,
) -> None:
    duplicate_count = sum(len(waiters) for waiters in duplicate_waiters.values())
    decision_count = len(pending) + len(scalar_items) + duplicate_count
    if decision_count <= 0:
        return
    per_decision = elapsed / decision_count
    profile["unique_non_t3_decisions"] += float(len(pending))
    for decision in pending:
        _add_non_t3_actor_turn_only(
            profile,
            actor=decision.actor,
            card_count=decision.board.card_count(),
            elapsed=per_decision,
        )
    for waiters in duplicate_waiters.values():
        for _item, actor, card_count in waiters:
            _add_non_t3_actor_turn_only(
                profile,
                actor=actor,
                card_count=card_count,
                elapsed=per_decision,
            )
    for item, actor, _dealt in scalar_items:
        card_count = item.hero.card_count() if actor == "hero" else item.opponent.card_count()
        _add_non_t3_actor_turn_only(
            profile,
            actor=actor,
            card_count=card_count,
            elapsed=per_decision,
        )


def _add_non_t3_actor_turn_only(
    profile: dict[str, float],
    *,
    actor: str,
    card_count: int,
    elapsed: float,
) -> None:
    if actor == "opponent":
        profile["opponent_policy_decision_time"] += elapsed
        profile["non_t3_opponent_decision_seconds"] += elapsed
    else:
        profile["hero_non_t3_policy_decision_time"] += elapsed
    profile[_turn_profile_key(card_count)] += elapsed


def _advance_to_turn3_pair(
    item: _BatchedRollout,
    *,
    hero_seat: str,
    hero_policy: RegularAiPolicy,
    opponent_policy: RegularAiPolicy,
    profile: dict[str, float],
) -> bool:
    while item.hero.card_count() < 9 or item.opponent.card_count() < 9:
        actor = _next_actor(item.hero, item.opponent, hero_seat)
        dealt = _draw3_from_item(item)
        if dealt is None:
            item.valid = False
            return False
        decision_started_at = time.perf_counter()
        actor_card_count = item.hero.card_count() if actor == "hero" else item.opponent.card_count()
        if actor == "hero":
            action = hero_policy.choose_action(
                item.hero,
                dealt,
                dead_cards=_item_visible_dead_cards(item, actor),
                opponent_board=item.opponent,
                decision_seed=_item_policy_decision_seed(
                    item, actor, actor_card_count
                ),
                street={5: "T1", 7: "T2", 9: "T3"}.get(actor_card_count),
            )
            item.hero = item.hero.place(action.placements)
            item.hero_private_discards.extend(action.discards)
        else:
            action = opponent_policy.choose_action(
                item.opponent,
                dealt,
                dead_cards=_item_visible_dead_cards(item, actor),
                opponent_board=item.hero,
                decision_seed=_item_policy_decision_seed(
                    item, actor, actor_card_count
                ),
                street={5: "T1", 7: "T2", 9: "T3"}.get(actor_card_count),
            )
            item.opponent = item.opponent.place(action.placements)
            item.opponent_private_discards.extend(action.discards)
        elapsed = time.perf_counter() - decision_started_at
        profile["non_t3_continuation_decision_seconds"] += elapsed
        _add_non_t3_decision_profile(
            profile,
            actor=actor,
            card_count=actor_card_count,
            elapsed=elapsed,
            hero_seat=hero_seat,
            include_model_bucket=True,
        )
        item.dead.extend(action.discards)
    return True


def _apply_t3_batch_wave(
    items: list[_BatchedRollout],
    *,
    hero_seat: str,
    hero_policy: RegularAiPolicy,
    opponent_policy: RegularAiPolicy,
    profile: dict[str, float],
    config: HuTurn3Stage7BatchConfig,
    cache: HuTurn3DecisionCache | None,
    reference_cache: HuTurn3Stage3ReferenceCache | None,
    state_feature_cache: Stage3StateFeatureCache | None,
    action_cache: HuTurn3ActionCache | None,
    batch_size: int,
) -> None:
    states_by_actor: dict[str, list[HuTurn3State]] = {"hero": [], "opponent": []}
    mapping_by_actor: dict[str, list[_BatchedRollout]] = {"hero": [], "opponent": []}
    for item in items:
        if not item.valid:
            continue
        actor = _next_actor(item.hero, item.opponent, hero_seat)
        if actor == "hero" and item.hero.card_count() != 9:
            continue
        if actor == "opponent" and item.opponent.card_count() != 9:
            continue
        if item.hero.card_count() >= 11 and item.opponent.card_count() >= 11:
            continue
        dealt = _draw3_from_item(item)
        if dealt is None:
            item.valid = False
            continue
        if actor == "hero":
            states_by_actor[actor].append(
                HuTurn3State(
                    board=item.hero,
                    dealt_cards=dealt,
                    opponent_board=item.opponent,
                    dead_cards=_item_visible_dead_cards(item, actor),
                    seat=hero_policy.seat,
                    to_act_order=_t3_to_act_order(item.hero, item.opponent),
                    decision_seed=_item_policy_decision_seed(item, actor, 9),
                )
            )
        else:
            states_by_actor[actor].append(
                HuTurn3State(
                    board=item.opponent,
                    dealt_cards=dealt,
                    opponent_board=item.hero,
                    dead_cards=_item_visible_dead_cards(item, actor),
                    seat=opponent_policy.seat,
                    to_act_order=_t3_to_act_order(item.opponent, item.hero),
                    decision_seed=_item_policy_decision_seed(item, actor, 9),
                )
            )
        mapping_by_actor[actor].append(item)
    if not any(states_by_actor.values()):
        return

    # A model bundle is a policy identity.  Mixing both actors in one call made
    # opponent continuations silently use the hero's T3 models.  Keep the wave
    # batched, but partition it at the policy boundary.
    for actor, actor_policy in (("hero", hero_policy), ("opponent", opponent_policy)):
        actor_states = states_by_actor[actor]
        if not actor_states:
            continue
        result = decide_hu_turn3_stage7_batch(
            actor_states,
            config,
            HuTurn3BatchModels(
                fallback_turn3_model=actor_policy.turn3_model,
                stage7_model=actor_policy.hu_turn3_model,
                stage3_reference_model=actor_policy.hu_turn3_reference_model,
                support_model=actor_policy.hu_turn3_support_model,
                gate_model=actor_policy.hu_turn3_gate_model,
            ),
            batch_size,
            cache=cache,
            reference_cache=reference_cache,
            state_feature_cache=state_feature_cache,
            action_cache=action_cache,
        )
        _merge_t3_batch_profile(profile, result.profile)
        for item, decision in zip(mapping_by_actor[actor], result.decisions):
            _apply_t3_decision(item, actor, decision)


def _apply_t3_decision(item: _BatchedRollout, actor: str, decision: Turn3Decision) -> None:
    item.t3_eval_count += 1
    if decision.override_fired:
        item.t3_fired_count += 1
    reason = decision.no_override_reason or "override_fired"
    item.no_override_reason_counts.update([reason])
    if decision.final_action is None:
        item.valid = False
        return
    if actor == "hero":
        item.hero = item.hero.place(decision.final_action.placements)
        item.hero_private_discards.extend(decision.final_action.discards)
    else:
        item.opponent = item.opponent.place(decision.final_action.placements)
        item.opponent_private_discards.extend(decision.final_action.discards)
    item.dead.extend(decision.final_action.discards)


def _choose_rollout_t4_action(
    *,
    actor_board: Board,
    opponent_board: Board,
    dealt_cards: tuple[str, str, str],
    actor_private_discards: Iterable[str],
    actor_dead_cards: Iterable[str],
    actor: str,
    actor_seat: str,
    hero_seat: str,
    profile: dict[str, Any] | None,
    final_turn_cache: FinalTurnDecisionCache | None,
    use_final_turn_cache: bool,
    t4_search_config: T4SearchConfig | None,
) -> Action | None:
    """Choose T4 without exposing a realized future deal to the first actor.

    Supplying ``t4_search_config`` opts the rollout into the M2 sequential
    selector.  Its first-to-act observation contains only public boards, the
    actor's current deal, and that actor's private discards.  The second actor
    still uses the existing exhaustive terminal solver.  ``None`` preserves
    legacy teacher artifacts until they are regenerated and promoted.
    """

    actor_order = _final_to_act_order(actor_board, opponent_board)
    if t4_search_config is not None and actor_order == "first":
        observation = ActorObservation(
            hero_board=actor_board,
            opponent_public_board=opponent_board,
            dealt_cards=dealt_cards,
            hero_private_discards=tuple(actor_private_discards),
            seat=actor_seat,  # type: ignore[arg-type]
            street="T4",
            to_act_order="first",
            scoring=ScoringContext(),
        )
        started_at = time.perf_counter()
        action = select_t4_action(
            observation,
            config=t4_search_config,
            fl_ev=DEFAULT_FL_EV,
        )
        elapsed = time.perf_counter() - started_at
        if profile is not None:
            profile["final_turn_decision_seconds"] += elapsed
            profile["final_turn_policy_decision_time"] += elapsed
            profile["final_turn_infoset_safe_decision_seconds"] = float(
                profile.get("final_turn_infoset_safe_decision_seconds", 0.0)
            ) + elapsed
            profile["final_turn_infoset_safe_state_count"] = float(
                profile.get("final_turn_infoset_safe_state_count", 0.0)
            ) + 1.0
            if actor == "hero":
                profile["final_turn_hero_state_count"] += 1.0
            else:
                profile["final_turn_opponent_state_count"] += 1.0
            if (hero_seat == "first" and actor == "hero") or (
                hero_seat == "second" and actor == "opponent"
            ):
                profile["final_turn_first_position_state_count"] += 1.0
            else:
                profile["final_turn_second_position_state_count"] += 1.0
            _add_non_t3_decision_profile(
                profile,
                actor=actor,
                card_count=11,
                elapsed=elapsed,
                hero_seat=hero_seat,
                include_model_bucket=False,
            )
        return action

    decision, final_profile, slow_record = decide_final_turn_exact(
        board=actor_board,
        dealt_cards=dealt_cards,
        dead_cards=actor_dead_cards,
        opponent_board=opponent_board,
        actor=actor,
        seat=actor_seat,
        to_act_order=actor_order,
        cache=final_turn_cache,
        use_cache=use_final_turn_cache,
    )
    if decision is None:
        return None
    if profile is not None:
        _merge_final_turn_profile(
            profile,
            final_profile,
            actor=actor,
            actor_card_count=11,
            hero_seat=hero_seat,
            slow_record=final_turn_slow_state_record(
                profile=final_profile,
                board=actor_board,
                opponent_board=opponent_board,
                dealt_cards=dealt_cards,
                dead_cards=actor_dead_cards,
                actor=actor,
                seat=actor_seat,
                to_act_order=actor_order,
                metadata=slow_record,
            ),
        )
    return decision.action


def _finish_rollout_after_t3(
    item: _BatchedRollout,
    *,
    hero_seat: str,
    hero_policy: RegularAiPolicy,
    opponent_policy: RegularAiPolicy,
    profile: dict[str, float],
    final_turn_cache: FinalTurnDecisionCache | None = None,
    use_final_turn_cache: bool = True,
    t4_search_config: T4SearchConfig | None = None,
) -> tuple[float, dict[str, float]] | None:
    while item.hero.card_count() < 13 or item.opponent.card_count() < 13:
        actor = _next_actor(item.hero, item.opponent, hero_seat)
        dealt = _draw3_from_item(item)
        if dealt is None:
            item.valid = False
            return None
        actor_card_count = item.hero.card_count() if actor == "hero" else item.opponent.card_count()
        if actor_card_count != 11:
            item.valid = False
            return None
        actor_board = item.hero if actor == "hero" else item.opponent
        actor_opponent = item.opponent if actor == "hero" else item.hero
        actor_policy = hero_policy if actor == "hero" else opponent_policy
        actor_dead = _item_visible_dead_cards(item, actor)
        actor_private = (
            item.hero_private_discards
            if actor == "hero"
            else item.opponent_private_discards
        )
        action = _choose_rollout_t4_action(
            actor_board=actor_board,
            opponent_board=actor_opponent,
            dealt_cards=dealt,
            actor_private_discards=actor_private,
            actor_dead_cards=actor_dead,
            actor=actor,
            actor_seat=actor_policy.seat,
            hero_seat=hero_seat,
            profile=profile,
            final_turn_cache=final_turn_cache,
            use_final_turn_cache=use_final_turn_cache,
            t4_search_config=t4_search_config,
        )
        if action is None:
            item.valid = False
            return None
        if actor == "hero":
            item.hero = item.hero.place(action.placements)
            item.hero_private_discards.extend(action.discards)
        else:
            item.opponent = item.opponent.place(action.placements)
            item.opponent_private_discards.extend(action.discards)
        profile["final_turn_state_count"] += 1.0
        item.dead.extend(action.discards)

    scoring_started_at = time.perf_counter()
    score, _board_score = terminal_score(item.hero, item.opponent, fl_ev=DEFAULT_FL_EV)
    profile["hand_evaluation_seconds"] += time.perf_counter() - scoring_started_at
    return float(score), _terminal_component_breakdown(item.hero, item.opponent)


def _final_to_act_order(board: Board, opponent_board: Board) -> str:
    return "second" if opponent_board.card_count() > board.card_count() else "first"


def _merge_final_turn_profile(
    profile: dict[str, Any],
    final_profile: Any,
    *,
    actor: str,
    actor_card_count: int,
    hero_seat: str,
    slow_record: dict[str, Any],
) -> None:
    elapsed = float(final_profile.seconds)
    profile["final_turn_decision_seconds"] += elapsed
    profile["final_turn_exact_enumeration_time"] += float(final_profile.exact_enumeration_seconds)
    profile["final_turn_exact_scoring_time"] += float(final_profile.exact_scoring_seconds)
    profile["final_turn_policy_decision_time"] += float(final_profile.policy_decision_seconds)
    profile["final_turn_legal_action_generation_seconds"] += float(final_profile.legal_action_generation_seconds)
    profile["final_turn_hand_eval_seconds"] += float(final_profile.hand_eval_seconds)
    profile["final_turn_royalty_scoring_seconds"] += float(final_profile.royalty_scoring_seconds)
    profile["final_turn_foul_check_seconds"] += float(final_profile.foul_check_seconds)
    profile["final_turn_cache_lookup_seconds"] += float(final_profile.cache_lookup_seconds)
    profile["final_turn_cache_update_seconds"] += float(final_profile.cache_update_seconds)
    profile["final_turn_other_seconds"] += float(final_profile.other_seconds)
    if final_profile.cache_hit:
        profile["final_turn_cache_hit"] += 1.0
    else:
        profile["final_turn_cache_miss"] += 1.0
    if actor == "hero":
        profile["final_turn_hero_state_count"] += 1.0
    else:
        profile["final_turn_opponent_state_count"] += 1.0
    if (hero_seat == "first" and actor == "hero") or (hero_seat == "second" and actor == "opponent"):
        profile["final_turn_first_position_state_count"] += 1.0
    else:
        profile["final_turn_second_position_state_count"] += 1.0
    _add_non_t3_decision_profile(
        profile,
        actor=actor,
        card_count=actor_card_count,
        elapsed=elapsed,
        hero_seat=hero_seat,
        include_model_bucket=False,
    )
    slow_states = profile.setdefault("_final_turn_slow_states", [])
    if isinstance(slow_states, list):
        if len(slow_states) < 100:
            slow_states.append(slow_record)
        else:
            min_index = min(range(len(slow_states)), key=lambda index: float(slow_states[index].get("seconds", 0.0)))
            if float(slow_record.get("seconds", 0.0)) > float(slow_states[min_index].get("seconds", 0.0)):
                slow_states[min_index] = slow_record


def _draw3_from_item(item: _BatchedRollout) -> tuple[str, str, str] | None:
    if item.cursor + 3 > len(item.future_cards):
        return None
    dealt = tuple(item.future_cards[item.cursor : item.cursor + 3])
    item.cursor += 3
    return dealt  # type: ignore[return-value]


def _t3_to_act_order(board: Board, opponent_board: Board) -> str:
    return "second" if opponent_board.card_count() > board.card_count() else "first"


def _add_non_t3_decision_profile(
    profile: dict[str, float],
    *,
    actor: str,
    card_count: int,
    elapsed: float,
    hero_seat: str,
    include_model_bucket: bool,
) -> None:
    del hero_seat
    if actor == "opponent":
        profile["opponent_policy_decision_time"] += elapsed
        profile["non_t3_opponent_decision_seconds"] += elapsed
    else:
        profile["hero_non_t3_policy_decision_time"] += elapsed
    profile[_turn_profile_key(card_count)] += elapsed
    if include_model_bucket:
        profile["non_t3_model_inference_time"] += elapsed
    else:
        profile["non_t3_fallback_time"] += elapsed


def _turn_profile_key(card_count: int) -> str:
    if card_count == 0:
        return "non_t3_turn_T0_seconds"
    if card_count == 5:
        return "non_t3_turn_T1_seconds"
    if card_count == 7:
        return "non_t3_turn_T2_seconds"
    if card_count == 9:
        return "non_t3_turn_T3_seconds"
    return "non_t3_turn_T4plus_seconds"


def _merge_t3_batch_profile(profile: dict[str, Any], batch_profile: dict[str, Any]) -> None:
    derived_keys = {
        "unique_raw_ratio",
        "ms_per_raw_t3_decision",
        "ms_per_raw_t3_continuation_decision",
        "ms_per_unique_t3_state",
        "peak_memory_mb",
        "stage3_feature_rows_per_second",
        "stage3_predict_rows_per_second",
        "stage3_reference_cache_hit_rate",
        "stage3_state_feature_cache_hit_rate",
        "t3_action_generation_cache_hit_rate",
        "row_summary_cache_hit_rate",
        "top_summary_cache_hit_rate",
        "complete_row_summary_cache_hit_rate",
        "feature_column_count",
        "direct_column_count",
        "scalar_fallback_column_count",
        "direct_column_coverage_ratio",
        "memory_peak_mb",
    }
    feature_metadata_keys = {
        "feature_dtype",
        "stage3_feature_mode",
        "stage3_feature_encoder_mode",
        "feature_schema_version",
    }
    constant_keys = {
        *feature_metadata_keys,
        "summary_aggregation_version",
    }
    for key, value in batch_profile.items():
        if key in derived_keys:
            continue
        if isinstance(value, (int, float)):
            current = profile.get(key, 0.0)
            if isinstance(current, (int, float)):
                profile[key] = current + float(value)
            elif current in (None, ""):
                profile[key] = float(value)
            elif key not in constant_keys:
                profile.setdefault(f"{key}_numeric_sum", float(value))
        elif key in feature_metadata_keys:
            if value not in (None, ""):
                profile[key] = value
        elif key in constant_keys:
            if profile.get(key) in (None, "", 0, 0.0):
                profile[key] = value
        elif isinstance(value, str) and key not in profile:
            profile[key] = value
    if "feature_column_count" in batch_profile:
        profile["feature_column_count"] = float(batch_profile["feature_column_count"])  # type: ignore[arg-type]
    for key in ("direct_column_count", "scalar_fallback_column_count", "direct_column_coverage_ratio"):
        if key in batch_profile:
            profile[key] = float(batch_profile[key])  # type: ignore[arg-type]
    if "memory_peak_mb" in batch_profile:
        profile["memory_peak_mb"] = max(
            float(profile.get("memory_peak_mb", 0.0)),
            float(batch_profile["memory_peak_mb"]),  # type: ignore[arg-type]
        )
    profile["continuation_decision_seconds"] += float(
        batch_profile.get("t3_continuation_total_seconds", 0.0)
    )


def _stage7_aggregate_stats(rollout_by_action: dict[int, _ActionRolloutAggregate]) -> dict[str, Any]:
    decision_count = sum(aggregate.t3_eval_count for aggregate in rollout_by_action.values())
    override_count = sum(aggregate.t3_fired_count for aggregate in rollout_by_action.values())
    return {
        "stage7_decision_count": decision_count,
        "stage7_override_count": override_count,
        "stage7_override_rate": override_count / decision_count if decision_count else 0.0,
        "stage7_policy_latency_ms": 0.0,
    }


def _rollout_after_hero_t2_action(
    *,
    hero_board: Board,
    opponent_board: Board,
    dead_cards: Iterable[str],
    hero_private_discards: Iterable[str] | None = None,
    opponent_private_discards: Iterable[str] | None = None,
    hero_seat: str,
    future_cards: list[str],
    hero_policy: RegularAiPolicy,
    opponent_policy: RegularAiPolicy,
    profile: dict[str, float] | None = None,
    final_turn_cache: FinalTurnDecisionCache | None = None,
    use_final_turn_cache: bool = True,
    t4_search_config: T4SearchConfig | None = None,
    future_index: int = 0,
    future_rollout_seed: int = 0,
    root_fingerprint: str = "legacy_t2_rollout",
) -> float | None:
    cursor = 0
    hero = hero_board
    opponent = opponent_board
    dead = list(dead_cards)
    hero_private = list(() if hero_private_discards is None else hero_private_discards)
    opponent_private = list(() if opponent_private_discards is None else opponent_private_discards)

    def decision_seed(actor: str, card_count: int) -> int:
        street = {5: "T1", 7: "T2", 9: "T3", 11: "T4"}.get(
            card_count, "T4"
        )
        return policy_decision_seed(
            base_seed=future_rollout_seed,
            run_id="hu_turn2_teacher_rollout",
            root_fingerprint=root_fingerprint,
            future_index=future_index,
            actor=actor,  # type: ignore[arg-type]
            street=street,
            decision_ordinal=card_count * 2 + (0 if actor == "hero" else 1),
        )

    def visible_dead_for(actor: str) -> tuple[str, ...]:
        if actor == "hero":
            return (*opponent.all_cards(), *tuple(hero_private))
        if actor == "opponent":
            return (*hero.all_cards(), *tuple(opponent_private))
        raise ValueError(f"unknown actor: {actor}")

    def draw3() -> tuple[str, str, str] | None:
        nonlocal cursor
        if cursor + 3 > len(future_cards):
            return None
        dealt = tuple(future_cards[cursor : cursor + 3])
        cursor += 3
        return dealt  # type: ignore[return-value]

    while hero.card_count() < 13 or opponent.card_count() < 13:
        actor = _next_actor(hero, opponent, hero_seat)
        if actor == "hero":
            if hero.card_count() >= 13:
                continue
            dealt = draw3()
            if dealt is None:
                return None
            actor_card_count = hero.card_count()
            if actor_card_count == 11:
                action = _choose_rollout_t4_action(
                    actor_board=hero,
                    opponent_board=opponent,
                    dealt_cards=dealt,
                    actor_private_discards=hero_private,
                    actor_dead_cards=visible_dead_for("hero"),
                    actor="hero",
                    actor_seat=hero_policy.seat,
                    hero_seat=hero_seat,
                    profile=profile,
                    final_turn_cache=final_turn_cache,
                    use_final_turn_cache=use_final_turn_cache,
                    t4_search_config=t4_search_config,
                )
                if action is None:
                    return None
                if profile is not None:
                    profile["final_turn_state_count"] += 1.0
            else:
                decision_started_at = time.perf_counter()
                action = hero_policy.choose_action(
                    hero,
                    dealt,
                    dead_cards=visible_dead_for("hero"),
                    opponent_board=opponent,
                    decision_seed=decision_seed("hero", actor_card_count),
                    street={5: "T1", 7: "T2", 9: "T3"}.get(actor_card_count),
                )
                if profile is not None:
                    elapsed = time.perf_counter() - decision_started_at
                    profile["continuation_decision_seconds"] += elapsed
                    profile["non_t3_continuation_decision_seconds"] += elapsed
                    _add_non_t3_decision_profile(
                        profile,
                        actor="hero",
                        card_count=actor_card_count,
                        elapsed=elapsed,
                        hero_seat=hero_seat,
                        include_model_bucket=True,
                    )
            hero = hero.place(action.placements)
            hero_private.extend(action.discards)
            dead.extend(action.discards)
        else:
            if opponent.card_count() >= 13:
                continue
            dealt = draw3()
            if dealt is None:
                return None
            actor_card_count = opponent.card_count()
            if actor_card_count == 11:
                action = _choose_rollout_t4_action(
                    actor_board=opponent,
                    opponent_board=hero,
                    dealt_cards=dealt,
                    actor_private_discards=opponent_private,
                    actor_dead_cards=visible_dead_for("opponent"),
                    actor="opponent",
                    actor_seat=opponent_policy.seat,
                    hero_seat=hero_seat,
                    profile=profile,
                    final_turn_cache=final_turn_cache,
                    use_final_turn_cache=use_final_turn_cache,
                    t4_search_config=t4_search_config,
                )
                if action is None:
                    return None
                if profile is not None:
                    profile["final_turn_state_count"] += 1.0
            else:
                decision_started_at = time.perf_counter()
                action = opponent_policy.choose_action(
                    opponent,
                    dealt,
                    dead_cards=visible_dead_for("opponent"),
                    opponent_board=hero,
                    decision_seed=decision_seed("opponent", actor_card_count),
                    street={5: "T1", 7: "T2", 9: "T3"}.get(actor_card_count),
                )
                if profile is not None:
                    elapsed = time.perf_counter() - decision_started_at
                    profile["continuation_decision_seconds"] += elapsed
                    profile["non_t3_continuation_decision_seconds"] += elapsed
                    _add_non_t3_decision_profile(
                        profile,
                        actor="opponent",
                        card_count=actor_card_count,
                        elapsed=elapsed,
                        hero_seat=hero_seat,
                        include_model_bucket=True,
                    )
            opponent = opponent.place(action.placements)
            opponent_private.extend(action.discards)
            dead.extend(action.discards)

    scoring_started_at = time.perf_counter()
    score, _board_score = terminal_score(hero, opponent, fl_ev=DEFAULT_FL_EV)
    if profile is not None:
        profile["hand_evaluation_seconds"] += time.perf_counter() - scoring_started_at
    return float(score)


def _next_actor(hero: Board, opponent: Board, hero_seat: str) -> str:
    if hero.card_count() < opponent.card_count():
        return "hero"
    if opponent.card_count() < hero.card_count():
        return "opponent"
    return "hero" if hero_seat == "first" else "opponent"


def _line_score_component(own_value: tuple[int, tuple[int, ...]], opp_value: tuple[int, tuple[int, ...]]) -> int:
    if own_value > opp_value:
        return 1
    if own_value < opp_value:
        return -1
    return 0


def _terminal_component_breakdown(hero: Board, opponent: Board) -> dict[str, float]:
    hero_score = score_board(hero.top, hero.middle, hero.bottom)
    opponent_score = score_board(opponent.top, opponent.middle, opponent.bottom)
    hero_royalty = 0 if hero_score.busted else hero_score.total_royalty
    opponent_royalty = 0 if opponent_score.busted else opponent_score.total_royalty
    hero_fl = 0.0 if hero_score.busted else float(DEFAULT_FL_EV.get(hero_score.fl_entry.card_count, 0.0))
    opponent_fl = 0.0 if opponent_score.busted else float(DEFAULT_FL_EV.get(opponent_score.fl_entry.card_count, 0.0))
    line_score_delta = 0
    scoop_delta = 0
    foul_delta = 0
    if hero_score.busted and opponent_score.busted:
        terminal = 0.0
    elif hero_score.busted:
        foul_delta = -6
        terminal = float(foul_delta - opponent_royalty - opponent_fl)
    elif opponent_score.busted:
        foul_delta = 6
        terminal = float(foul_delta + hero_royalty + hero_fl)
    else:
        line_score_delta = sum(
            _line_score_component(own, opp)
            for own, opp in (
                (hero_score.top_value, opponent_score.top_value),
                (hero_score.middle_value, opponent_score.middle_value),
                (hero_score.bottom_value, opponent_score.bottom_value),
            )
        )
        scoop_delta = 3 if line_score_delta == 3 else (-3 if line_score_delta == -3 else 0)
        terminal = float(line_score_delta + scoop_delta + hero_royalty - opponent_royalty + hero_fl - opponent_fl)
    return {
        "terminal_score": float(terminal),
        "royalty_delta": float(hero_royalty - opponent_royalty),
        "fl_delta": float(hero_fl - opponent_fl),
        "line_score_delta": float(line_score_delta),
        "scoop_delta": float(scoop_delta),
        "foul_delta": float(foul_delta),
    }


def _future_rollout_digest(future_rollouts: list[tuple[str, ...]]) -> str:
    digest = hashlib.sha256()
    for rollout in future_rollouts:
        digest.update(",".join(rollout).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _prepare_policy_log(policy: RegularAiPolicy) -> int:
    if policy.hu_turn3_decision_log is None:
        policy.hu_turn3_decision_log = []
    return len(policy.hu_turn3_decision_log)


def _stage7_log_stats(
    hero_policy: RegularAiPolicy,
    opponent_policy: RegularAiPolicy,
    hero_start: int,
    opponent_start: int,
) -> dict[str, Any]:
    hero_records = (
        hero_policy.hu_turn3_decision_log[hero_start:]
        if hero_policy.hu_turn3_decision_log is not None
        else []
    )
    opponent_records = (
        opponent_policy.hu_turn3_decision_log[opponent_start:]
        if opponent_policy.hu_turn3_decision_log is not None
        else []
    )
    records = [*hero_records, *opponent_records]
    decision_count = len(records)
    override_count = sum(1 for record in records if bool(record.get("override_fired", False)))
    latency_ms = sum(float(record.get("runtime_latency_ms", 0.0) or 0.0) for record in records)
    return {
        "stage7_decision_count": decision_count,
        "stage7_override_count": override_count,
        "stage7_override_rate": override_count / decision_count if decision_count else 0.0,
        "stage7_policy_latency_ms": latency_ms,
    }


def _standard_error(scores: list[float]) -> float:
    if len(scores) <= 1:
        return 0.0
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / (len(scores) - 1)
    return math.sqrt(variance / len(scores))


def _prediction_margin(predictions: Any, best_index: int) -> float:
    values = [float(value) for value in predictions]
    if len(values) <= 1:
        return 0.0
    best = values[best_index]
    second = max(value for index, value in enumerate(values) if index != best_index)
    return best - second


def _find_sorted_action(actions: list[dict[str, Any]], original_index: int) -> dict[str, Any] | None:
    for action in actions:
        if int(action.get("original_index", -1)) == original_index:
            return action
    return None


def _action_summary(action: dict[str, Any] | None) -> dict[str, Any] | None:
    if action is None:
        return None
    return {
        "sorted_index": None,
        "original_index": int(action.get("original_index", -1)),
        "score": float(action.get("score", 0.0)),
        "ev": float(action.get("score", 0.0)),
        "ev_standard_error": float(action.get("ev_standard_error", 0.0)),
        "rollout_count": int(action.get("rollout_count", 0)),
        "action": {
            "placements": action.get("placements", []),
            "discards": action.get("discards", []),
            "next_board": action.get("next_board", {}),
        },
    }


def _gate_label(delta: float, se_delta: float) -> str:
    if delta >= 0.25 and delta >= 2.0 * se_delta:
        return "positive"
    if delta <= 0.05:
        return "negative"
    return "gray"


def _passes_source_bucket(sample: dict[str, Any], source_bucket: SourceBucket) -> bool:
    delta = float(sample.get("delta_best_vs_baseline", 0.0))
    margin = float(sample.get("best_margin", 0.0))
    disagreement = bool(sample.get("teacher_distribution_metrics", {}).get("baseline_disagreement", False))
    if source_bucket in {"natural", "random_off_policy", "from_pool"}:
        return True
    if source_bucket == "teacher_disagreement":
        return disagreement
    if source_bucket == "high_regret":
        return delta >= 1.0
    if source_bucket == "low_margin":
        return margin <= 0.25
    if source_bucket == "high_margin":
        return margin >= 3.0 or delta >= 3.0
    raise ValueError(f"unknown source bucket: {source_bucket}")


def _attach_broad_bucket_metadata(
    sample: dict[str, Any],
    *,
    source_bucket_requested: str,
    source_bucket_actual: str,
    pool_record: dict[str, Any] | None = None,
) -> None:
    delta = float(sample.get("delta_best_vs_baseline", 0.0))
    margin = float(sample.get("best_margin", 0.0))
    disagreement = bool(sample.get("teacher_distribution_metrics", {}).get("baseline_disagreement", False))
    actual_high_regret = delta >= 1.0
    actual_low_margin = margin <= 0.25
    actual_teacher_disagreement = disagreement
    sample["source_bucket_requested"] = source_bucket_requested
    sample["source_bucket_actual"] = source_bucket_actual
    sample["actual_high_regret"] = actual_high_regret
    sample["actual_low_margin"] = actual_low_margin
    sample["actual_teacher_disagreement"] = actual_teacher_disagreement
    sample["actual_high_margin"] = margin >= 3.0 or delta >= 3.0
    sample["actual_bucket"] = (
        "actual_high_regret"
        if actual_high_regret
        else "actual_low_margin"
        if actual_low_margin
        else "actual_teacher_disagreement"
        if actual_teacher_disagreement
        else "actual_other"
    )
    sample["actual_candidate_bucket"] = sample["actual_bucket"]
    sample["reference_margin_raw"] = float(sample.get("baseline_model_margin", 0.0) or 0.0)
    sample["gate_label"] = sample.get("teacher_label")
    sample["teacher_delta"] = delta
    sample["SE_delta"] = float(sample.get("SE_delta_best_vs_baseline", 0.0) or 0.0)

    if pool_record is None:
        sample.setdefault("predicted_bucket", source_bucket_actual)
        sample.setdefault("predicted_delta", None)
        return

    sample["candidate_pool_state_hash"] = pool_record.get("state_hash")
    sample["candidate_pool_prefilter_version"] = pool_record.get("prefilter_version")
    sample["candidate_pool_predicted_bucket"] = pool_record.get("predicted_bucket")
    sample["candidate_pool_accept_reason"] = pool_record.get("accept_reason")
    sample["candidate_pool_source_bucket_requested"] = pool_record.get("source_bucket_requested")
    sample["candidate_pool_source_bucket_actual"] = pool_record.get("source_bucket_actual")
    sample["predicted_bucket"] = pool_record.get("predicted_bucket")
    sample["predicted_delta"] = pool_record.get("predicted_delta_vs_baseline")
    sample["predicted_delta_vs_baseline"] = pool_record.get("predicted_delta_vs_baseline")
    sample["predicted_delta_vs_reference"] = pool_record.get("predicted_delta_vs_reference")
    sample["predicted_margin"] = pool_record.get("predicted_margin")
    sample["predicted_score_spread"] = pool_record.get("predicted_score_spread")
    sample["predicted_reference_margin"] = pool_record.get("reference_margin")
    sample["predicted_baseline_margin"] = pool_record.get("baseline_margin")
    sample["predicted_baseline_reference_score_gap"] = pool_record.get("baseline_reference_score_gap")


def _uses_bucket_prefilter(source_bucket: SourceBucket, prefilter_future_samples: int) -> bool:
    return prefilter_future_samples > 0 and source_bucket in {
        "teacher_disagreement",
        "high_regret",
        "low_margin",
    }


def _passes_bucket_prefilter(sample: dict[str, Any], source_bucket: SourceBucket) -> tuple[bool, str]:
    delta = float(sample.get("delta_best_vs_baseline", 0.0))
    margin = float(sample.get("best_margin", 0.0))
    disagreement = bool(sample.get("teacher_distribution_metrics", {}).get("baseline_disagreement", False))
    if source_bucket == "teacher_disagreement":
        if disagreement:
            return True, "cheap_disagreement"
        return False, "prefilter_no_disagreement"
    if source_bucket == "high_regret":
        if delta >= 0.50:
            return True, "cheap_high_delta"
        if disagreement and delta >= 0.25:
            return True, "cheap_disagreement_delta"
        return False, "prefilter_low_delta"
    if source_bucket == "low_margin":
        if margin <= 0.50:
            return True, "cheap_low_margin"
        return False, "prefilter_high_margin"
    return True, "prefilter_not_required"


def _cheap_no_rollout_proxy(
    *,
    board: Board,
    dealt_cards: Iterable[str],
    opponent_board: Board,
    dead_cards: Iterable[str],
    visible_dead_cards: Iterable[str] | None = None,
    hero_private_discards: Iterable[str] = (),
    hero_seat: str,
    baseline_turn2_model: object,
) -> dict[str, Any]:
    dealt = tuple(dealt_cards)
    if visible_dead_cards is None and not tuple(hero_private_discards):
        raise ValueError(
            "T2 cheap proxy requires explicit hero_private_discards or "
            "visible_dead_cards; ambiguous dead_cards are forbidden"
        )
    visible_dead = _visible_dead_cards_for_turn2_state(
        opponent_board=opponent_board,
        dead_cards=dead_cards,
        visible_dead_cards=visible_dead_cards,
        hero_private_discards=hero_private_discards,
    )
    actions = generate_turn_actions(board, dealt)
    if not actions:
        return {
            "actions": [],
            "legal_action_count": 0,
            "baseline_reference_disagree": False,
        }
    baseline_sample = policy_sample(board, dealt, actions)
    predictions = np.asarray(baseline_turn2_model.predict_sample(baseline_sample), dtype=np.float64)
    order = canonical_descending_indices(predictions, actions)
    top1 = int(order[0])
    top2 = int(order[1]) if len(order) > 1 else top1
    heuristic_scores = [
        _cheap_action_heuristic(board.place(action.placements), opponent_board, action)
        for action in actions
    ]
    heuristic_order = sorted(
        range(len(actions)),
        key=lambda idx: (-heuristic_scores[idx], action_key(actions[idx]).sort_key()),
    )
    reference = int(heuristic_order[0])
    reference2 = int(heuristic_order[1]) if len(heuristic_order) > 1 else reference
    spread = float(np.max(predictions) - np.min(predictions)) if len(predictions) else 0.0
    predicted_margin = float(predictions[top1] - predictions[top2]) if top1 != top2 else 0.0
    reference_margin = float(heuristic_scores[reference] - heuristic_scores[reference2]) if reference != reference2 else 0.0
    baseline_reference_gap = float(predictions[top1] - predictions[reference])
    row_pressure = _row_pressure(board)
    best_after = board.place(actions[top1].placements)
    reference_after = board.place(actions[reference].placements)
    foul_risk = _cheap_foul_risk(best_after)
    royalty_potential = _cheap_royalty_potential(best_after)
    fl_distance = _cheap_fl_distance(best_after)
    dead_pressure = _cheap_dead_card_pressure(best_after, visible_dead)
    action_payloads = []
    for index, action in enumerate(actions):
        action_payloads.append(
            {
                "placements": [list(item) for item in action.placements],
                "discards": list(action.discards),
                "predicted_score": float(predictions[index]),
                "heuristic_score": float(heuristic_scores[index]),
                "original_index": index,
                "canonical_action_key": action_key(action).to_token(),
            }
        )
    return {
        "schema": "hu_turn2_cheap_no_rollout_v1",
        "action_key_schema": ACTION_KEY_SCHEMA,
        "legal_action_set_digest": legal_action_set_digest(actions),
        "legal_action_order_digest": ordered_action_mapping_digest(actions),
        "actions": action_payloads,
        "baseline_action": _cheap_action_summary(actions[top1], top1, float(predictions[top1])),
        "reference_action": _cheap_action_summary(actions[reference], reference, float(predictions[reference])),
        "baseline_reference_disagree": top1 != reference,
        "baseline_index": top1,
        "baseline_action_key": action_key(actions[top1]).to_token(),
        "reference_index": reference,
        "reference_action_key": action_key(actions[reference]).to_token(),
        "legal_action_count": len(actions),
        "top1_predicted_score": float(predictions[top1]),
        "top2_predicted_score": float(predictions[top2]),
        "predicted_score_spread": spread,
        "predicted_margin": predicted_margin,
        "predicted_delta_vs_baseline": 0.0,
        "predicted_delta_vs_reference": baseline_reference_gap,
        "reference_margin": reference_margin,
        "baseline_margin": predicted_margin,
        "baseline_reference_score_gap": baseline_reference_gap,
        "foul_risk_heuristic": foul_risk,
        "royalty_potential_heuristic": royalty_potential,
        "fl_distance_heuristic": fl_distance,
        "dead_card_pressure": dead_pressure,
        "row_empty_top": 3 - len(board.top),
        "row_empty_middle": 5 - len(board.middle),
        "row_empty_bottom": 5 - len(board.bottom),
        "front_pressure": row_pressure["front_pressure"],
        "middle_pressure": row_pressure["middle_pressure"],
        "back_pressure": row_pressure["back_pressure"],
        "position": hero_seat,
        "seat": hero_seat,
        "to_act_order": to_act_order_for(board, opponent_board),
        "best_after_board": board_to_json(best_after),
        "reference_after_board": board_to_json(reference_after),
    }


def _passes_no_rollout_pool(proxy: dict[str, Any], predicted_bucket: str) -> tuple[bool, str]:
    if int(proxy.get("legal_action_count", 0)) <= 0:
        return False, "no_legal_actions"
    disagreement = bool(proxy.get("baseline_reference_disagree", False))
    predicted_margin = float(proxy.get("predicted_margin", 0.0))
    spread = float(proxy.get("predicted_score_spread", 0.0))
    ref_gap = abs(float(proxy.get("baseline_reference_score_gap", 0.0)))
    ref_margin = float(proxy.get("reference_margin", 0.0))
    legal_count = int(proxy.get("legal_action_count", 0))
    swing = (
        abs(float(proxy.get("royalty_potential_heuristic", 0.0)))
        + abs(float(proxy.get("foul_risk_heuristic", 0.0)))
        + max(0.0, 3.0 - float(proxy.get("fl_distance_heuristic", 3.0))) * 0.5
    )
    if predicted_bucket == "predicted_teacher_disagreement":
        if disagreement:
            return True, "no_rollout_disagreement"
        return False, "no_rollout_no_disagreement"
    if predicted_bucket == "predicted_high_regret":
        if disagreement and predicted_margin <= 0.50 and ref_margin >= 1.00 and ref_gap <= 2.00:
            return True, "no_rollout_disagreement_reference_edge"
        if disagreement and predicted_margin <= 0.25 and swing >= 3.50:
            return True, "no_rollout_uncertain_high_swing"
        if legal_count >= 24 and predicted_margin <= 0.15 and swing >= 3.00:
            return True, "no_rollout_many_action_high_swing"
        return False, "no_rollout_low_regret_proxy"
    if predicted_bucket == "predicted_low_margin":
        if predicted_margin <= 0.20:
            return True, "no_rollout_low_predicted_margin"
        if predicted_margin <= 0.35 and legal_count >= 18:
            return True, "no_rollout_many_close_actions"
        if spread <= 1.0 and legal_count >= 18:
            return True, "no_rollout_low_spread"
        return False, "no_rollout_high_margin_proxy"
    return True, "no_rollout_unknown_target"


def _default_predicted_bucket(source_bucket: SourceBucket) -> str:
    if source_bucket == "high_regret":
        return "predicted_high_regret"
    if source_bucket == "low_margin":
        return "predicted_low_margin"
    if source_bucket == "teacher_disagreement":
        return "predicted_teacher_disagreement"
    return f"predicted_{source_bucket}"


def _actual_bucket_for_predicted(predicted_bucket: str) -> SourceBucket:
    if predicted_bucket == "predicted_high_regret":
        return "high_regret"
    if predicted_bucket == "predicted_low_margin":
        return "low_margin"
    if predicted_bucket == "predicted_teacher_disagreement":
        return "teacher_disagreement"
    return "from_pool"


def _cheap_action_summary(action: Action, index: int, score: float) -> dict[str, Any]:
    return {
        "original_index": index,
        "canonical_action_key": action_key(action).to_token(),
        "predicted_score": score,
        "action": {
            "placements": [list(item) for item in action.placements],
            "discards": list(action.discards),
        },
    }


def _cheap_action_heuristic(board: Board, opponent_board: Board, action: Action) -> float:
    score = 0.0
    score += _cheap_royalty_potential(board) * 1.2
    score += max(0.0, 3.0 - _cheap_fl_distance(board)) * 0.8
    score -= _cheap_foul_risk(board) * 1.5
    score -= len(action.discards) * 0.05
    score += len(board.bottom) * 0.02 + len(board.middle) * 0.015 + len(board.top) * 0.01
    if opponent_board.card_count() > board.card_count():
        score += 0.05
    return score


def _row_pressure(board: Board) -> dict[str, float]:
    return {
        "front_pressure": len(board.top) / 3.0,
        "middle_pressure": len(board.middle) / 5.0,
        "back_pressure": len(board.bottom) / 5.0,
    }


def _rank_counts(cards: Iterable[str]) -> Counter[str]:
    return Counter(card[0] for card in cards)


def _cheap_royalty_potential(board: Board) -> float:
    top_counts = _rank_counts(board.top)
    middle_counts = _rank_counts(board.middle)
    bottom_counts = _rank_counts(board.bottom)
    score = 0.0
    for rank, count in top_counts.items():
        if count >= 2 and rank in {"Q", "K", "A"}:
            score += 2.0
        elif count >= 2:
            score += 0.4
        if count >= 3:
            score += 3.0
    for counts, weight in ((middle_counts, 0.9), (bottom_counts, 0.7)):
        pairs = sum(1 for count in counts.values() if count >= 2)
        trips = sum(1 for count in counts.values() if count >= 3)
        quads = sum(1 for count in counts.values() if count >= 4)
        score += weight * (pairs * 0.4 + trips * 1.0 + quads * 2.0)
    score += _cheap_suit_potential(board.middle) * 0.25
    score += _cheap_suit_potential(board.bottom) * 0.20
    return score


def _cheap_suit_potential(cards: Iterable[str]) -> float:
    suits = Counter(card[1] for card in cards)
    return float(max(suits.values(), default=0))


def _cheap_fl_distance(board: Board) -> float:
    counts = _rank_counts(board.top)
    if any(count >= 3 for count in counts.values()):
        return 0.0
    if any(count >= 2 and rank in {"Q", "K", "A"} for rank, count in counts.items()):
        return 0.0
    if any(rank in {"Q", "K", "A"} for rank in counts):
        return 1.0
    if len(board.top) <= 1:
        return 2.0
    return 3.0


def _cheap_foul_risk(board: Board) -> float:
    risk = 0.0
    if len(board.top) >= 2 and len(board.middle) >= 3:
        top_pair = max(_rank_counts(board.top).values(), default=0) >= 2
        middle_pair = max(_rank_counts(board.middle).values(), default=0) >= 2
        if top_pair and not middle_pair:
            risk += 1.0
    if len(board.middle) >= 4 and len(board.bottom) >= 4:
        middle_pairs = sum(1 for count in _rank_counts(board.middle).values() if count >= 2)
        bottom_pairs = sum(1 for count in _rank_counts(board.bottom).values() if count >= 2)
        if middle_pairs > bottom_pairs + 1:
            risk += 0.8
    return risk


def _cheap_dead_card_pressure(board: Board, visible_dead_cards: Iterable[str]) -> float:
    dead_counts = _rank_counts(visible_dead_cards)
    pressure = 0.0
    for row in (board.top, board.middle, board.bottom):
        for rank, count in _rank_counts(row).items():
            if count == 1:
                pressure += min(dead_counts.get(rank, 0), 3) * 0.1
    return pressure


def _new_attempt_profile(source_bucket: SourceBucket) -> dict[str, Any]:
    return {
        "source_bucket": source_bucket,
        "attempt_count": 0,
        "accepted_count": 0,
        "skipped_count": 0,
        "accept_rate": 0.0,
        "skipped_reason_counts": {},
        "state_generation_time": 0.0,
        "bucket_predicate_time": 0.0,
        "baseline_reference_policy_time": 0.0,
        "cheap_score_time": 0.0,
        "candidate_eval_time": 0.0,
        "full_teacher_rollout_time": 0.0,
        "skip_before_rollout_count": 0,
        "skip_after_rollout_count": 0,
        "attempt_wall_seconds": 0.0,
        "wall_seconds_per_written": 0.0,
        "profile_seconds_per_written": 0.0,
    }


def _attempt_profile_add(profile: dict[str, Any], key: str, value: float) -> None:
    profile[key] = float(profile.get(key, 0.0)) + float(value)
    if key in {"cheap_score_time", "full_teacher_rollout_time"}:
        profile["candidate_eval_time"] = (
            float(profile.get("cheap_score_time", 0.0))
            + float(profile.get("full_teacher_rollout_time", 0.0))
        )


def _attempt_profile_count(profile: dict[str, Any], key: str, value: int = 1) -> None:
    profile[key] = int(profile.get(key, 0)) + int(value)


def _attempt_profile_reason(profile: dict[str, Any], reason: str) -> None:
    counts = dict(profile.get("skipped_reason_counts", {}))
    counts[reason] = int(counts.get(reason, 0)) + 1
    profile["skipped_reason_counts"] = counts


def _finalize_attempt_profile(profile: dict[str, Any], sample_profiles: list[dict[str, Any]]) -> dict[str, Any]:
    result = dict(profile)
    attempts = int(result.get("attempt_count", 0))
    accepted = int(result.get("accepted_count", 0))
    result["accept_rate"] = accepted / attempts if attempts else 0.0
    result["wall_seconds_per_written"] = (
        float(result.get("attempt_wall_seconds", 0.0)) / accepted if accepted else 0.0
    )
    profile_seconds = sum(float(item.get("seconds_total", 0.0)) for item in sample_profiles)
    result["profile_seconds_per_written"] = profile_seconds / accepted if accepted else 0.0
    return result


def write_hu_turn2_attempt_profile_csv(profile: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    row = dict(profile)
    row["skipped_reason_counts"] = json.dumps(
        row.get("skipped_reason_counts", {}),
        sort_keys=True,
        separators=(",", ":"),
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _default_attempt_profile_output(output: Path) -> Path:
    return output.with_name(f"{output.stem}_attempt_profile.csv")


def _cheap_future_seed(seed: int, hand_seed: int, attempt: int, player: int) -> int:
    return (seed * 1_000_003 + hand_seed * 193 + attempt * 31 + player + 911) & 0x7FFFFFFF


def _pool_position_quota_allows(
    counts: dict[str, int],
    seat: str,
    target_size: int,
) -> bool:
    if target_size <= 0:
        return True
    first_quota = (target_size + 1) // 2
    second_quota = target_size // 2
    quota = first_quota if seat == "first" else second_quota
    return int(counts.get(seat, 0)) < quota


def _candidate_pool_record(
    *,
    seed: int,
    hand_seed: int,
    hand_index: int,
    state_id: int,
    player: int,
    board: Board,
    dealt_cards: Iterable[str],
    opponent_board: Board,
    dead_cards: Iterable[str],
    visible_dead_cards: Iterable[str] | None = None,
    hero_private_discards: Iterable[str] = (),
    opponent_private_discards: Iterable[str] = (),
    hero_seat: str,
    source_bucket: SourceBucket,
    cheap_sample: dict[str, Any] | None,
    accept_reason: str,
    predicted_bucket: str | None = None,
    prefilter_version: str = BUCKET_PREFILTER_VERSION,
) -> dict[str, Any]:
    dealt = tuple(dealt_cards)
    dead = tuple(dead_cards)
    visible_dead = _visible_dead_cards_for_turn2_state(
        opponent_board=opponent_board,
        dead_cards=dead,
        visible_dead_cards=visible_dead_cards,
        hero_private_discards=hero_private_discards,
    )
    hero_private = tuple(hero_private_discards)
    opponent_private = tuple(opponent_private_discards)
    observation = ActorObservation(
        hero_board=board,
        opponent_public_board=opponent_board,
        dealt_cards=dealt,
        hero_private_discards=hero_private,
        seat=hero_seat,  # type: ignore[arg-type]
        street="T2",
        to_act_order=to_act_order_for(board, opponent_board),  # type: ignore[arg-type]
    )
    if observation.legacy_dead_cards() != visible_dead:
        raise ValueError(
            "candidate-pool visible cards disagree with ActorObservation"
        )
    replay_truth = ReplayTruth(
        true_dead_cards=dead,
        visible_dead_cards=visible_dead,
        hero_private_discards=hero_private,
        opponent_private_discards=opponent_private,
    )
    cheap = cheap_sample or {}
    actions = cheap.get("actions", [])
    spread = 0.0
    if actions:
        scores = [float(action.get("score", 0.0)) for action in actions]
        if not any("score" in action for action in actions):
            scores = [float(action.get("predicted_score", 0.0)) for action in actions]
        spread = max(scores) - min(scores) if scores else 0.0
    record = {
        "schema": "hu_turn2_candidate_pool_v2",
        "turn": "T2",
        "prefilter_version": prefilter_version,
        "seed": seed,
        "hand_seed": hand_seed,
        "hand_index": hand_index,
        "state_id": state_id,
        "player": player,
        "seat": hero_seat,
        "to_act_order": to_act_order_for(board, opponent_board),
        "source_bucket_requested": source_bucket,
        "source_bucket_actual": source_bucket,
        "predicted_bucket": predicted_bucket or _default_predicted_bucket(source_bucket),
        "board": board_to_json(board),
        "opponent_board": board_to_json(opponent_board),
        "cards_to_place": list(dealt),
        "dealt": list(dealt),
        "dead_cards": list(visible_dead),
        "visible_dead_cards": list(visible_dead),
        "hero_private_discards": list(hero_private),
        "policy_observation": observation.to_dict(),
        **replay_truth.to_legacy_record_fields(),
        "baseline_action": cheap.get("baseline_action"),
        "reference_action": cheap.get("reference_action"),
        "baseline_reference_disagree": bool(cheap.get("baseline_reference_disagree", False)),
        "legal_action_count": int(cheap.get("legal_action_count", len(actions))),
        "top1_predicted_score": float(cheap.get("top1_predicted_score", 0.0)),
        "top2_predicted_score": float(cheap.get("top2_predicted_score", 0.0)),
        "predicted_score_spread": float(cheap.get("predicted_score_spread", spread)),
        "predicted_margin": float(cheap.get("predicted_margin", cheap.get("best_margin", 0.0))),
        "predicted_delta_vs_baseline": float(cheap.get("predicted_delta_vs_baseline", 0.0)),
        "predicted_delta_vs_reference": float(cheap.get("predicted_delta_vs_reference", 0.0)),
        "reference_margin": float(cheap.get("reference_margin", 0.0)),
        "baseline_margin": float(cheap.get("baseline_margin", 0.0)),
        "baseline_reference_score_gap": float(cheap.get("baseline_reference_score_gap", 0.0)),
        "foul_risk_heuristic": float(cheap.get("foul_risk_heuristic", 0.0)),
        "royalty_potential_heuristic": float(cheap.get("royalty_potential_heuristic", 0.0)),
        "fl_distance_heuristic": float(cheap.get("fl_distance_heuristic", 0.0)),
        "dead_card_pressure": float(cheap.get("dead_card_pressure", 0.0)),
        "row_empty_top": int(cheap.get("row_empty_top", 3 - len(board.top))),
        "row_empty_middle": int(cheap.get("row_empty_middle", 5 - len(board.middle))),
        "row_empty_bottom": int(cheap.get("row_empty_bottom", 5 - len(board.bottom))),
        "front_pressure": float(cheap.get("front_pressure", len(board.top) / 3.0)),
        "middle_pressure": float(cheap.get("middle_pressure", len(board.middle) / 5.0)),
        "back_pressure": float(cheap.get("back_pressure", len(board.bottom) / 5.0)),
        "cheap_delta_estimate": float(cheap.get("delta_best_vs_baseline", 0.0)),
        "cheap_margin_estimate": float(cheap.get("best_margin", 0.0)),
        "cheap_score_spread": spread,
        "cheap_teacher_label": cheap.get("teacher_label"),
        "accept_reason": accept_reason,
    }
    record["state_hash"] = _candidate_pool_state_hash(record)
    return record


def _candidate_pool_state_hash(record: dict[str, Any]) -> str:
    payload = {
        "schema": record.get("schema"),
        "board": record.get("board"),
        "opponent_board": record.get("opponent_board"),
        "cards_to_place": record.get("cards_to_place"),
        "dead_cards": record.get("dead_cards"),
        "seat": record.get("seat"),
        "to_act_order": record.get("to_act_order"),
        "source_bucket_requested": record.get("source_bucket_requested"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _board_from_json(payload: dict[str, Any]) -> Board:
    return Board.from_rows(
        top=list(payload.get("top", [])),
        middle=list(payload.get("middle", [])),
        bottom=list(payload.get("bottom", [])),
    )


def _future_seed(seed: int, hand_seed: int, attempt: int, player: int) -> int:
    return (seed * 1_000_003 + hand_seed * 97 + attempt * 17 + player) & 0x7FFFFFFF


def _stage3_feature_replay_source(future_samples: int, samples: int) -> str:
    return f"mc{future_samples}_{samples}state"


def _stage3_feature_teacher_run_hash(
    *,
    seed: int,
    samples: int,
    future_samples: int,
    source_bucket: SourceBucket,
    stage3_feature_encoder_mode: str,
    disable_stage3_feature_fast_path: bool,
    t3_continuation: T3ContinuationMode,
) -> str:
    continuation_metadata = _t3_continuation_metadata(t3_continuation)
    payload = {
        "seed": seed,
        "samples": samples,
        "future_samples": future_samples,
        "source_bucket": source_bucket,
        "stage3_feature_encoder_mode": stage3_feature_encoder_mode,
        "disable_stage3_feature_fast_path": disable_stage3_feature_fast_path,
        "t3_continuation": t3_continuation,
        "continuation_policy_T3": _t3_continuation_policy_name(t3_continuation),
        "stage7_enabled": continuation_metadata["stage7_enabled"],
        "stage7_model": continuation_metadata["stage7_model_path"],
        "stage3_reference_model": str(DEFAULT_HU_TURN3_STAGE7_REFERENCE_MODEL),
        "hu_turn3_min_margin": continuation_metadata["stage7_hu_turn3_min_margin"],
        "hu_turn3_reference_min_margin": continuation_metadata[
            "stage7_hu_turn3_reference_min_margin"
        ],
        "support_model": continuation_metadata.get("support_model_path", ""),
        "gate_model": continuation_metadata.get("gate_model_path", ""),
        "hu_turn3_min_support_margin": continuation_metadata.get("hu_turn3_min_support_margin", 0.0),
        "hu_turn3_min_model_score": continuation_metadata.get("hu_turn3_min_model_score"),
        "hu_turn3_min_gate_probability": continuation_metadata.get("hu_turn3_min_gate_probability", 0.0),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--future-samples", type=int, default=4096)
    parser.add_argument(
        "--source-bucket",
        choices=(
            "natural",
            "teacher_disagreement",
            "high_regret",
            "low_margin",
            "high_margin",
            "random_off_policy",
            "from_pool",
        ),
        default="natural",
    )
    parser.add_argument("--max-hands", type=int, default=1000000)
    parser.add_argument("--opening-lookahead-samples", type=int, default=64)
    parser.add_argument("--prediction-threads", type=int, default=1)
    parser.add_argument(
        "--t3-continuation",
        choices=("stage3_reference_default", "stage7_m5_r10", "stage9d_p07_relaxed_both"),
        default=DEFAULT_T3_CONTINUATION,
        help=(
            "T3 continuation used inside T2 teacher rollouts. "
            "Hidden-discard default is stage3_reference_default; "
            "stage7_m5_r10 and stage9d_p07_relaxed_both must be opted in explicitly."
        ),
    )
    parser.add_argument("--use-batched-continuation", action="store_true")
    parser.add_argument("--batch-continuation-size", type=int, default=8192)
    parser.add_argument("--disable-continuation-cache", action="store_true")
    parser.add_argument("--continuation-cache-size", type=int, default=200000)
    parser.add_argument("--disable-final-turn-cache", action="store_true")
    parser.add_argument("--final-turn-cache-size", type=int, default=200000)
    parser.add_argument(
        "--enable-m2-t4-search",
        action="store_true",
        help=(
            "Use the ActorObservation-only sequential T4 selector for first-to-act "
            "rollout decisions. Omit only to reproduce quarantined legacy artifacts."
        ),
    )
    parser.add_argument("--m2-t4-candidate-samples", type=int, default=16)
    parser.add_argument("--m2-t4-evaluation-samples", type=int, default=32)
    parser.add_argument("--m2-t4-seed", type=int, default=2026071303)
    parser.add_argument("--m2-t4-candidate-seed", type=int, default=2026071301)
    parser.add_argument("--m2-t4-evaluation-seed", type=int, default=2026071302)
    parser.add_argument("--m2-t4-run-id", default="hu-m2-t2-continuation-v1")
    parser.add_argument("--prefilter-future-samples", type=int, default=0)
    parser.add_argument("--build-candidate-pool", action="store_true")
    parser.add_argument("--candidate-pool-output", type=Path)
    parser.add_argument("--candidate-pool-input", type=Path)
    parser.add_argument("--candidate-pool-size", type=int, default=0)
    parser.add_argument(
        "--candidate-pool-prefilter",
        choices=(BUCKET_PREFILTER_VERSION, NO_ROLLOUT_PREFILTER_VERSION),
        default=BUCKET_PREFILTER_VERSION,
    )
    parser.add_argument(
        "--candidate-pool-target-bucket",
        choices=(
            "",
            "predicted_high_regret",
            "predicted_low_margin",
            "predicted_teacher_disagreement",
        ),
        default="",
    )
    parser.add_argument("--candidate-pool-max-attempts", type=int, default=0)
    parser.add_argument("--candidate-pool-balance-position", action="store_true")
    parser.add_argument("--candidate-pool-balance-source", action="store_true")
    parser.add_argument("--disable-stage3-feature-fast-path", action="store_true")
    parser.add_argument(
        "--stage3-feature-encoder-mode",
        choices=("scalar_fast", "numpy_direct_partial", "numpy_direct_full", "rust_direct"),
        default="scalar_fast",
    )
    parser.add_argument("--dump-stage3-feature-replay", type=Path)
    parser.add_argument("--stage3-feature-replay-sample-limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--opening-model", type=Path, default=DEFAULT_OPENING_MODEL)
    parser.add_argument("--turn1-model", type=Path, default=DEFAULT_TURN1_MODEL)
    parser.add_argument("--turn2-model", type=Path, default=DEFAULT_TURN2_MODEL)
    parser.add_argument("--turn3-model", type=Path, default=DEFAULT_TURN3_MODEL)
    parser.add_argument("--stage7-model", type=Path, default=DEFAULT_HU_TURN3_STAGE7_MODEL)
    parser.add_argument("--stage3-reference-model", type=Path, default=DEFAULT_HU_TURN3_STAGE7_REFERENCE_MODEL)
    parser.add_argument("--stage9d-model", type=Path, default=DEFAULT_HU_TURN3_STAGE9D_MODEL)
    parser.add_argument("--stage9d-support-model", type=Path, default=DEFAULT_HU_TURN3_STAGE9D_SUPPORT_MODEL)
    parser.add_argument("--stage9d-gate-model", type=Path, default=DEFAULT_HU_TURN3_STAGE9D_GATE_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    if args.future_samples <= 0:
        raise SystemExit("--future-samples must be positive")
    if args.batch_continuation_size <= 0:
        raise SystemExit("--batch-continuation-size must be positive")
    if args.prefilter_future_samples < 0:
        raise SystemExit("--prefilter-future-samples must be non-negative")
    if args.candidate_pool_max_attempts < 0:
        raise SystemExit("--candidate-pool-max-attempts must be non-negative")
    if args.m2_t4_candidate_samples < 0 or args.m2_t4_evaluation_samples < 0:
        raise SystemExit("M2 T4 sample counts must be non-negative")
    if args.build_candidate_pool and args.candidate_pool_output is None:
        raise SystemExit("--build-candidate-pool requires --candidate-pool-output")
    if args.build_candidate_pool and args.candidate_pool_input is not None:
        raise SystemExit("--build-candidate-pool cannot be combined with --candidate-pool-input")
    if args.prediction_threads > 0:
        os.environ.setdefault("OMP_NUM_THREADS", str(args.prediction_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(args.prediction_threads))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.prediction_threads))

    paths = ModelPaths(
        opening=args.opening_model,
        turn1=args.turn1_model,
        turn2=args.turn2_model,
        turn3=args.turn3_model,
        hu_turn3_stage7=args.stage7_model,
        hu_turn3_stage7_reference=args.stage3_reference_model,
        hu_turn3_stage9d=args.stage9d_model,
        hu_turn3_stage9d_support=args.stage9d_support_model,
        hu_turn3_stage9d_gate=args.stage9d_gate_model,
    )
    requested_model_profile = (
        {
            "stage3_reference_default": "stage3_baseline",
            "stage7_m5_r10": "stage7_m5_r10",
            "stage9d_p07_relaxed_both": "stage9d_p07_relaxed_both",
        }[args.t3_continuation]
        if args.enable_m2_t4_search
        else "current"
    )
    bundle = load_model_bundle(paths, {requested_model_profile})
    if bundle.turn2 is None:
        bundle.turn2 = load_action_value_model(args.turn2_model)
    t4_search_config = (
        T4SearchConfig(
            candidate_samples=args.m2_t4_candidate_samples,
            evaluation_samples=args.m2_t4_evaluation_samples,
            seed=args.m2_t4_seed,
            candidate_seed=args.m2_t4_candidate_seed,
            evaluation_seed=args.m2_t4_evaluation_seed,
            run_id=args.m2_t4_run_id,
        )
        if args.enable_m2_t4_search
        else None
    )
    with _prediction_thread_context(args.prediction_threads):
        if args.candidate_pool_input is not None:
            summary = collect_hu_turn2_dataset_from_candidate_pool(
                candidate_pool_input=args.candidate_pool_input,
                output=args.output,
                samples=args.samples,
                seed=args.seed,
                policy_bundle=bundle,
                future_samples=args.future_samples,
                opening_lookahead_samples=args.opening_lookahead_samples,
                source_bucket=args.source_bucket,
                use_batched_continuation=args.use_batched_continuation,
                batched_continuation_batch_size=args.batch_continuation_size,
                disable_continuation_cache=args.disable_continuation_cache,
                continuation_cache_size=args.continuation_cache_size,
                disable_stage3_feature_fast_path=args.disable_stage3_feature_fast_path,
                stage3_feature_encoder_mode=args.stage3_feature_encoder_mode,
                disable_final_turn_cache=args.disable_final_turn_cache,
                final_turn_cache_size=args.final_turn_cache_size,
                t3_continuation=args.t3_continuation,
                t4_search_config=t4_search_config,
            )
        else:
            summary = collect_hu_turn2_dataset(
                output=args.output,
                samples=args.samples,
                seed=args.seed,
                policy_bundle=bundle,
                future_samples=args.future_samples,
                opening_lookahead_samples=args.opening_lookahead_samples,
                max_hands=args.max_hands,
                source_bucket=args.source_bucket,
                progress_every=args.progress_every,
                use_batched_continuation=args.use_batched_continuation,
                batched_continuation_batch_size=args.batch_continuation_size,
                disable_continuation_cache=args.disable_continuation_cache,
                continuation_cache_size=args.continuation_cache_size,
                disable_stage3_feature_fast_path=args.disable_stage3_feature_fast_path,
                dump_stage3_feature_replay=args.dump_stage3_feature_replay,
                stage3_feature_replay_sample_limit=args.stage3_feature_replay_sample_limit,
                stage3_feature_encoder_mode=args.stage3_feature_encoder_mode,
                disable_final_turn_cache=args.disable_final_turn_cache,
                final_turn_cache_size=args.final_turn_cache_size,
                prefilter_future_samples=args.prefilter_future_samples,
                candidate_pool_output=args.candidate_pool_output,
                build_candidate_pool=args.build_candidate_pool,
                candidate_pool_size=args.candidate_pool_size,
                candidate_pool_prefilter=args.candidate_pool_prefilter,
                candidate_pool_target_bucket=args.candidate_pool_target_bucket,
                candidate_pool_max_attempts=args.candidate_pool_max_attempts,
                candidate_pool_balance_position=args.candidate_pool_balance_position,
                candidate_pool_balance_source=args.candidate_pool_balance_source,
                t3_continuation=args.t3_continuation,
                t4_search_config=t4_search_config,
            )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
