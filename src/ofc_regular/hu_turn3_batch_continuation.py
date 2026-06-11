"""Batched HU Turn3 Stage7 selective override decisions."""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from .action_space import Action, generate_turn_actions
from .cards import ALL_CARDS
from .hu_turn3_model import HU_FEATURE_DIM, hu_policy_sample, sample_to_matrix as hu_sample_to_matrix
from .hu_turn3_stage3_feature_fast import (
    FEATURE_SCHEMA_VERSION,
    Stage3StateFeatureCache,
    build_hu_turn3_stage3_feature_matrix_batch,
)
from .policy import action_to_json, board_to_json, policy_sample
from .state import Board
from .turn3_model import sample_to_matrix as self_sample_to_matrix

CARD_ORDER = {card: index for index, card in enumerate(ALL_CARDS)}


@dataclass(frozen=True)
class HuTurn3State:
    board: Board
    dealt_cards: tuple[str, str, str]
    opponent_board: Board
    dead_cards: tuple[str, ...] = ()
    seat: str = "first"
    to_act_order: str | None = None
    hand_id: str | int | None = None
    game_id: str | int | None = None
    decision_seed: int | None = None
    street: str = "T3"


@dataclass(frozen=True)
class HuTurn3Stage7BatchConfig:
    stage7_enabled: bool = True
    hu_turn3_min_margin: float = 5.0
    hu_turn3_reference_min_margin: float = 10.0
    batch_size: int = 8192
    use_cache: bool = True
    use_stage3_feature_fast_path: bool = True
    stage3_feature_encoder_mode: str = "scalar_fast"
    dump_stage3_feature_replay: str | None = None
    stage3_feature_replay_sample_limit: int = 0
    stage3_feature_replay_model_path: str = ""
    stage3_feature_replay_stage7_model_path: str = ""
    stage3_feature_replay_source: str = ""
    stage3_feature_replay_teacher_run_hash: str = ""


@dataclass
class HuTurn3BatchModels:
    fallback_turn3_model: object | None
    stage7_model: object | None
    stage3_reference_model: object | None


@dataclass
class Turn3Decision:
    final_action: Action | None
    fallback_action: Action | None
    stage3_action: Action | None
    stage7_action: Action | None
    override_fired: bool
    no_override_reason: str
    stage7_predicted_margin: float | None
    reference_margin: float | None
    model_score: float | None
    reference_score: float | None
    legality_check_result: str
    action_count: int
    record: dict[str, Any]


@dataclass
class HuTurn3DecisionCache:
    max_size: int = 200_000
    _items: OrderedDict[tuple[Any, ...], Turn3Decision] = field(default_factory=OrderedDict)
    hits: int = 0
    misses: int = 0

    def get(self, key: tuple[Any, ...]) -> Turn3Decision | None:
        decision = self._items.get(key)
        if decision is None:
            self.misses += 1
            return None
        self.hits += 1
        self._items.move_to_end(key)
        return decision

    def put(self, key: tuple[Any, ...], decision: Turn3Decision) -> None:
        if self.max_size <= 0:
            return
        self._items[key] = decision
        self._items.move_to_end(key)
        while len(self._items) > self.max_size:
            self._items.popitem(last=False)


@dataclass
class Stage3ReferenceDecision:
    stage3_action: Action | None
    action_index: int | None
    reference_margin: float | None
    score: float | None
    rank_score: float | None
    action_count: int
    legality_status: str
    fallback_reason: str


@dataclass
class HuTurn3Stage3ReferenceCache:
    max_size: int = 200_000
    _items: OrderedDict[tuple[Any, ...], Stage3ReferenceDecision] = field(default_factory=OrderedDict)
    hits: int = 0
    misses: int = 0

    def get(self, key: tuple[Any, ...]) -> Stage3ReferenceDecision | None:
        decision = self._items.get(key)
        if decision is None:
            self.misses += 1
            return None
        self.hits += 1
        self._items.move_to_end(key)
        return decision

    def put(self, key: tuple[Any, ...], decision: Stage3ReferenceDecision) -> None:
        if self.max_size <= 0:
            return
        self._items[key] = decision
        self._items.move_to_end(key)
        while len(self._items) > self.max_size:
            self._items.popitem(last=False)


@dataclass
class HuTurn3ActionCache:
    max_size: int = 200_000
    _items: OrderedDict[tuple[Any, ...], list[Action]] = field(default_factory=OrderedDict)
    hits: int = 0
    misses: int = 0

    def get(self, key: tuple[Any, ...]) -> list[Action] | None:
        actions = self._items.get(key)
        if actions is None:
            self.misses += 1
            return None
        self.hits += 1
        self._items.move_to_end(key)
        return actions

    def put(self, key: tuple[Any, ...], actions: list[Action]) -> None:
        if self.max_size <= 0:
            return
        self._items[key] = actions
        self._items.move_to_end(key)
        while len(self._items) > self.max_size:
            self._items.popitem(last=False)


@dataclass
class HuTurn3BatchResult:
    decisions: list[Turn3Decision]
    profile: dict[str, float | int]


@dataclass
class HuTurn3Stage3ReferenceBatchResult:
    decisions: list[Stage3ReferenceDecision]
    profile: dict[str, float | int]


@dataclass
class _PreparedState:
    state: HuTurn3State
    key: tuple[Any, ...]
    actions: list[Action]
    fallback_sample: dict[str, Any] | None = None
    hu_sample: dict[str, Any] | None = None
    fallback_index: int | None = None
    stage3_index: int | None = None
    reference_index: int | None = None
    reference_margin: float | None = None
    reference_score: float | None = None
    stage7_index: int | None = None
    stage7_predicted_margin: float | None = None
    model_score: float | None = None
    legality_check_result: str = "not_evaluated"
    no_override_reason: str = ""


@dataclass
class _PredictionBatchResult:
    predictions: list[np.ndarray | None]
    feature_seconds: float
    inference_seconds: float
    feature_rows: int
    predict_calls: int
    profile: dict[str, float | int | str] = field(default_factory=dict)


def decide_hu_turn3_stage7_batch(
    t3_states: Sequence[HuTurn3State],
    config: HuTurn3Stage7BatchConfig,
    models: HuTurn3BatchModels,
    batch_size: int | None = None,
    *,
    cache: HuTurn3DecisionCache | None = None,
    reference_cache: HuTurn3Stage3ReferenceCache | None = None,
    state_feature_cache: Stage3StateFeatureCache | None = None,
    action_cache: HuTurn3ActionCache | None = None,
) -> HuTurn3BatchResult:
    """Return Stage7 selective override decisions for HU Turn3 states."""

    started_at = time.perf_counter()
    effective_batch_size = int(batch_size or config.batch_size or 8192)
    raw_count = len(t3_states)
    profile: dict[str, float | int] = {
        "raw_t3_states": raw_count,
        "unique_t3_states": 0,
        "unique_raw_ratio": 0.0,
        "t3_continuation_total_seconds": 0.0,
        "t3_dedup_seconds": 0.0,
        "t3_action_generation_seconds": 0.0,
        "t3_action_generation_cache_hit": 0,
        "t3_action_generation_cache_miss": 0,
        "t3_action_generation_cache_hit_rate": 0.0,
        "stage3_fallback_batch_seconds": 0.0,
        "stage3_fallback_feature_seconds": 0.0,
        "stage3_fallback_model_inference_seconds": 0.0,
        "stage3_reference_batch_seconds": 0.0,
        "stage3_reference_feature_seconds": 0.0,
        "stage3_reference_model_inference_seconds": 0.0,
        "stage3_reference_computed_count": 0,
        "stage3_fallback_recomputed_count": 0,
        "stage3_fallback_reuse_count": 0,
        "duplicate_stage3_compute_avoided_count": 0,
        "stage3_feature_rows": 0,
        "stage3_feature_generation_time": 0.0,
        "stage3_hgb_predict_calls": 0,
        "stage3_hgb_predict_time": 0.0,
        "stage3_feature_rows_per_second": 0.0,
        "stage3_predict_rows_per_second": 0.0,
        "stage3_feature_mode": 0,
        "stage3_unique_states": 0,
        "stage3_total_legal_actions": 0,
        "stage3_feature_generation_total": 0.0,
        "stage3_state_feature_time": 0.0,
        "stage3_action_feature_time": 0.0,
        "stage3_matrix_assembly_time": 0.0,
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
        "stage3_state_feature_cache_hit": 0,
        "stage3_state_feature_cache_miss": 0,
        "stage3_state_feature_cache_hit_rate": 0.0,
        "feature_column_count": 0,
        "direct_column_count": 0,
        "scalar_fallback_column_count": 0,
        "direct_column_coverage_ratio": 0.0,
        "memory_peak_mb": 0.0,
        "stage3_reference_cache_hit": 0,
        "stage3_reference_cache_miss": 0,
        "stage3_reference_cache_hit_rate": 0.0,
        "stage3_reference_action_generation_seconds": 0.0,
        "stage3_reference_sample_generation_seconds": 0.0,
        "stage7_feature_batch_seconds": 0.0,
        "stage7_model_inference_seconds": 0.0,
        "stage7_sample_generation_seconds": 0.0,
        "t3_postprocess_gate_seconds": 0.0,
        "cache_hits": 0,
        "cache_misses": 0,
        "skipped_by_reference_margin": 0,
        "stage7_model_called_count": 0,
        "ms_per_raw_t3_decision": 0.0,
        "ms_per_unique_t3_state": 0.0,
        "peak_memory_mb": 0.0,
    }
    if raw_count == 0:
        return HuTurn3BatchResult(decisions=[], profile=profile)

    cache_enabled = config.use_cache and cache is not None
    dedup_started_at = time.perf_counter()
    resolved: list[Turn3Decision | None] = [None] * raw_count
    unique: list[_PreparedState] = []
    first_by_key: dict[tuple[Any, ...], int] = {}
    inverse: dict[int, list[int]] = {}
    cache_hits_before = cache.hits if cache_enabled else 0
    cache_misses_before = cache.misses if cache_enabled else 0
    for raw_index, state in enumerate(t3_states):
        key = canonical_t3_state_key(state)
        if cache_enabled:
            cached = cache.get(key)  # type: ignore[union-attr]
            if cached is not None:
                resolved[raw_index] = cached
                continue
        unique_index = first_by_key.get(key)
        if unique_index is None:
            unique_index = len(unique)
            first_by_key[key] = unique_index
            unique.append(_PreparedState(state=state, key=key, actions=[]))
        inverse.setdefault(unique_index, []).append(raw_index)

    profile["cache_hits"] = (cache.hits - cache_hits_before) if cache_enabled else 0
    profile["cache_misses"] = (cache.misses - cache_misses_before) if cache_enabled else 0
    profile["t3_dedup_seconds"] = time.perf_counter() - dedup_started_at
    profile["unique_t3_states"] = len(unique)
    profile["unique_raw_ratio"] = len(unique) / raw_count if raw_count else 0.0

    action_started_at = time.perf_counter()
    active: list[_PreparedState] = []
    action_cache_hits_before = action_cache.hits if action_cache is not None else 0
    action_cache_misses_before = action_cache.misses if action_cache is not None else 0
    for unique_index, prepared in enumerate(unique):
        cached_actions = action_cache.get(prepared.key) if action_cache is not None else None
        if cached_actions is None:
            try:
                prepared.actions = generate_turn_actions(prepared.state.board, prepared.state.dealt_cards)
            except Exception:
                prepared.actions = []
            if action_cache is not None:
                action_cache.put(prepared.key, prepared.actions)
        else:
            prepared.actions = cached_actions
        if not prepared.actions:
            decision = _decision_from_indices(
                prepared,
                config=config,
                fallback_index=None,
                stage3_index=None,
                stage7_index=None,
                final_index=None,
                override_fired=False,
                no_override_reason="fallback_to_stage3",
            )
            _store_prepared_decision(prepared, decision, resolved, inverse, unique_index, cache)
            continue
        active.append(prepared)
    profile["t3_action_generation_seconds"] = time.perf_counter() - action_started_at
    profile["t3_action_generation_cache_hit"] = (
        action_cache.hits - action_cache_hits_before if action_cache is not None else 0
    )
    profile["t3_action_generation_cache_miss"] = (
        action_cache.misses - action_cache_misses_before if action_cache is not None else len(unique)
    )
    action_hits = float(profile["t3_action_generation_cache_hit"])
    action_misses = float(profile["t3_action_generation_cache_miss"])
    profile["t3_action_generation_cache_hit_rate"] = (
        action_hits / (action_hits + action_misses) if action_hits + action_misses else 0.0
    )

    if active:
        reference_result = decide_hu_turn3_stage3_reference_batch(
            [prepared.state for prepared in active],
            models.stage3_reference_model,
            effective_batch_size,
            cache=reference_cache,
            use_fast_feature_path=config.use_stage3_feature_fast_path,
            encoder_mode=config.stage3_feature_encoder_mode,
            state_feature_cache=state_feature_cache,
            precomputed_actions_by_state=[prepared.actions for prepared in active],
            state_keys=[prepared.key for prepared in active],
            replay_path=config.dump_stage3_feature_replay,
            replay_sample_limit=config.stage3_feature_replay_sample_limit,
            replay_metadata={
                "stage3_reference_model_path": config.stage3_feature_replay_model_path,
                "stage7_model_path": config.stage3_feature_replay_stage7_model_path,
                "hu_turn3_min_margin": config.hu_turn3_min_margin,
                "hu_turn3_reference_min_margin": config.hu_turn3_reference_min_margin,
                "stage7_mode": "selective_override_m5_r10",
                "replay_source": config.stage3_feature_replay_source,
                "source_teacher_run_hash": config.stage3_feature_replay_teacher_run_hash,
                "raw_t3_states": raw_count,
                "unique_t3_states": len(active),
            },
        )
        _merge_profile(profile, reference_result.profile)
        for metadata_key in ("stage3_feature_mode", "feature_dtype"):
            if metadata_key in reference_result.profile:
                profile[metadata_key] = reference_result.profile[metadata_key]  # type: ignore[assignment]
        profile["stage3_reference_batch_seconds"] = float(
            reference_result.profile.get("stage3_reference_total_seconds", 0.0)
        )
        profile["stage3_reference_feature_seconds"] = float(
            reference_result.profile.get("stage3_feature_generation_time", 0.0)
        )
        profile["stage3_reference_model_inference_seconds"] = float(
            reference_result.profile.get("stage3_hgb_predict_time", 0.0)
        )
        for prepared, reference_decision in zip(active, reference_result.decisions):
            prepared.fallback_index = reference_decision.action_index
            prepared.stage3_index = reference_decision.action_index
            prepared.reference_index = reference_decision.action_index
            prepared.reference_margin = reference_decision.reference_margin
            prepared.reference_score = reference_decision.score
            if reference_decision.action_index is None:
                prepared.no_override_reason = reference_decision.fallback_reason or "model_load_failed"
            else:
                profile["stage3_fallback_reuse_count"] = int(profile["stage3_fallback_reuse_count"]) + 1
                profile["duplicate_stage3_compute_avoided_count"] = (
                    int(profile["duplicate_stage3_compute_avoided_count"]) + 1
                )

        fallback_needed = [
            prepared
            for prepared in active
            if prepared.stage3_index is None and prepared.actions
        ]
        if fallback_needed:
            fallback_started_at = time.perf_counter()
            fallback_samples: list[dict[str, Any] | None] = []
            fallback_feature_started_at = time.perf_counter()
            for prepared in fallback_needed:
                try:
                    prepared.fallback_sample = policy_sample(
                        prepared.state.board,
                        prepared.state.dealt_cards,
                        prepared.actions,
                    )
                except Exception:
                    prepared.fallback_sample = None
                fallback_samples.append(prepared.fallback_sample)
            profile["stage3_fallback_feature_seconds"] = time.perf_counter() - fallback_feature_started_at
            fallback_result = _predict_samples_batched(
                models.fallback_turn3_model,
                fallback_samples,
                self_sample_to_matrix,
                effective_batch_size,
            )
            profile["stage3_fallback_batch_seconds"] = time.perf_counter() - fallback_started_at
            profile["stage3_fallback_feature_seconds"] = (
                float(profile["stage3_fallback_feature_seconds"]) + fallback_result.feature_seconds
            )
            profile["stage3_fallback_model_inference_seconds"] = fallback_result.inference_seconds
            profile["stage3_fallback_recomputed_count"] = len(fallback_needed)
            for prepared, predictions in zip(fallback_needed, fallback_result.predictions):
                fallback_index = _safe_argmax(predictions, len(prepared.actions))
                if fallback_index is None:
                    prepared.no_override_reason = prepared.no_override_reason or "model_load_failed"
                    prepared.stage3_index = 0
                    prepared.fallback_index = 0
                else:
                    prepared.no_override_reason = ""
                    prepared.fallback_index = fallback_index
                    prepared.stage3_index = fallback_index

    reference_candidates = [
        prepared
        for prepared in active
        if prepared.no_override_reason == "" and prepared.fallback_index is not None
    ]
    for prepared in reference_candidates:
        if (prepared.reference_margin or 0.0) < config.hu_turn3_reference_min_margin:
            prepared.no_override_reason = "below_reference_margin"
            profile["skipped_by_reference_margin"] = int(profile["skipped_by_reference_margin"]) + 1

    stage7_candidates = [
        prepared
        for prepared in reference_candidates
        if prepared.no_override_reason == ""
        and prepared.stage3_index is not None
        and config.stage7_enabled
        and models.stage7_model is not None
    ]
    if not config.stage7_enabled:
        for prepared in reference_candidates:
            if prepared.no_override_reason == "":
                prepared.no_override_reason = "stage7_disabled"
    elif models.stage7_model is None:
        for prepared in reference_candidates:
            if prepared.no_override_reason == "":
                prepared.no_override_reason = "model_load_failed"

    if stage7_candidates:
        sample_started_at = time.perf_counter()
        stage7_ready: list[_PreparedState] = []
        stage7_samples: list[dict[str, Any] | None] = []
        for prepared in stage7_candidates:
            try:
                prepared.hu_sample = hu_policy_sample(
                    prepared.state.board,
                    prepared.state.dealt_cards,
                    prepared.actions,
                    opponent_board=prepared.state.opponent_board,
                    dead_cards=prepared.state.dead_cards,
                    seat=prepared.state.seat,
                    to_act_order=_to_act_order(prepared.state),
                )
            except Exception:
                prepared.no_override_reason = "feature_failed"
                prepared.hu_sample = None
                continue
            stage7_ready.append(prepared)
            stage7_samples.append(prepared.hu_sample)
        profile["stage7_sample_generation_seconds"] = time.perf_counter() - sample_started_at
    else:
        stage7_ready = []
        stage7_samples = []

    if stage7_ready:
        stage7_result = _predict_samples_batched(
            models.stage7_model,
            stage7_samples,
            hu_sample_to_matrix,
            effective_batch_size,
        )
        profile["stage7_feature_batch_seconds"] = stage7_result.feature_seconds
        profile["stage7_model_inference_seconds"] = stage7_result.inference_seconds
        profile["stage7_model_called_count"] = len(stage7_ready)
        gate_started_at = time.perf_counter()
        for prepared, predictions in zip(stage7_ready, stage7_result.predictions):
            prediction_issue = _prediction_issue(predictions, len(prepared.actions))
            if prediction_issue:
                prepared.no_override_reason = prediction_issue
                continue
            action_index = _safe_argmax(predictions, len(prepared.actions))
            if action_index is None:
                prepared.no_override_reason = "nan_prediction"
                continue
            prepared.stage7_index = action_index
            prepared.legality_check_result = _candidate_legality_result(
                prepared.state.board,
                prepared.actions,
                action_index,
            )
            prepared.stage7_predicted_margin = float(predictions[action_index]) - float(
                predictions[prepared.stage3_index]  # type: ignore[index]
            )
            prepared.model_score = float(predictions[action_index])
            if prepared.legality_check_result != "legal":
                prepared.no_override_reason = "illegal_candidate"
            elif action_index == prepared.stage3_index:
                prepared.no_override_reason = "same_as_stage3"
            elif prepared.stage7_predicted_margin < config.hu_turn3_min_margin:
                prepared.no_override_reason = "below_stage7_margin"
        profile["t3_postprocess_gate_seconds"] = time.perf_counter() - gate_started_at

    for unique_index, prepared in enumerate(unique):
        if any(resolved[index] is not None for index in inverse.get(unique_index, [])):
            continue
        fallback_index = prepared.fallback_index
        stage3_index = prepared.stage3_index if prepared.stage3_index is not None else fallback_index
        if prepared.no_override_reason == "" and prepared.stage7_index is None:
            prepared.no_override_reason = "fallback_to_stage3"
        override_fired = prepared.no_override_reason == ""
        final_index = prepared.stage7_index if override_fired else stage3_index
        decision = _decision_from_indices(
            prepared,
            config=config,
            fallback_index=fallback_index,
            stage3_index=stage3_index,
            stage7_index=prepared.stage7_index,
            final_index=final_index,
            override_fired=override_fired,
            no_override_reason=prepared.no_override_reason,
        )
        _store_prepared_decision(
            prepared,
            decision,
            resolved,
            inverse,
            unique_index,
            cache if cache_enabled else None,
        )

    decisions = [decision for decision in resolved if decision is not None]
    total_seconds = time.perf_counter() - started_at
    profile["t3_continuation_total_seconds"] = total_seconds
    profile["ms_per_raw_t3_decision"] = (total_seconds * 1000.0 / raw_count) if raw_count else 0.0
    unique_count = int(profile["unique_t3_states"])
    profile["ms_per_unique_t3_state"] = (total_seconds * 1000.0 / unique_count) if unique_count else 0.0
    return HuTurn3BatchResult(decisions=decisions, profile=profile)


def decide_hu_turn3_stage3_reference_batch(
    t3_states: Sequence[HuTurn3State],
    stage3_model: object | None,
    batch_size: int = 8192,
    *,
    cache: HuTurn3Stage3ReferenceCache | None = None,
    use_fast_feature_path: bool = True,
    encoder_mode: str = "scalar_fast",
    state_feature_cache: Stage3StateFeatureCache | None = None,
    precomputed_actions_by_state: Sequence[list[Action]] | None = None,
    state_keys: Sequence[tuple[Any, ...]] | None = None,
    replay_path: str | None = None,
    replay_sample_limit: int = 0,
    replay_metadata: dict[str, Any] | None = None,
) -> HuTurn3Stage3ReferenceBatchResult:
    """Batch the HU Stage3 reference/default policy for Turn3 states."""

    started_at = time.perf_counter()
    raw_count = len(t3_states)
    profile: dict[str, float | int] = {
        "stage3_reference_total_seconds": 0.0,
        "stage3_reference_computed_count": 0,
        "stage3_feature_rows": 0,
        "stage3_feature_generation_time": 0.0,
        "stage3_hgb_predict_calls": 0,
        "stage3_hgb_predict_time": 0.0,
        "stage3_feature_rows_per_second": 0.0,
        "stage3_predict_rows_per_second": 0.0,
        "stage3_feature_mode": "fast" if use_fast_feature_path else "scalar",
        "stage3_unique_states": 0,
        "stage3_total_legal_actions": 0,
        "stage3_feature_generation_total": 0.0,
        "stage3_state_feature_time": 0.0,
        "stage3_action_feature_time": 0.0,
        "stage3_matrix_assembly_time": 0.0,
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
        "stage3_state_feature_cache_hit": 0,
        "stage3_state_feature_cache_miss": 0,
        "stage3_state_feature_cache_hit_rate": 0.0,
        "feature_column_count": 0,
        "direct_column_count": 0,
        "scalar_fallback_column_count": 0,
        "direct_column_coverage_ratio": 0.0,
        "memory_peak_mb": 0.0,
        "stage3_reference_cache_hit": 0,
        "stage3_reference_cache_miss": 0,
        "stage3_reference_cache_hit_rate": 0.0,
        "stage3_reference_action_generation_seconds": 0.0,
        "stage3_reference_sample_generation_seconds": 0.0,
    }
    if raw_count == 0:
        return HuTurn3Stage3ReferenceBatchResult(decisions=[], profile=profile)

    resolved: list[Stage3ReferenceDecision | None] = [None] * raw_count
    unique_states: list[HuTurn3State] = []
    unique_keys: list[tuple[Any, ...]] = []
    unique_precomputed_actions: list[list[Action] | None] = []
    first_by_key: dict[tuple[Any, ...], int] = {}
    inverse: dict[int, list[int]] = {}
    cache_hits_before = cache.hits if cache is not None else 0
    cache_misses_before = cache.misses if cache is not None else 0
    for raw_index, state in enumerate(t3_states):
        key = state_keys[raw_index] if state_keys is not None else canonical_t3_state_key(state)
        if cache is not None:
            cached = cache.get(key)
            if cached is not None:
                resolved[raw_index] = cached
                continue
        unique_index = first_by_key.get(key)
        if unique_index is None:
            unique_index = len(unique_states)
            first_by_key[key] = unique_index
            unique_states.append(state)
            unique_keys.append(key)
            unique_precomputed_actions.append(
                precomputed_actions_by_state[raw_index]
                if precomputed_actions_by_state is not None
                else None
            )
        inverse.setdefault(unique_index, []).append(raw_index)

    profile["stage3_reference_cache_hit"] = (cache.hits - cache_hits_before) if cache is not None else 0
    profile["stage3_reference_cache_miss"] = (
        cache.misses - cache_misses_before if cache is not None else len(unique_states)
    )

    samples: list[dict[str, Any] | None] = []
    actions_by_unique: list[list[Action]] = []
    keys_by_unique: list[tuple[Any, ...]] = []
    fast_stage3 = (
        use_fast_feature_path
        and stage3_model is not None
        and hasattr(stage3_model, "predict_matrix")
    )
    action_generation_seconds = 0.0
    sample_generation_seconds = 0.0
    for state_index, state in enumerate(unique_states):
        key = unique_keys[state_index]
        keys_by_unique.append(key)
        actions = unique_precomputed_actions[state_index]
        if actions is None:
            action_started_at = time.perf_counter()
            try:
                actions = generate_turn_actions(state.board, state.dealt_cards)
            except Exception:
                actions = []
            action_generation_seconds += time.perf_counter() - action_started_at
        actions_by_unique.append(actions)
        if not actions:
            samples.append(None)
            continue
        if fast_stage3:
            samples.append(None)
        else:
            sample_started_at = time.perf_counter()
            try:
                samples.append(
                    hu_policy_sample(
                        state.board,
                        state.dealt_cards,
                        actions,
                        opponent_board=state.opponent_board,
                        dead_cards=state.dead_cards,
                        seat=state.seat,
                        to_act_order=_to_act_order(state),
                    )
                )
            except Exception:
                samples.append(None)
            sample_generation_seconds += time.perf_counter() - sample_started_at
    profile["stage3_reference_action_generation_seconds"] = action_generation_seconds
    profile["stage3_reference_sample_generation_seconds"] = sample_generation_seconds

    if fast_stage3:
        prediction_result = _predict_stage3_reference_fast(
            stage3_model,
            unique_states,
            actions_by_unique,
            keys_by_unique,
            batch_size=batch_size,
            state_feature_cache=state_feature_cache,
            encoder_mode=encoder_mode,
        )
        profile.update(
            {
                key: value
                for key, value in prediction_result.profile.items()
                if isinstance(value, (int, float, str))
            }
        )
    else:
        profile["stage3_feature_mode"] = "scalar"
        prediction_result = _predict_samples_batched(
            stage3_model,
            samples,
            hu_sample_to_matrix,
            batch_size,
        )
    profile["stage3_reference_computed_count"] = len(unique_states)
    profile["stage3_unique_states"] = len(unique_states)
    profile["stage3_total_legal_actions"] = sum(len(actions) for actions in actions_by_unique)
    profile["stage3_feature_rows"] = prediction_result.feature_rows
    profile["stage3_feature_generation_time"] = prediction_result.feature_seconds
    profile["stage3_feature_generation_total"] = prediction_result.feature_seconds
    profile["stage3_hgb_predict_calls"] = prediction_result.predict_calls
    profile["stage3_hgb_predict_time"] = prediction_result.inference_seconds
    if prediction_result.feature_seconds > 0.0:
        profile["stage3_feature_rows_per_second"] = (
            prediction_result.feature_rows / prediction_result.feature_seconds
        )
    if prediction_result.inference_seconds > 0.0:
        profile["stage3_predict_rows_per_second"] = (
            prediction_result.feature_rows / prediction_result.inference_seconds
        )

    unique_decisions: list[Stage3ReferenceDecision] = []
    for unique_index, (state, actions, predictions, key) in enumerate(
        zip(unique_states, actions_by_unique, prediction_result.predictions, keys_by_unique)
    ):
        decision = _stage3_reference_decision_from_predictions(actions, predictions)
        unique_decisions.append(decision)
        for raw_index in inverse.get(unique_index, []):
            resolved[raw_index] = decision
        if cache is not None:
            cache.put(key, decision)

    if replay_path:
        try:
            from .hu_turn3_stage3_feature_replay import append_stage3_feature_replay

            append_stage3_feature_replay(
                replay_path,
                states=unique_states,
                actions_by_state=actions_by_unique,
                state_keys=keys_by_unique,
                decisions=unique_decisions,
                predictions=prediction_result.predictions,
                feature_schema_version=FEATURE_SCHEMA_VERSION,
                feature_column_names=[f"f{index}" for index in range(HU_FEATURE_DIM)],
                feature_dtype=str(prediction_result.profile.get("feature_dtype", "float32")),
                profile=profile,
                metadata=replay_metadata or {},
                sample_limit=replay_sample_limit,
            )
        except Exception:
            profile["stage3_feature_replay_failed"] = 1

    decisions = [decision for decision in resolved if decision is not None]
    profile["stage3_reference_total_seconds"] = time.perf_counter() - started_at
    cache_hits = float(profile["stage3_reference_cache_hit"])
    cache_misses = float(profile["stage3_reference_cache_miss"])
    profile["stage3_reference_cache_hit_rate"] = (
        cache_hits / (cache_hits + cache_misses) if cache_hits + cache_misses else 0.0
    )
    return HuTurn3Stage3ReferenceBatchResult(decisions=decisions, profile=profile)


def canonical_t3_state_key(state: HuTurn3State) -> tuple[Any, ...]:
    return (
        "regular_hu_t3_stage7_m5_r10",
        "T3",
        _canonical_board(state.board),
        _canonical_board(state.opponent_board),
        tuple(_sort_cards(state.dealt_cards)),
        tuple(_sort_cards(state.dead_cards)),
        state.seat,
        _to_act_order(state),
        "regular_ofc_v1",
    )


def _predict_stage3_reference_fast(
    model: object | None,
    states: Sequence[HuTurn3State],
    actions_by_state: Sequence[list[Action]],
    state_keys: Sequence[tuple[Any, ...]],
    *,
    batch_size: int,
    state_feature_cache: Stage3StateFeatureCache | None,
    encoder_mode: str = "scalar_fast",
) -> _PredictionBatchResult:
    if model is None or not hasattr(model, "predict_matrix"):
        return _PredictionBatchResult(
            predictions=[None for _state in states],
            feature_seconds=0.0,
            inference_seconds=0.0,
            feature_rows=0,
            predict_calls=0,
            profile={"stage3_feature_mode": "scalar_fallback"},
        )
    feature_batch = build_hu_turn3_stage3_feature_matrix_batch(
        states,
        actions_by_state,
        FEATURE_SCHEMA_VERSION,
        state_keys=state_keys,
        state_feature_cache=state_feature_cache,
        include_action_encodings=False,
        encoder_mode=encoder_mode,
    )
    if feature_batch.X.shape[1] <= 0:
        return _PredictionBatchResult(
            predictions=[None for _state in states],
            feature_seconds=float(feature_batch.profile.get("stage3_feature_generation_total", 0.0)),
            inference_seconds=0.0,
            feature_rows=0,
            predict_calls=0,
            profile=feature_batch.profile,
        )
    expected_features = getattr(getattr(model, "feature_mean", None), "shape", (None,))[0]
    if expected_features is not None and int(expected_features) != int(feature_batch.X.shape[1]):
        raise ValueError(
            f"Stage3 feature column count mismatch: model={expected_features} fast={feature_batch.X.shape[1]}"
        )

    inference_seconds = 0.0
    predict_calls = 0
    row_predictions: list[np.ndarray] = []
    for start in range(0, feature_batch.X.shape[0], batch_size):
        end = min(start + batch_size, feature_batch.X.shape[0])
        inference_started_at = time.perf_counter()
        row_predictions.append(
            np.asarray(model.predict_matrix(feature_batch.X[start:end]), dtype=np.float64)
        )
        inference_seconds += time.perf_counter() - inference_started_at
        predict_calls += 1
    flat_predictions = (
        np.concatenate(row_predictions) if row_predictions else np.zeros(0, dtype=np.float64)
    )
    predictions: list[np.ndarray | None] = [
        np.full(len(actions), np.nan, dtype=np.float64) for actions in actions_by_state
    ]
    for row_index, score in enumerate(flat_predictions):
        state_index = int(feature_batch.row_to_state_index[row_index])
        action_index = int(feature_batch.row_to_action_index[row_index])
        predictions[state_index][action_index] = float(score)  # type: ignore[index]
    for state_index, actions in enumerate(actions_by_state):
        if not actions or predictions[state_index] is None:
            predictions[state_index] = None
        elif np.any(~np.isfinite(predictions[state_index])):  # type: ignore[arg-type]
            predictions[state_index] = None

    profile = dict(feature_batch.profile)
    profile["stage3_hgb_predict_time"] = inference_seconds
    profile["stage3_hgb_predict_calls"] = predict_calls
    profile["stage3_predict_rows_per_second"] = (
        feature_batch.X.shape[0] / inference_seconds if inference_seconds else 0.0
    )
    return _PredictionBatchResult(
        predictions=predictions,
        feature_seconds=float(feature_batch.profile.get("stage3_feature_generation_total", 0.0)),
        inference_seconds=inference_seconds,
        feature_rows=int(feature_batch.X.shape[0]),
        predict_calls=predict_calls,
        profile=profile,
    )


def _canonical_board(board: Board) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    return (
        tuple(_sort_cards(board.top)),
        tuple(_sort_cards(board.middle)),
        tuple(_sort_cards(board.bottom)),
    )


def _sort_cards(cards: Iterable[str]) -> list[str]:
    return sorted(cards, key=lambda card: CARD_ORDER.get(card, 999))


def _merge_profile(target: dict[str, float | int], source: dict[str, float | int]) -> None:
    for key, value in source.items():
        if isinstance(value, (int, float)):
            target[key] = target.get(key, 0.0) + value


def _stage3_reference_decision_from_predictions(
    actions: list[Action],
    predictions: np.ndarray | None,
) -> Stage3ReferenceDecision:
    issue = _prediction_issue(predictions, len(actions))
    if issue:
        return Stage3ReferenceDecision(
            stage3_action=None,
            action_index=None,
            reference_margin=None,
            score=None,
            rank_score=None,
            action_count=len(actions),
            legality_status="not_evaluated",
            fallback_reason=issue,
        )
    values = [float(value) for value in predictions]  # type: ignore[union-attr]
    action_index = int(max(range(len(values)), key=lambda index: values[index]))
    best = values[action_index]
    if len(values) <= 1:
        margin = 0.0
    else:
        second = max(value for index, value in enumerate(values) if index != action_index)
        margin = best - second
    return Stage3ReferenceDecision(
        stage3_action=actions[action_index],
        action_index=action_index,
        reference_margin=margin,
        score=best,
        rank_score=best,
        action_count=len(actions),
        legality_status="legal",
        fallback_reason="",
    )


def _to_act_order(state: HuTurn3State) -> str:
    if state.to_act_order:
        return state.to_act_order
    return "second" if state.opponent_board.card_count() > state.board.card_count() else "first"


def _predict_samples_batched(
    model: object | None,
    samples: Sequence[dict[str, Any] | None],
    matrix_builder: Any,
    batch_size: int,
) -> _PredictionBatchResult:
    if model is None:
        return _PredictionBatchResult(
            predictions=[None for _sample in samples],
            feature_seconds=0.0,
            inference_seconds=0.0,
            feature_rows=0,
            predict_calls=0,
        )
    if not hasattr(model, "predict_matrix"):
        inference_started_at = time.perf_counter()
        predictions = [_predict_single(model, sample) for sample in samples]
        return _PredictionBatchResult(
            predictions=predictions,
            feature_seconds=0.0,
            inference_seconds=time.perf_counter() - inference_started_at,
            feature_rows=sum(len(sample.get("actions", ())) for sample in samples if sample is not None),
            predict_calls=sum(1 for sample in samples if sample is not None),
        )

    outputs: list[np.ndarray | None] = [None] * len(samples)
    feature_blocks: list[np.ndarray] = []
    output_indices: list[int] = []
    block_sizes: list[int] = []
    row_count = 0
    feature_seconds = 0.0
    inference_seconds = 0.0
    feature_rows = 0
    predict_calls = 0

    def flush() -> None:
        nonlocal feature_blocks, output_indices, block_sizes, row_count, inference_seconds, predict_calls
        if not feature_blocks:
            return
        inference_started_at = time.perf_counter()
        try:
            predict_calls += 1
            predictions = np.asarray(model.predict_matrix(np.vstack(feature_blocks)), dtype=np.float64)
        except Exception:
            inference_seconds += time.perf_counter() - inference_started_at
            for output_index in output_indices:
                outputs[output_index] = _predict_single(model, samples[output_index])
            feature_blocks = []
            output_indices = []
            block_sizes = []
            row_count = 0
            return
        inference_seconds += time.perf_counter() - inference_started_at
        cursor = 0
        for output_index, block_size in zip(output_indices, block_sizes):
            outputs[output_index] = predictions[cursor : cursor + block_size]
            cursor += block_size
        feature_blocks = []
        output_indices = []
        block_sizes = []
        row_count = 0

    for sample_index, sample in enumerate(samples):
        if sample is None:
            continue
        try:
            feature_started_at = time.perf_counter()
            features, _targets = matrix_builder(sample)
            feature_seconds += time.perf_counter() - feature_started_at
        except Exception:
            outputs[sample_index] = None
            continue
        feature_rows += int(features.shape[0])
        if row_count and row_count + features.shape[0] > batch_size:
            flush()
        feature_blocks.append(features)
        output_indices.append(sample_index)
        block_sizes.append(features.shape[0])
        row_count += int(features.shape[0])
        if row_count >= batch_size:
            flush()
    flush()
    return _PredictionBatchResult(
        predictions=outputs,
        feature_seconds=feature_seconds,
        inference_seconds=inference_seconds,
        feature_rows=feature_rows,
        predict_calls=predict_calls,
    )


def _predict_single(model: object, sample: dict[str, Any] | None) -> np.ndarray | None:
    if sample is None or not hasattr(model, "predict_sample"):
        return None
    try:
        return np.asarray(model.predict_sample(sample), dtype=np.float64)
    except Exception:
        return None


def _safe_argmax(predictions: np.ndarray | None, expected_len: int) -> int | None:
    if _prediction_issue(predictions, expected_len):
        return None
    values = [float(value) for value in predictions]
    return int(max(range(len(values)), key=lambda index: values[index]))


def _prediction_issue(predictions: np.ndarray | None, expected_len: int) -> str:
    if predictions is None:
        return "model_load_failed"
    if len(predictions) != expected_len:
        return "illegal_candidate"
    values = [float(value) for value in predictions]
    if not values:
        return "model_load_failed"
    if any(not math.isfinite(value) for value in values):
        return "nan_prediction"
    return ""


def _candidate_legality_result(board: Board, actions: list[Action], action_index: int) -> str:
    if action_index < 0 or action_index >= len(actions):
        return "illegal_candidate"
    try:
        board.place(actions[action_index].placements)
    except Exception:
        return "illegal_candidate"
    return "legal"


def _decision_from_indices(
    prepared: _PreparedState,
    *,
    config: HuTurn3Stage7BatchConfig,
    fallback_index: int | None,
    stage3_index: int | None,
    stage7_index: int | None,
    final_index: int | None,
    override_fired: bool,
    no_override_reason: str,
) -> Turn3Decision:
    actions = prepared.actions
    fallback_action = actions[fallback_index] if fallback_index is not None and 0 <= fallback_index < len(actions) else None
    stage3_action = actions[stage3_index] if stage3_index is not None and 0 <= stage3_index < len(actions) else None
    stage7_action = actions[stage7_index] if stage7_index is not None and 0 <= stage7_index < len(actions) else None
    final_action = actions[final_index] if final_index is not None and 0 <= final_index < len(actions) else None
    record = {
        "hand_id": prepared.state.hand_id,
        "game_id": prepared.state.game_id,
        "seed": prepared.state.decision_seed,
        "street": prepared.state.street,
        "turn": prepared.state.street,
        "seat": prepared.state.seat,
        "hero_board": board_to_json(prepared.state.board),
        "opponent_board": board_to_json(prepared.state.opponent_board),
        "cards_to_place": list(prepared.state.dealt_cards),
        "stage3_action": action_to_json(prepared.state.board, stage3_action) if stage3_action is not None else None,
        "stage7_action": action_to_json(prepared.state.board, stage7_action) if stage7_action is not None else None,
        "fallback_action": action_to_json(prepared.state.board, fallback_action) if fallback_action is not None else None,
        "final_action": action_to_json(prepared.state.board, final_action) if final_action is not None else None,
        "override_fired": override_fired,
        "no_override_reason": no_override_reason or "",
        "hu_turn3_min_margin": config.hu_turn3_min_margin,
        "hu_turn3_reference_min_margin": config.hu_turn3_reference_min_margin,
        "stage7_predicted_margin": prepared.stage7_predicted_margin,
        "reference_margin": prepared.reference_margin,
        "model_score": prepared.model_score,
        "ev": prepared.model_score,
        "rank_score": prepared.model_score,
        "reference_score": prepared.reference_score,
        "legality_check_result": prepared.legality_check_result,
        "runtime_latency_ms": 0.0,
    }
    return Turn3Decision(
        final_action=final_action,
        fallback_action=fallback_action,
        stage3_action=stage3_action,
        stage7_action=stage7_action,
        override_fired=override_fired,
        no_override_reason=no_override_reason or "",
        stage7_predicted_margin=prepared.stage7_predicted_margin,
        reference_margin=prepared.reference_margin,
        model_score=prepared.model_score,
        reference_score=prepared.reference_score,
        legality_check_result=prepared.legality_check_result,
        action_count=len(actions),
        record=record,
    )


def _store_prepared_decision(
    prepared: _PreparedState,
    decision: Turn3Decision,
    resolved: list[Turn3Decision | None],
    inverse: dict[int, list[int]],
    unique_index: int,
    cache: HuTurn3DecisionCache | None,
) -> None:
    for raw_index in inverse.get(unique_index, []):
        resolved[raw_index] = decision
    if cache is not None:
        cache.put(prepared.key, decision)
