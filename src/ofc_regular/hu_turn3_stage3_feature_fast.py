"""Fast HU Turn3 Stage3/reference feature matrix assembly."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .action_space import Action
from .action_key import (
    ACTION_KEY_SCHEMA,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from .hu_turn3_model import (
    DISCARD_OFFSET,
    GLOBAL_OFFSET,
    HU_FEATURE_DIM,
    HU_DERIVED_OFFSET,
    HU_MATCHUP_OFFSET,
    NEXT_OFFSET,
    PLACEMENT_OFFSET,
    ROWS,
    ROW_CAPACITY,
    SELF_FEATURE_DIM,
    _build_sample_encode_context,
    _card_count,
    _complete_row_summary_cached,
    _complete_row_summary,
    _encode_self_board,
    _encode_self_cards,
    _encode_self_extra_stats,
    _encode_self_row_stats,
    _opponent_needed_top_ranks,
    _row_matchup_summary,
    _row_matchup_summary_cached,
    _set_self_card_row,
    _top_summary_cached,
    _top_summary,
    _write_top_summary,
    hu_policy_sample,
    sample_to_matrix as scalar_hu_sample_to_matrix,
)

FEATURE_SCHEMA_VERSION = "hu_turn3_stage3_fast_v1"
STATE_FEATURE_CACHE_SCHEMA = "hu_turn3_stage3_state_feature_cache_v2"


@dataclass
class FeatureBatch:
    X: np.ndarray
    row_to_state_index: np.ndarray
    row_to_action_index: np.ndarray
    action_encodings: list[dict[str, Any]]
    feature_column_names: list[str]
    feature_schema_version: str
    debug_sample_rows: list[dict[str, Any]] = field(default_factory=list)
    profile: dict[str, float | int | str] = field(default_factory=dict)


@dataclass
class Stage3StateFeatureCache:
    max_size: int = 200_000
    _items: dict[tuple[Any, ...], tuple[dict[str, Any], dict[str, Any]]] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def get(self, key: tuple[Any, ...]) -> tuple[dict[str, Any], dict[str, Any]] | None:
        value = self._items.get(key)
        if value is None:
            self.misses += 1
            return None
        self.hits += 1
        return value

    def put(self, key: tuple[Any, ...], value: tuple[dict[str, Any], dict[str, Any]]) -> None:
        if self.max_size <= 0:
            return
        if len(self._items) >= self.max_size:
            self._items.clear()
        self._items[key] = value


def stage3_state_feature_cache_key(
    state_key: tuple[Any, ...],
    actions: Sequence[Action],
    feature_schema: str,
) -> tuple[Any, ...]:
    return (
        STATE_FEATURE_CACHE_SCHEMA,
        feature_schema,
        ACTION_KEY_SCHEMA,
        legal_action_set_digest(actions),
        ordered_action_mapping_digest(actions),
        state_key,
    )


def build_hu_turn3_stage3_feature_matrix_batch(
    t3_states: Sequence[Any],
    legal_actions_by_state: Sequence[list[Action]],
    feature_schema: str = FEATURE_SCHEMA_VERSION,
    *,
    state_keys: Sequence[tuple[Any, ...]] | None = None,
    state_feature_cache: Stage3StateFeatureCache | None = None,
    debug_sample_limit: int = 0,
    include_action_encodings: bool = True,
    encoder_mode: str = "scalar_fast",
) -> FeatureBatch:
    if feature_schema != FEATURE_SCHEMA_VERSION:
        raise ValueError(f"unsupported feature schema: {feature_schema}")
    if len(t3_states) != len(legal_actions_by_state):
        raise ValueError("states/actions length mismatch")
    if encoder_mode == "rust_direct":
        from .hu_turn3_stage3_feature_rust import (
            build_hu_turn3_stage3_feature_matrix_rust,
        )

        return build_hu_turn3_stage3_feature_matrix_rust(
            t3_states,
            legal_actions_by_state,
            feature_schema,
            state_keys=state_keys,
            state_feature_cache=state_feature_cache,
            debug_sample_limit=debug_sample_limit,
            include_action_encodings=include_action_encodings,
            encoder_mode=encoder_mode,
        )
    if encoder_mode in {"numpy_direct_partial", "numpy_direct_full"}:
        from .hu_turn3_stage3_feature_numpy_direct import (
            build_hu_turn3_stage3_feature_matrix_numpy_direct,
        )

        return build_hu_turn3_stage3_feature_matrix_numpy_direct(
            t3_states,
            legal_actions_by_state,
            feature_schema,
            state_keys=state_keys,
            state_feature_cache=state_feature_cache,
            debug_sample_limit=debug_sample_limit,
            include_action_encodings=include_action_encodings,
            encoder_mode=encoder_mode,
        )
    if encoder_mode != "scalar_fast":
        raise ValueError(f"unsupported encoder mode: {encoder_mode}")

    started_at = time.perf_counter()
    total_rows = sum(len(actions) for actions in legal_actions_by_state)
    X = np.empty((total_rows, HU_FEATURE_DIM), dtype=np.float32)
    row_to_state_index = np.empty(total_rows, dtype=np.int32)
    row_to_action_index = np.empty(total_rows, dtype=np.int16)
    action_encodings: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []

    cache_hits_before = state_feature_cache.hits if state_feature_cache is not None else 0
    cache_misses_before = state_feature_cache.misses if state_feature_cache is not None else 0
    row_cache_before = _row_matchup_summary_cached.cache_info()
    top_cache_before = _top_summary_cached.cache_info()
    complete_cache_before = _complete_row_summary_cached.cache_info()
    state_started_at = time.perf_counter()
    sample_seconds = 0.0
    context_seconds = 0.0
    prepared: list[tuple[dict[str, Any] | None, dict[str, Any] | None]] = []
    for state_index, (state, actions) in enumerate(zip(t3_states, legal_actions_by_state)):
        raw_key = state_keys[state_index] if state_keys is not None else None
        key = (
            stage3_state_feature_cache_key(raw_key, actions, feature_schema)
            if raw_key is not None
            else None
        )
        cached = state_feature_cache.get(key) if key is not None and state_feature_cache is not None else None
        if cached is None:
            try:
                sample_started_at = time.perf_counter()
                sample = hu_policy_sample(
                    state.board,
                    state.dealt_cards,
                    actions,
                    opponent_board=state.opponent_board,
                    dead_cards=state.dead_cards,
                    seat=state.seat,
                    to_act_order=state.to_act_order or "first",
                )
                sample_seconds += time.perf_counter() - sample_started_at
                context_started_at = time.perf_counter()
                context = _build_sample_encode_context(sample, sample["actions"])
                context_seconds += time.perf_counter() - context_started_at
            except Exception:
                sample = None
                context = None
            if key is not None and state_feature_cache is not None and sample is not None and context is not None:
                state_feature_cache.put(key, (sample, context))
        else:
            sample, context = cached
        prepared.append((sample, context))
    state_seconds = time.perf_counter() - state_started_at

    action_started_at = time.perf_counter()
    action_encode_seconds = 0.0
    row = 0
    for state_index, (_state, actions) in enumerate(zip(t3_states, legal_actions_by_state)):
        del actions
        sample, context = prepared[state_index]
        if sample is None or context is None:
            continue
        action_dicts = sample["actions"]
        for action_index, action_dict in enumerate(action_dicts):
            encode_started_at = time.perf_counter()
            _encode_action_into_with_context_fast(X[row], sample, action_dict, context)
            action_encode_seconds += time.perf_counter() - encode_started_at
            row_to_state_index[row] = state_index
            row_to_action_index[row] = action_index
            if debug_sample_limit > 0 and len(debug_rows) < debug_sample_limit:
                debug_rows.append(
                    {
                        "state_index": state_index,
                        "action_index": action_index,
                        "row_sum": float(X[row].sum()),
                    }
                )
            row += 1
        if include_action_encodings:
            action_encodings.extend(action_dicts)
    action_seconds = time.perf_counter() - action_started_at

    if row != total_rows:
        X = X[:row]
        row_to_state_index = row_to_state_index[:row]
        row_to_action_index = row_to_action_index[:row]
    total_seconds = time.perf_counter() - started_at
    row_cache_after = _row_matchup_summary_cached.cache_info()
    top_cache_after = _top_summary_cached.cache_info()
    complete_cache_after = _complete_row_summary_cached.cache_info()
    cache_hits = (
        state_feature_cache.hits - cache_hits_before if state_feature_cache is not None else 0
    )
    cache_misses = (
        state_feature_cache.misses - cache_misses_before if state_feature_cache is not None else len(t3_states)
    )
    return FeatureBatch(
        X=X,
        row_to_state_index=row_to_state_index,
        row_to_action_index=row_to_action_index,
        action_encodings=action_encodings,
        feature_column_names=[f"f{index}" for index in range(HU_FEATURE_DIM)],
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        debug_sample_rows=debug_rows,
        profile={
            "stage3_feature_mode": "fast",
            "stage3_feature_generation_total": total_seconds,
            "stage3_feature_rows": int(X.shape[0]),
            "stage3_state_feature_time": state_seconds,
            "stage3_action_feature_time": action_seconds,
            "stage3_matrix_assembly_time": max(0.0, total_seconds - state_seconds - action_seconds),
            "stage3_sample_generation_time": sample_seconds,
            "stage3_context_generation_time": context_seconds,
            "stage3_row_global_summary_time": context_seconds,
            "stage3_candidate_delta_feature_time": action_encode_seconds,
            "stage3_encoder_matrix_build_seconds": total_seconds,
            "stage3_after_board_construction_seconds": 0.0,
            "stage3_row_summary_seconds": context_seconds,
            "stage3_global_summary_seconds": 0.0,
            "stage3_action_delta_seconds": action_encode_seconds,
            "stage3_scalar_fallback_seconds": total_seconds,
            "stage3_cache_lookup_update_seconds": 0.0,
            "stage3_column_validation_seconds": 0.0,
            "stage3_numpy_allocation_seconds": 0.0,
            "stage3_hgb_input_preparation_seconds": 0.0,
            "stage3_non_encoder_overhead_seconds": max(
                0.0,
                total_seconds - sample_seconds - context_seconds - action_encode_seconds,
            ),
            "stage3_state_feature_cache_hit": cache_hits,
            "stage3_state_feature_cache_miss": cache_misses,
            "stage3_state_feature_cache_hit_rate": (
                cache_hits / (cache_hits + cache_misses) if cache_hits + cache_misses else 0.0
            ),
            "feature_dtype": str(X.dtype),
            "feature_column_count": int(X.shape[1]) if X.ndim == 2 else 0,
            "direct_column_count": 0,
            "scalar_fallback_column_count": int(X.shape[1]) if X.ndim == 2 else 0,
            "direct_column_coverage_ratio": 0.0,
            "memory_peak_mb": float(X.nbytes) / (1024.0 * 1024.0),
            "row_summary_cache_hit": row_cache_after.hits - row_cache_before.hits,
            "row_summary_cache_miss": row_cache_after.misses - row_cache_before.misses,
            "row_summary_cache_hit_rate": _cache_hit_rate(
                row_cache_after.hits - row_cache_before.hits,
                row_cache_after.misses - row_cache_before.misses,
            ),
            "top_summary_cache_hit": top_cache_after.hits - top_cache_before.hits,
            "top_summary_cache_miss": top_cache_after.misses - top_cache_before.misses,
            "top_summary_cache_hit_rate": _cache_hit_rate(
                top_cache_after.hits - top_cache_before.hits,
                top_cache_after.misses - top_cache_before.misses,
            ),
            "complete_row_summary_cache_hit": complete_cache_after.hits - complete_cache_before.hits,
            "complete_row_summary_cache_miss": complete_cache_after.misses - complete_cache_before.misses,
            "complete_row_summary_cache_hit_rate": _cache_hit_rate(
                complete_cache_after.hits - complete_cache_before.hits,
                complete_cache_after.misses - complete_cache_before.misses,
            ),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
        },
    )


def _encode_action_into_with_context_fast(
    vector: np.ndarray,
    sample: dict[str, Any],
    action: dict[str, Any],
    context: dict[str, Any],
) -> None:
    vector[:] = context["common"]
    self_vector = vector[:SELF_FEATURE_DIM]
    self_vector[:] = context["self_common"]

    for card, row in action.get("placements", ()):
        _set_self_card_row(self_vector, card, row, PLACEMENT_OFFSET)
    discards = tuple(action.get("discards", ()))
    _encode_self_cards(self_vector, discards, DISCARD_OFFSET)

    next_board = action.get("next_board", sample["board"])
    _encode_self_board(self_vector, next_board, NEXT_OFFSET)
    _encode_self_row_stats(self_vector, next_board)
    _encode_self_extra_stats(self_vector, next_board)

    opponent_board = context["opponent_board"]
    available_by_rank = context["available_by_rank"]
    opponent_top = context["opponent_top"]
    opponent_rows = context["opponent_rows"]
    opponent_summaries = context["opponent_matchups"]
    opponent_count = int(context["opponent_count"])

    hero_count = _card_count(next_board)
    vector[GLOBAL_OFFSET + 0] = hero_count / 13.0
    vector[GLOBAL_OFFSET + 1] = opponent_count / 13.0
    for row_index, row in enumerate(ROWS):
        hero_open = ROW_CAPACITY[row] - len(next_board.get(row, ()))
        opp_open = ROW_CAPACITY[row] - len(opponent_board.get(row, ()))
        vector[GLOBAL_OFFSET + 2 + row_index] = hero_open / ROW_CAPACITY[row]
        vector[GLOBAL_OFFSET + 5 + row_index] = opp_open / ROW_CAPACITY[row]

    terminal = context["opponent_terminal"]
    if terminal is not None:
        vector[GLOBAL_OFFSET + 8] = 1.0
        vector[GLOBAL_OFFSET + 9] = terminal["busted"]
        vector[GLOBAL_OFFSET + 10] = terminal["royalty"]
        vector[GLOBAL_OFFSET + 11] = terminal["fl_entry"]

    hero_top = _top_summary(next_board.get("top", ()), available_by_rank)
    hero_rows = _complete_row_summary(next_board)
    hero_summaries = {
        row: _row_matchup_summary(row, next_board.get(row, ()), available_by_rank)
        for row in ROWS
    }

    derived_offset = HU_DERIVED_OFFSET
    _write_top_summary(vector, derived_offset, hero_top)
    _write_top_summary(vector, derived_offset + 12, opponent_top)

    vector[derived_offset + 24] = hero_rows["top_royalty"] / 22.0
    vector[derived_offset + 25] = hero_rows["middle_royalty"] / 50.0
    vector[derived_offset + 26] = hero_rows["bottom_royalty"] / 25.0
    vector[derived_offset + 27] = hero_rows["top_middle_order"]
    vector[derived_offset + 28] = hero_rows["middle_bottom_order"]
    vector[derived_offset + 29] = opponent_rows["top_royalty"] / 22.0
    vector[derived_offset + 30] = opponent_rows["middle_royalty"] / 50.0
    vector[derived_offset + 31] = opponent_rows["bottom_royalty"] / 25.0
    vector[derived_offset + 32] = opponent_rows["top_middle_order"]
    vector[derived_offset + 33] = opponent_rows["middle_bottom_order"]

    vector[derived_offset + 34] = hero_top["fl_potential"] - opponent_top["fl_potential"]
    vector[derived_offset + 35] = hero_top["high_pair_rank"] - opponent_top["high_pair_rank"]
    vector[derived_offset + 36] = hero_rows["total_royalty"] / 100.0 - opponent_rows["total_royalty"] / 100.0
    vector[derived_offset + 37] = sum(1 for card in discards if card[0] in "QKA") / max(len(discards), 1)
    vector[derived_offset + 38] = sum(
        1 for card in discards if card[0] in context["opponent_needed_top"]
    ) / max(len(discards), 1)

    for index, rank in enumerate(("Q", "K", "A")):
        vector[derived_offset + 39 + index] = available_by_rank[rank] / 4.0
        vector[derived_offset + 42 + index] = opponent_top[f"needs_{rank.lower()}"]
        vector[derived_offset + 45 + index] = hero_top[f"needs_{rank.lower()}"]

    matchup_offset = HU_MATCHUP_OFFSET
    for row_index, row in enumerate(ROWS):
        hero = hero_summaries[row]
        opponent = opponent_summaries[row]
        row_offset = matchup_offset + row_index * 12
        vector[row_offset + 0] = hero["count"]
        vector[row_offset + 1] = opponent["count"]
        vector[row_offset + 2] = hero["count"] - opponent["count"]
        vector[row_offset + 3] = hero["slots"]
        vector[row_offset + 4] = opponent["slots"]
        vector[row_offset + 5] = hero["category"]
        vector[row_offset + 6] = opponent["category"]
        vector[row_offset + 7] = hero["category"] - opponent["category"]
        vector[row_offset + 8] = hero["royalty"]
        vector[row_offset + 9] = opponent["royalty"]
        vector[row_offset + 10] = hero["total_royalty"] / 100.0 - opponent["total_royalty"] / 100.0
        vector[row_offset + 11] = hero["premium_potential"] - opponent["premium_potential"]

    completed_hero = sum(
        int(len(next_board.get(row, ())) == ROW_CAPACITY[row])
        for row in ROWS
    )
    completed_opponent = sum(
        int(len(opponent_board.get(row, ())) == ROW_CAPACITY[row])
        for row in ROWS
    )
    global_offset = matchup_offset + 36
    vector[global_offset + 0] = hero_rows["total_royalty"] / 100.0 - opponent_rows["total_royalty"] / 100.0
    vector[global_offset + 1] = (completed_hero - completed_opponent) / 3.0
    vector[global_offset + 2] = hero_rows["top_middle_order"]
    vector[global_offset + 3] = hero_rows["middle_bottom_order"]
    vector[global_offset + 4] = opponent_rows["top_middle_order"]
    vector[global_offset + 5] = opponent_rows["middle_bottom_order"]
    vector[global_offset + 6] = hero_rows["top_middle_order"] - opponent_rows["top_middle_order"]
    vector[global_offset + 7] = hero_rows["middle_bottom_order"] - opponent_rows["middle_bottom_order"]
    vector[global_offset + 8] = hero_summaries["top"]["premium_potential"] - opponent_summaries["top"]["premium_potential"]
    vector[global_offset + 9] = hero_summaries["middle"]["premium_potential"] - opponent_summaries["middle"]["premium_potential"]
    vector[global_offset + 10] = hero_summaries["bottom"]["premium_potential"] - opponent_summaries["bottom"]["premium_potential"]
    vector[global_offset + 11] = sum(
        hero_summaries[row]["premium_potential"] - opponent_summaries[row]["premium_potential"]
        for row in ROWS
    ) / 3.0


def assert_fast_feature_parity(sample: dict[str, Any], fast_rows: np.ndarray) -> None:
    scalar_rows, _targets = scalar_hu_sample_to_matrix(sample)
    if scalar_rows.shape != fast_rows.shape:
        raise AssertionError(f"feature shape mismatch: scalar={scalar_rows.shape} fast={fast_rows.shape}")
    if not np.allclose(scalar_rows, fast_rows, rtol=0.0, atol=1e-6, equal_nan=True):
        diff = np.abs(scalar_rows - fast_rows)
        raise AssertionError(f"feature values differ: max_abs_diff={float(np.nanmax(diff))}")


def _cache_hit_rate(hits: int, misses: int) -> float:
    total = hits + misses
    return hits / total if total else 0.0
