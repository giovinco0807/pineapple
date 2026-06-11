"""NumPy direct HU Turn3 Stage3 feature matrix writer spike."""

from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np

from .action_space import Action
from .cards import RANKS
from .evaluator import score_board
from .hu_turn3_model import (
    CARD_INDEX,
    DEAD_OFFSET,
    DISCARD_OFFSET,
    GLOBAL_OFFSET,
    HU_DERIVED_OFFSET,
    HU_FEATURE_DIM,
    HU_MATCHUP_OFFSET,
    NEXT_OFFSET,
    OPPONENT_OFFSET,
    OPP_RANK_COUNT_OFFSET,
    OPP_ROW_LEN_OFFSET,
    OPP_SUIT_COUNT_OFFSET,
    ORDER_OFFSET,
    PLACEMENT_OFFSET,
    RANK_INDEX,
    ROWS,
    ROW_CAPACITY,
    SEAT_INDEX,
    SEAT_OFFSET,
    SELF_FEATURE_DIM,
    _available_by_rank,
    _complete_row_summary,
    _complete_row_summary_cached,
    _opponent_needed_top_ranks,
    _row_matchup_summary,
    _row_matchup_summary_cached,
    _top_summary,
    _top_summary_cached,
    _write_top_summary,
)
from .hu_turn3_stage3_feature_fast import (
    FEATURE_SCHEMA_VERSION,
    FeatureBatch,
    Stage3StateFeatureCache,
    _cache_hit_rate,
)
from .state import Board
from .turn3_model import (
    CURRENT_OFFSET,
    DEALT_OFFSET,
    GLOBAL_EXTRA_OFFSET,
    NEXT_OFFSET as SELF_NEXT_OFFSET,
    RANK_COUNT_OFFSET,
    ROW_EXTRA_DIM,
    ROW_EXTRA_OFFSET,
    ROW_LEN_OFFSET,
    SUIT_COUNT_OFFSET,
    _cards_cache_key,
    _global_extra_values,
    _row_extra_values,
)

ENCODER_MODE_NUMPY_DIRECT_PARTIAL = "numpy_direct_partial"
ENCODER_MODE_NUMPY_DIRECT_FULL = "numpy_direct_full"

SUIT_INDEX = {"h": 0, "d": 1, "c": 2, "s": 3}


def build_hu_turn3_stage3_feature_matrix_numpy_direct(
    t3_states: Sequence[Any],
    legal_actions_by_state: Sequence[list[Action]],
    feature_schema: str = FEATURE_SCHEMA_VERSION,
    *,
    state_keys: Sequence[tuple[Any, ...]] | None = None,
    state_feature_cache: Stage3StateFeatureCache | None = None,
    debug_sample_limit: int = 0,
    include_action_encodings: bool = False,
    encoder_mode: str = ENCODER_MODE_NUMPY_DIRECT_FULL,
) -> FeatureBatch:
    del state_feature_cache, include_action_encodings
    if feature_schema != FEATURE_SCHEMA_VERSION:
        raise ValueError(f"unsupported feature schema: {feature_schema}")
    if len(t3_states) != len(legal_actions_by_state):
        raise ValueError("states/actions length mismatch")
    if encoder_mode not in {ENCODER_MODE_NUMPY_DIRECT_PARTIAL, ENCODER_MODE_NUMPY_DIRECT_FULL}:
        raise ValueError(f"unsupported numpy direct encoder mode: {encoder_mode}")

    started_at = time.perf_counter()
    total_rows = sum(len(actions) for actions in legal_actions_by_state)
    allocation_started_at = time.perf_counter()
    X = np.zeros((total_rows, HU_FEATURE_DIM), dtype=np.float32)
    row_to_state_index = np.empty(total_rows, dtype=np.int32)
    row_to_action_index = np.empty(total_rows, dtype=np.int16)
    allocation_seconds = time.perf_counter() - allocation_started_at
    debug_rows: list[dict[str, Any]] = []
    attribution = {
        "stage3_after_board_construction_seconds": 0.0,
        "stage3_row_summary_seconds": 0.0,
        "stage3_global_summary_seconds": 0.0,
        "stage3_action_delta_seconds": 0.0,
    }

    row_cache_before = _row_matchup_summary_cached.cache_info()
    top_cache_before = _top_summary_cached.cache_info()
    complete_cache_before = _complete_row_summary_cached.cache_info()

    state_started_at = time.perf_counter()
    state_contexts = [
        _build_direct_state_context(state, actions)
        for state, actions in zip(t3_states, legal_actions_by_state)
    ]
    state_seconds = time.perf_counter() - state_started_at

    action_started_at = time.perf_counter()
    row = 0
    for state_index, (state, actions, context) in enumerate(
        zip(t3_states, legal_actions_by_state, state_contexts)
    ):
        for action_index, action in enumerate(actions):
            vector = X[row]
            vector[:] = context["common"]
            _encode_action_direct(vector, state, action, context, attribution)
            row_to_state_index[row] = state_index
            row_to_action_index[row] = action_index
            if debug_sample_limit > 0 and len(debug_rows) < debug_sample_limit:
                debug_rows.append(
                    {
                        "state_index": state_index,
                        "action_index": action_index,
                        "row_sum": float(vector.sum()),
                    }
                )
            row += 1
    action_seconds = time.perf_counter() - action_started_at

    total_seconds = time.perf_counter() - started_at
    row_cache_after = _row_matchup_summary_cached.cache_info()
    top_cache_after = _top_summary_cached.cache_info()
    complete_cache_after = _complete_row_summary_cached.cache_info()
    non_encoder_seconds = max(
        0.0,
        total_seconds
        - allocation_seconds
        - attribution["stage3_after_board_construction_seconds"]
        - attribution["stage3_row_summary_seconds"]
        - attribution["stage3_global_summary_seconds"]
        - attribution["stage3_action_delta_seconds"],
    )
    return FeatureBatch(
        X=X,
        row_to_state_index=row_to_state_index,
        row_to_action_index=row_to_action_index,
        action_encodings=[],
        feature_column_names=[f"f{index}" for index in range(HU_FEATURE_DIM)],
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        debug_sample_rows=debug_rows,
        profile={
            "stage3_feature_mode": encoder_mode,
            "stage3_feature_generation_total": total_seconds,
            "stage3_feature_rows": int(X.shape[0]),
            "stage3_state_feature_time": state_seconds,
            "stage3_action_feature_time": action_seconds,
            "stage3_matrix_assembly_time": max(0.0, total_seconds - state_seconds - action_seconds),
            "stage3_sample_generation_time": 0.0,
            "stage3_context_generation_time": state_seconds,
            "stage3_row_global_summary_time": state_seconds,
            "stage3_candidate_delta_feature_time": action_seconds,
            "stage3_encoder_matrix_build_seconds": total_seconds,
            "stage3_after_board_construction_seconds": attribution["stage3_after_board_construction_seconds"],
            "stage3_row_summary_seconds": attribution["stage3_row_summary_seconds"],
            "stage3_global_summary_seconds": attribution["stage3_global_summary_seconds"],
            "stage3_action_delta_seconds": attribution["stage3_action_delta_seconds"],
            "stage3_scalar_fallback_seconds": 0.0,
            "stage3_cache_lookup_update_seconds": 0.0,
            "stage3_column_validation_seconds": 0.0,
            "stage3_numpy_allocation_seconds": allocation_seconds,
            "stage3_hgb_input_preparation_seconds": 0.0,
            "stage3_non_encoder_overhead_seconds": non_encoder_seconds,
            "stage3_state_feature_cache_hit": 0,
            "stage3_state_feature_cache_miss": len(t3_states),
            "stage3_state_feature_cache_hit_rate": 0.0,
            "feature_dtype": str(X.dtype),
            "feature_column_count": int(X.shape[1]) if X.ndim == 2 else 0,
            "direct_column_count": int(X.shape[1]) if X.ndim == 2 else 0,
            "scalar_fallback_column_count": 0,
            "direct_column_coverage_ratio": 1.0 if X.ndim == 2 and X.shape[1] else 0.0,
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


def _build_direct_state_context(state: Any, actions: list[Action]) -> dict[str, Any]:
    board = _board_rows(state.board)
    opponent_board = _board_rows(state.opponent_board)
    common = np.zeros(HU_FEATURE_DIM, dtype=np.float32)

    _encode_board_rows(common, board, CURRENT_OFFSET)
    _encode_cards(common, state.dealt_cards, DEALT_OFFSET)
    _encode_board_rows(common, opponent_board, OPPONENT_OFFSET)
    _encode_cards(common, state.dead_cards, DEAD_OFFSET)
    _encode_binary(common, state.seat, SEAT_OFFSET)
    _encode_binary(common, state.to_act_order or "first", ORDER_OFFSET)
    _encode_opponent_row_stats(common, opponent_board)

    visible_cards = _visible_cards_for_state(state, actions[0] if actions else None)
    available_by_rank = _available_by_rank(visible_cards)
    opponent_top = _top_summary(opponent_board.get("top", ()), available_by_rank)
    opponent_rows = _complete_row_summary(opponent_board)
    opponent_matchups = {
        row: _row_matchup_summary(row, opponent_board.get(row, ()), available_by_rank)
        for row in ROWS
    }
    opponent_count = sum(len(opponent_board.get(row, ())) for row in ROWS)
    opponent_terminal: dict[str, float] | None = None
    if opponent_count == 13:
        try:
            score = score_board(
                opponent_board.get("top", ()),
                opponent_board.get("middle", ()),
                opponent_board.get("bottom", ()),
            )
            opponent_terminal = {
                "busted": 1.0 if score.busted else 0.0,
                "royalty": score.total_royalty / 100.0,
                "fl_entry": 1.0 if score.fl_entry.qualifies else 0.0,
            }
        except ValueError:
            opponent_terminal = None
    return {
        "common": common,
        "board": board,
        "opponent_board": opponent_board,
        "available_by_rank": available_by_rank,
        "opponent_top": opponent_top,
        "opponent_rows": opponent_rows,
        "opponent_matchups": opponent_matchups,
        "opponent_needed_top": _opponent_needed_top_ranks(opponent_board, available_by_rank),
        "opponent_count": opponent_count,
        "opponent_terminal": opponent_terminal,
    }


def _encode_action_direct(
    vector: np.ndarray,
    state: Any,
    action: Action,
    context: dict[str, Any],
    attribution: dict[str, float],
) -> None:
    started_at = time.perf_counter()
    for card, row_name in action.placements:
        _set_card_row(vector, card, row_name, PLACEMENT_OFFSET)
    _encode_cards(vector, action.discards, DISCARD_OFFSET)
    attribution["stage3_action_delta_seconds"] += time.perf_counter() - started_at

    started_at = time.perf_counter()
    next_board = _after_board_rows(context["board"], action)
    _encode_board_rows(vector, next_board, SELF_NEXT_OFFSET)
    attribution["stage3_after_board_construction_seconds"] += time.perf_counter() - started_at

    started_at = time.perf_counter()
    _encode_self_row_stats(vector, next_board)
    _encode_self_extra_stats_direct(vector, next_board)

    opponent_board = context["opponent_board"]
    available_by_rank = context["available_by_rank"]
    opponent_top = context["opponent_top"]
    opponent_rows = context["opponent_rows"]
    opponent_summaries = context["opponent_matchups"]

    hero_count = sum(len(next_board.get(row, ())) for row in ROWS)
    vector[GLOBAL_OFFSET + 0] = hero_count / 13.0
    vector[GLOBAL_OFFSET + 1] = int(context["opponent_count"]) / 13.0
    for row_index, row_name in enumerate(ROWS):
        hero_open = ROW_CAPACITY[row_name] - len(next_board.get(row_name, ()))
        opp_open = ROW_CAPACITY[row_name] - len(opponent_board.get(row_name, ()))
        vector[GLOBAL_OFFSET + 2 + row_index] = hero_open / ROW_CAPACITY[row_name]
        vector[GLOBAL_OFFSET + 5 + row_index] = opp_open / ROW_CAPACITY[row_name]

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
    attribution["stage3_row_summary_seconds"] += time.perf_counter() - started_at

    started_at = time.perf_counter()
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
    vector[derived_offset + 37] = sum(1 for card in action.discards if card[0] in "QKA") / max(len(action.discards), 1)
    vector[derived_offset + 38] = sum(
        1 for card in action.discards if card[0] in context["opponent_needed_top"]
    ) / max(len(action.discards), 1)
    for index, rank in enumerate(("Q", "K", "A")):
        vector[derived_offset + 39 + index] = available_by_rank[rank] / 4.0
        vector[derived_offset + 42 + index] = opponent_top[f"needs_{rank.lower()}"]
        vector[derived_offset + 45 + index] = hero_top[f"needs_{rank.lower()}"]

    matchup_offset = HU_MATCHUP_OFFSET
    for row_index, row_name in enumerate(ROWS):
        hero = hero_summaries[row_name]
        opponent = opponent_summaries[row_name]
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

    completed_hero = sum(int(len(next_board.get(row, ())) == ROW_CAPACITY[row]) for row in ROWS)
    completed_opponent = sum(int(len(opponent_board.get(row, ())) == ROW_CAPACITY[row]) for row in ROWS)
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
    attribution["stage3_global_summary_seconds"] += time.perf_counter() - started_at


def _board_rows(board: Board) -> dict[str, tuple[str, ...]]:
    return {
        "top": tuple(board.top),
        "middle": tuple(board.middle),
        "bottom": tuple(board.bottom),
    }


def _after_board_rows(board: dict[str, tuple[str, ...]], action: Action) -> dict[str, tuple[str, ...]]:
    rows = {row: list(cards) for row, cards in board.items()}
    for card, row_name in action.placements:
        rows[row_name].append(card)
    return {row: tuple(rows[row]) for row in ROWS}


def _encode_board_rows(vector: np.ndarray, board: dict[str, Sequence[str]], offset: int) -> None:
    for row_index, row_name in enumerate(ROWS):
        for card in board.get(row_name, ()):
            vector[offset + row_index * 52 + CARD_INDEX[card]] = 1.0


def _set_card_row(vector: np.ndarray, card: str, row_name: str, offset: int) -> None:
    row_index = 0 if row_name == "top" else 1 if row_name == "middle" else 2
    vector[offset + row_index * 52 + CARD_INDEX[card]] = 1.0


def _encode_cards(vector: np.ndarray, cards: Sequence[str], offset: int) -> None:
    for card in cards:
        index = CARD_INDEX.get(card)
        if index is not None:
            vector[offset + index] = 1.0


def _encode_binary(vector: np.ndarray, value: str, offset: int) -> None:
    vector[offset + SEAT_INDEX.get(value, 0)] = 1.0


def _encode_self_row_stats(vector: np.ndarray, board: dict[str, Sequence[str]]) -> None:
    for row_index, row_name in enumerate(ROWS):
        cards = board.get(row_name, ())
        vector[ROW_LEN_OFFSET + row_index] = len(cards) / 5.0
        for card in cards:
            vector[RANK_COUNT_OFFSET + row_index * 13 + RANK_INDEX[card[0]]] += 1.0 / 4.0
            vector[SUIT_COUNT_OFFSET + row_index * 4 + SUIT_INDEX[card[1]]] += 1.0 / 5.0


def _encode_opponent_row_stats(vector: np.ndarray, board: dict[str, Sequence[str]]) -> None:
    for row_index, row_name in enumerate(ROWS):
        cards = board.get(row_name, ())
        capacity = ROW_CAPACITY[row_name]
        vector[OPP_ROW_LEN_OFFSET + row_index] = len(cards) / capacity
        for card in cards:
            vector[OPP_RANK_COUNT_OFFSET + row_index * 13 + RANK_INDEX[card[0]]] += 1.0 / 4.0
            vector[OPP_SUIT_COUNT_OFFSET + row_index * 4 + SUIT_INDEX[card[1]]] += 1.0 / 5.0


def _encode_self_extra_stats_direct(vector: np.ndarray, board: dict[str, Sequence[str]]) -> None:
    for row_index, row_name in enumerate(ROWS):
        offset = ROW_EXTRA_OFFSET + row_index * ROW_EXTRA_DIM
        vector[offset : offset + ROW_EXTRA_DIM] = _row_extra_values(
            row_name,
            _cards_cache_key(board.get(row_name, ())),
        )
    vector[GLOBAL_EXTRA_OFFSET : GLOBAL_EXTRA_OFFSET + 4] = _global_extra_values(
        _cards_cache_key(board.get("top", ())),
        _cards_cache_key(board.get("middle", ())),
        _cards_cache_key(board.get("bottom", ())),
    )


def _visible_cards_for_state(state: Any, action: Action | None) -> tuple[str, ...]:
    cards: list[str] = []
    cards.extend(state.board.all_cards())
    cards.extend(state.opponent_board.all_cards())
    cards.extend(state.dealt_cards)
    cards.extend(state.dead_cards)
    if action is not None:
        cards.extend(card for card, _row in action.placements)
        cards.extend(action.discards)
    return tuple(dict.fromkeys(card for card in cards if card in CARD_INDEX))
