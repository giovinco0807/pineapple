"""Action-level feature helpers for candidate reranker inputs.

The base reranker state encodes the board after an action.  That is enough in
principle, but it makes the network infer the action delta indirectly.  These
features make the discard, row assignment, and before/after row capacity
explicit while preserving old 520/522-dim data when not requested.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import numpy as np

from ai.engine.action_space import Action, get_semantic_action_index
from ai.engine.encoding import RANKS, SUITS
from ai.engine.turn_order import normalize_position, validate_decision_board_counts


ROW_ALIASES = {
    "top": "top",
    "mid": "middle",
    "middle": "middle",
    "bot": "bottom",
    "bottom": "bottom",
}
ROWS = ("top", "middle", "bottom")
ROW_LIMITS = {"top": 3, "middle": 5, "bottom": 5}
ACTION_FEATURE_DIM = 97
FL_OPPORTUNITY_FEATURE_DIM = 76
BOARD_CONTEXT_FEATURE_DIM = 96
BASE_COMPAT_DIMS = (520, 522, 822)


def canonical_decision_position(decision: dict[str, Any], *, validate_counts: bool = True) -> str:
    """Resolve a decision's HU role and optionally validate its public shape.

    Both legacy ``is_btn``-only records and ``position``-only records remain
    supported.  A record with neither field is ambiguous, and supplying both
    with different meanings is an error.
    """
    raw_position = decision.get("position")
    if raw_position in (None, ""):
        raw_position = decision.get("player_position")
    has_position = raw_position not in (None, "")
    has_is_btn = "is_btn" in decision and decision.get("is_btn") is not None
    if not has_position and not has_is_btn:
        raise ValueError("HU decision requires explicit position or is_btn")

    position = normalize_position(
        raw_position if has_position else None,
        is_btn=decision["is_btn"] if has_is_btn else None,
    )
    if validate_counts:
        board = decision.get("board") or {}
        opponent = decision.get("opponent_board") or decision.get("board_opponent") or {}
        validate_decision_board_counts(
            int(decision.get("turn", 0)),
            position,
            sum(_row_len(board, row) for row in ROWS),
            sum(_row_len(opponent, row) for row in ROWS),
        )
    return position


def _row_name(row: Any) -> str:
    return ROW_ALIASES.get(str(row), str(row))


def _rank_bucket(card: Any) -> int:
    text = str(card or "")
    if text.startswith("X"):
        return 13
    if text and text[0] in RANKS:
        return RANKS.index(text[0])
    return 13


def _rank_value(card: Any) -> int:
    idx = _rank_bucket(card)
    return idx if idx < 13 else -1


def _suit_bucket(card: Any) -> int:
    text = str(card or "")
    if text.startswith("X"):
        return 4
    if len(text) > 1 and text[1] in SUITS:
        return SUITS.index(text[1])
    return 4


def _cards_for_row(board: dict[str, Any], row: str) -> list[Any]:
    cards = list(board.get(row, []) or [])
    if row == "middle":
        cards.extend(board.get("mid", []) or [])
    elif row == "bottom":
        cards.extend(board.get("bot", []) or [])
    return cards


def _board_after_candidate(decision: dict[str, Any], candidate: dict[str, Any]) -> dict[str, list[str]]:
    board = decision.get("board") or {}
    out = {row: [str(card) for card in _cards_for_row(board, row)] for row in ROWS}
    for card, row in _candidate_placements(candidate):
        if row in out:
            out[row].append(str(card))
    return out


def _row_len(board: dict[str, Any], row: str) -> int:
    return len(_cards_for_row(board, row))


def _candidate_placements(candidate: dict[str, Any]) -> list[tuple[str, str]]:
    placements: list[tuple[str, str]] = []
    for item in candidate.get("placements", []) or []:
        if len(item) != 2:
            continue
        card, row = item
        placements.append((str(card), _row_name(row)))
    return placements


def _rank_counts(cards: Sequence[Any]) -> tuple[Counter[int], int]:
    counts: Counter[int] = Counter()
    jokers = 0
    for card in cards:
        rank = _rank_value(card)
        if rank < 0:
            jokers += 1
        else:
            counts[rank] += 1
    return counts, jokers


def _effective_rank_counts(cards: Sequence[Any]) -> Counter[int]:
    counts, jokers = _rank_counts(cards)
    effective: Counter[int] = Counter(counts)
    if jokers <= 0:
        return effective
    if not effective:
        effective[12] = min(jokers, 3)
        return effective
    for rank in sorted(effective.keys(), reverse=True):
        if jokers <= 0:
            break
        while jokers > 0 and effective[rank] < 3:
            effective[rank] += 1
            jokers -= 1
    if jokers > 0:
        effective[max(effective.keys())] += jokers
    return effective


def _row_strength(cards: Sequence[Any], row: str) -> float:
    if not cards:
        return 0.0
    counts = _effective_rank_counts(cards)
    if not counts:
        return 0.0
    groups = sorted(counts.values(), reverse=True)
    high = max(counts.keys()) / 12.0
    best_group_rank = max((rank for rank, count in counts.items() if count == groups[0]), default=0) / 12.0
    if row == "top":
        if groups[0] >= 3:
            return 0.80 + 0.18 * best_group_rank
        if groups[0] >= 2:
            return 0.35 + 0.35 * best_group_rank
        return 0.18 * high
    if groups[0] >= 4:
        return 0.88 + 0.10 * best_group_rank
    if groups[0] >= 3 and len(groups) > 1 and groups[1] >= 2:
        return 0.78 + 0.10 * best_group_rank
    if groups[0] >= 3:
        return 0.55 + 0.18 * best_group_rank
    if groups[0] >= 2 and len(groups) > 1 and groups[1] >= 2:
        return 0.42 + 0.16 * best_group_rank
    if groups[0] >= 2:
        return 0.24 + 0.18 * best_group_rank
    return 0.15 * high


def _row_summary(cards: Sequence[Any], row: str) -> list[float]:
    counts, jokers = _rank_counts(cards)
    effective = _effective_rank_counts(cards)
    row_len = len(cards)
    limit = ROW_LIMITS[row]
    group_counts = list(effective.values())
    max_count = max(group_counts) if group_counts else 0
    pair_ranks = [rank for rank, count in effective.items() if count >= 2]
    trips_ranks = [rank for rank, count in effective.items() if count >= 3]
    high_rank = max(effective.keys()) if effective else -1
    pair_rank = max(pair_ranks) if pair_ranks else -1
    trips_rank = max(trips_ranks) if trips_ranks else -1
    return [
        row_len / limit,
        max(limit - row_len, 0) / limit,
        jokers / 2.0,
        max_count / max(limit, 1),
        min(len(pair_ranks), 3) / 3.0,
        1.0 if trips_ranks else 0.0,
        (high_rank + 1) / 13.0 if high_rank >= 0 else 0.0,
        (pair_rank + 1) / 13.0 if pair_rank >= 0 else 0.0,
        (trips_rank + 1) / 13.0 if trips_rank >= 0 else 0.0,
        _row_strength(cards, row),
    ]


def _fl_ready_top(cards: Sequence[Any]) -> tuple[float, float, float, float]:
    effective = _effective_rank_counts(cards)
    qq = 1.0 if effective.get(10, 0) >= 2 else 0.0
    kk = 1.0 if effective.get(11, 0) >= 2 else 0.0
    aa = 1.0 if effective.get(12, 0) >= 2 else 0.0
    trips = 1.0 if any(count >= 3 for count in effective.values()) else 0.0
    return qq, kk, aa, trips


def _visible_cards_after_action(decision: dict[str, Any], after_board: dict[str, list[str]], candidate: dict[str, Any]) -> list[str]:
    visible: list[str] = []
    for row in ROWS:
        visible.extend(after_board.get(row, []))
    opponent = decision.get("opponent_board") or decision.get("board_opponent") or {}
    for row in ROWS:
        visible.extend(str(card) for card in _cards_for_row(opponent, row))
    visible.extend(str(card) for card in (decision.get("known_discards") or []))
    visible.extend(str(card) for card in (decision.get("exclude") or []))
    discard = candidate.get("discard")
    if discard not in (None, ""):
        visible.append(str(discard))
    return visible


def _rank_mask(cards: Sequence[Any]) -> int:
    mask = 0
    for card in cards:
        rank = _rank_value(card)
        if rank >= 0:
            mask |= 1 << rank
    return mask


def _straight_summary(cards: Sequence[Any]) -> list[float]:
    ranks = {_rank_value(card) for card in cards if _rank_value(card) >= 0}
    jokers = sum(1 for card in cards if _rank_value(card) < 0)
    # RANKS is 2..A.  Add wheel A2345 as a separate window by reusing Ace.
    windows = [set(range(start, start + 5)) for start in range(0, 9)]
    windows.append({12, 0, 1, 2, 3})
    best_have = 0
    best_with_joker = 0
    best_missing_live = 0
    for window in windows:
        have = len(ranks & window)
        with_joker = min(5, have + jokers)
        best_have = max(best_have, have)
        best_with_joker = max(best_with_joker, with_joker)
        best_missing_live = max(best_missing_live, 5 - have)
    return [
        best_have / 5.0,
        best_with_joker / 5.0,
        max(0, 5 - best_with_joker) / 5.0,
        best_missing_live / 5.0,
    ]


def _suit_summary(cards: Sequence[Any]) -> list[float]:
    suits = Counter()
    jokers = 0
    for card in cards:
        suit = _suit_bucket(card)
        if suit >= 4:
            jokers += 1
        else:
            suits[suit] += 1
    values = [suits.get(i, 0) for i in range(4)]
    max_suit = max(values) if values else 0
    row_len = len(cards)
    return [
        max_suit / 5.0,
        min(max_suit + jokers, 5) / 5.0,
        jokers / 2.0,
        1.0 if max_suit + jokers >= min(5, row_len) and row_len >= 3 else 0.0,
        float(np.std(values)) / 2.5 if values else 0.0,
    ]


def _remaining_counts_by_rank(visible: Sequence[Any]) -> list[int]:
    counts = Counter()
    for card in visible:
        rank = _rank_value(card)
        if rank >= 0:
            counts[rank] += 1
    return [max(4 - counts.get(rank, 0), 0) for rank in range(13)]


def _remaining_counts_by_suit(visible: Sequence[Any]) -> list[int]:
    counts = Counter()
    for card in visible:
        suit = _suit_bucket(card)
        if suit < 4:
            counts[suit] += 1
    return [max(13 - counts.get(suit, 0), 0) for suit in range(4)]


def board_context_feature_vector(decision: dict[str, Any], candidate: dict[str, Any]) -> np.ndarray:
    """Return richer runtime board/deck context for one candidate.

    The older FL opportunity block focuses mostly on top-row FL and pair-like
    row strength.  This block adds split visible-card counts, rank/suit
    blockers, and middle/bottom straight-flush potential.  It uses no teacher
    labels.
    """
    position = canonical_decision_position(decision)
    out = np.zeros(BOARD_CONTEXT_FEATURE_DIM, dtype=np.float32)
    offset = 0

    after_board = _board_after_candidate(decision, candidate)
    opponent = decision.get("opponent_board") or decision.get("board_opponent") or {}
    discard = candidate.get("discard")
    self_cards = [card for row in ROWS for card in after_board.get(row, [])]
    opp_cards = [str(card) for row in ROWS for card in _cards_for_row(opponent, row)]
    known_cards = [str(card) for card in (decision.get("known_discards") or [])]
    exclude_cards = [str(card) for card in (decision.get("exclude") or [])]
    discard_cards = [str(discard)] if discard not in (None, "") else []
    visible = [*self_cards, *opp_cards, *known_cards, *exclude_cards, *discard_cards]

    remaining_rank = _remaining_counts_by_rank(visible)
    out[offset : offset + 13] = np.asarray(remaining_rank, dtype=np.float32) / 4.0
    offset += 13
    remaining_suit = _remaining_counts_by_suit(visible)
    out[offset : offset + 4] = np.asarray(remaining_suit, dtype=np.float32) / 13.0
    offset += 4

    for source_cards in (self_cards, opp_cards, known_cards + exclude_cards + discard_cards):
        counts = Counter(_rank_value(card) for card in source_cards if _rank_value(card) >= 0)
        out[offset : offset + 13] = np.asarray([min(counts.get(rank, 0), 4) / 4.0 for rank in range(13)], dtype=np.float32)
        offset += 13

    for row in ("middle", "bottom"):
        cards = after_board[row]
        out[offset : offset + 4] = np.asarray(_straight_summary(cards), dtype=np.float32)
        offset += 4
        out[offset : offset + 5] = np.asarray(_suit_summary(cards), dtype=np.float32)
        offset += 5
        row_ranks = {_rank_value(card) for card in cards if _rank_value(card) >= 0}
        live_pair_outs = 0
        live_trips_outs = 0
        counts = _effective_rank_counts(cards)
        for rank, count in counts.items():
            if count == 1:
                live_pair_outs += remaining_rank[rank]
            elif count == 2:
                live_trips_outs += remaining_rank[rank]
        straight_live = sum(remaining_rank[rank] for rank in range(13) if rank not in row_ranks)
        out[offset : offset + 4] = np.asarray(
            [
                min(live_pair_outs, 12) / 12.0,
                min(live_trips_outs, 8) / 8.0,
                min(straight_live, 32) / 32.0,
                len(cards) / ROW_LIMITS[row],
            ],
            dtype=np.float32,
        )
        offset += 4

    top_cards = after_board["top"]
    top_ranks = [_rank_value(card) for card in top_cards if _rank_value(card) >= 0]
    high_live = sum(remaining_rank[rank] for rank in (10, 11, 12)) / 12.0
    top_pair_outs = sum(remaining_rank[rank] for rank in set(top_ranks)) / 8.0 if top_ranks else 0.0
    out[offset : offset + 4] = np.asarray(
        [
            high_live,
            top_pair_outs,
            len(top_cards) / ROW_LIMITS["top"],
            max(ROW_LIMITS["top"] - len(top_cards), 0) / ROW_LIMITS["top"],
        ],
        dtype=np.float32,
    )
    offset += 4

    # Row rank masks help tree models distinguish made duplicated ranks from
    # broad summary values without leaking teacher labels.
    for row in ROWS:
        mask = _rank_mask(after_board[row])
        out[offset] = bin(mask).count("1") / 13.0
        out[offset + 1] = mask / float((1 << 13) - 1)
        offset += 2

    out[offset : offset + 4] = np.asarray(
        [
            len(self_cards) / 13.0,
            len(opp_cards) / 13.0,
            len(known_cards + exclude_cards + discard_cards) / 39.0,
            1.0 if position == "btn" else 0.0,
        ],
        dtype=np.float32,
    )
    offset += 4

    assert offset == BOARD_CONTEXT_FEATURE_DIM, offset
    return out


def fl_opportunity_feature_vector(decision: dict[str, Any], candidate: dict[str, Any]) -> np.ndarray:
    """Return runtime-available FL/safety/count features for one candidate.

    These features are computed only from the visible decision state and the
    candidate action.  They intentionally do not read teacher EV, bust, FL, or
    rank labels.
    """
    out = np.zeros(FL_OPPORTUNITY_FEATURE_DIM, dtype=np.float32)
    offset = 0

    after_board = _board_after_candidate(decision, candidate)
    placements = _candidate_placements(candidate)
    placement_rows = {row: [] for row in ROWS}
    for card, row in placements:
        if row in placement_rows:
            placement_rows[row].append(card)

    row_strengths = {}
    for row in ROWS:
        summary = _row_summary(after_board[row], row)
        out[offset : offset + 10] = np.asarray(summary, dtype=np.float32)
        row_strengths[row] = float(summary[-1])
        offset += 10

    for row in ROWS:
        before_cards = _cards_for_row(decision.get("board") or {}, row)
        before_counts = _effective_rank_counts(before_cards)
        placed_counts = _effective_rank_counts(placement_rows[row])
        after_counts = _effective_rank_counts(after_board[row])
        extends_pair = 0.0
        makes_trips = 0.0
        for rank, count in placed_counts.items():
            if count + before_counts.get(rank, 0) >= 2 and before_counts.get(rank, 0) < 2:
                extends_pair = 1.0
            if count + before_counts.get(rank, 0) >= 3 and before_counts.get(rank, 0) < 3:
                makes_trips = 1.0
        high_placed = sum(1 for card in placement_rows[row] if _rank_bucket(card) >= 9) / 2.0
        out[offset : offset + 4] = np.asarray(
            [
                len(placement_rows[row]) / 2.0,
                high_placed,
                extends_pair,
                makes_trips if after_counts else 0.0,
            ],
            dtype=np.float32,
        )
        offset += 4

    top_strength = row_strengths["top"]
    mid_strength = row_strengths["middle"]
    bot_strength = row_strengths["bottom"]
    top_cards = after_board["top"]
    mid_cards = after_board["middle"]
    bot_cards = after_board["bottom"]
    qq, kk, aa, trips = _fl_ready_top(top_cards)
    out[offset : offset + 8] = np.asarray(
        [
            top_strength,
            mid_strength,
            bot_strength,
            mid_strength - top_strength,
            bot_strength - mid_strength,
            1.0 if len(top_cards) and len(mid_cards) and top_strength > mid_strength else 0.0,
            1.0 if len(mid_cards) and len(bot_cards) and mid_strength > bot_strength else 0.0,
            max(qq, kk, aa, trips),
        ],
        dtype=np.float32,
    )
    offset += 8

    top_counts, top_jokers = _rank_counts(top_cards)
    q_count = min(top_counts.get(10, 0) + top_jokers, 3) / 3.0
    k_count = min(top_counts.get(11, 0) + top_jokers, 3) / 3.0
    a_count = min(top_counts.get(12, 0) + top_jokers, 3) / 3.0
    high_singletons = sum(1 for rank in (10, 11, 12) if top_counts.get(rank, 0) == 1) / 3.0
    top_open = max(ROW_LIMITS["top"] - len(top_cards), 0) / ROW_LIMITS["top"]
    out[offset : offset + 12] = np.asarray(
        [
            q_count,
            k_count,
            a_count,
            top_jokers / 2.0,
            qq,
            kk,
            aa,
            trips,
            high_singletons,
            top_open,
            1.0 if any(_rank_bucket(card) >= 10 for card in placement_rows["top"]) else 0.0,
            1.0 if any(str(card).startswith("X") for card in placement_rows["top"]) else 0.0,
        ],
        dtype=np.float32,
    )
    offset += 12

    visible = _visible_cards_after_action(decision, after_board, candidate)
    rank_visible = Counter()
    joker_visible = 0
    for card in visible:
        rank = _rank_value(card)
        if rank < 0:
            joker_visible += 1
        else:
            rank_visible[rank] += 1
    for rank in range(13):
        out[offset + rank] = max(4 - rank_visible.get(rank, 0), 0) / 4.0
    offset += 13
    out[offset] = max(2 - joker_visible, 0) / 2.0
    offset += 1

    assert offset == FL_OPPORTUNITY_FEATURE_DIM, offset
    return out


def _candidate_action_index(decision: dict[str, Any], candidate: dict[str, Any]) -> int | None:
    turn = int(decision.get("turn", 0))
    discard = candidate.get("discard")
    if turn <= 0 or discard in (None, ""):
        return None
    try:
        action = Action(
            placements=_candidate_placements(candidate),
            discard=str(discard),
        )
        return int(get_semantic_action_index(action, list(decision.get("dealt", []) or [])))
    except Exception:
        return None


def action_feature_vector(decision: dict[str, Any], candidate: dict[str, Any]) -> np.ndarray:
    """Return a 97-dim action feature vector for one candidate.

    Layout:
    - 27 regular-turn semantic action one-hot
    - 14 discard rank one-hot
    - 5 discard suit/joker one-hot
    - 3 placed-card row counts
    - 42 placed-card row/rank counts
    - 3 before row slot ratios
    - 3 after row slot ratios
    """
    out = np.zeros(ACTION_FEATURE_DIM, dtype=np.float32)
    offset = 0

    action_idx = _candidate_action_index(decision, candidate)
    if action_idx is not None and 0 <= action_idx < 27:
        out[offset + action_idx] = 1.0
    offset += 27

    discard = candidate.get("discard")
    if discard not in (None, ""):
        out[offset + _rank_bucket(discard)] = 1.0
    offset += 14

    if discard not in (None, ""):
        out[offset + _suit_bucket(discard)] = 1.0
    offset += 5

    placements = _candidate_placements(candidate)
    row_counts = {row: 0 for row in ROWS}
    for _card, row in placements:
        if row in row_counts:
            row_counts[row] += 1
    for i, row in enumerate(ROWS):
        out[offset + i] = row_counts[row] / max(len(placements), 1)
    offset += 3

    for card, row in placements:
        if row not in ROWS:
            continue
        row_idx = ROWS.index(row)
        rank_idx = _rank_bucket(card)
        out[offset + row_idx * 14 + rank_idx] += 0.5
    offset += 42

    board = decision.get("board") or {}
    for i, row in enumerate(ROWS):
        out[offset + i] = max(ROW_LIMITS[row] - _row_len(board, row), 0) / ROW_LIMITS[row]
    offset += 3

    after_lens = {row: _row_len(board, row) + row_counts[row] for row in ROWS}
    for i, row in enumerate(ROWS):
        out[offset + i] = max(ROW_LIMITS[row] - after_lens[row], 0) / ROW_LIMITS[row]

    return out


def base_dim_for_target(target_dim: int, source_dim: int) -> int:
    if target_dim in BASE_COMPAT_DIMS:
        return target_dim
    candidate_base = target_dim - ACTION_FEATURE_DIM - FL_OPPORTUNITY_FEATURE_DIM - BOARD_CONTEXT_FEATURE_DIM
    if candidate_base in BASE_COMPAT_DIMS:
        return candidate_base
    candidate_base = target_dim - ACTION_FEATURE_DIM
    if candidate_base in BASE_COMPAT_DIMS:
        return candidate_base
    candidate_base = target_dim - ACTION_FEATURE_DIM - FL_OPPORTUNITY_FEATURE_DIM
    if candidate_base in BASE_COMPAT_DIMS:
        return candidate_base
    if target_dim > source_dim + ACTION_FEATURE_DIM + FL_OPPORTUNITY_FEATURE_DIM + BOARD_CONTEXT_FEATURE_DIM:
        return target_dim - ACTION_FEATURE_DIM - FL_OPPORTUNITY_FEATURE_DIM - BOARD_CONTEXT_FEATURE_DIM
    return target_dim


def adapt_np_state(state: np.ndarray, target_dim: int) -> np.ndarray:
    if state.shape[0] == target_dim:
        return state.astype(np.float32, copy=False)
    if state.shape[0] == 522 and target_dim == 520:
        return np.concatenate([state[:488], state[490:]]).astype(np.float32, copy=False)
    if state.shape[0] > target_dim:
        return state[:target_dim].astype(np.float32, copy=False)
    out = np.zeros(target_dim, dtype=np.float32)
    out[: state.shape[0]] = state
    return out


def adapt_np_state_with_action(
    state: np.ndarray,
    target_dim: int,
    decision: dict[str, Any],
    candidate: dict[str, Any],
) -> np.ndarray:
    base_dim = base_dim_for_target(target_dim, int(state.shape[0]))
    base = adapt_np_state(state, base_dim)
    if target_dim == base_dim:
        return base
    if target_dim == base_dim + ACTION_FEATURE_DIM:
        return np.concatenate([base, action_feature_vector(decision, candidate)]).astype(np.float32, copy=False)
    if target_dim == base_dim + ACTION_FEATURE_DIM + FL_OPPORTUNITY_FEATURE_DIM:
        return np.concatenate(
            [base, action_feature_vector(decision, candidate), fl_opportunity_feature_vector(decision, candidate)]
        ).astype(np.float32, copy=False)
    if target_dim == base_dim + ACTION_FEATURE_DIM + FL_OPPORTUNITY_FEATURE_DIM + BOARD_CONTEXT_FEATURE_DIM:
        return np.concatenate(
            [
                base,
                action_feature_vector(decision, candidate),
                fl_opportunity_feature_vector(decision, candidate),
                board_context_feature_vector(decision, candidate),
            ]
        ).astype(np.float32, copy=False)
    return adapt_np_state(base, target_dim)


def action_augmented_dim(base_dim: int = 520) -> int:
    return int(base_dim) + ACTION_FEATURE_DIM


def action_fl_augmented_dim(base_dim: int = 520) -> int:
    return int(base_dim) + ACTION_FEATURE_DIM + FL_OPPORTUNITY_FEATURE_DIM


def action_fl_board_context_dim(base_dim: int = 520) -> int:
    return int(base_dim) + ACTION_FEATURE_DIM + FL_OPPORTUNITY_FEATURE_DIM + BOARD_CONTEXT_FEATURE_DIM
