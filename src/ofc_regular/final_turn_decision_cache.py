"""Final-turn exact decision cache and profiling helpers."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable

from .action_space import Action, generate_turn_actions
from .cards import ALL_CARDS, RANK_VALUE, validate_cards
from .evaluator import (
    HAND_FLUSH,
    HAND_FULL_HOUSE,
    HAND_PAIR,
    HAND_QUADS,
    HAND_STRAIGHT,
    HAND_STRAIGHT_FLUSH,
    HAND_TRIPS,
    BoardScore,
    evaluate_3_card,
    evaluate_5_card,
)
from .rules import FantasylandEntry, REGULAR_RULES
from .state import Board
from .teacher import DEFAULT_FL_EV


FINAL_TURN_CACHE_VERSION = "final_turn_exact_v1"


@dataclass(frozen=True)
class FinalTurnDecision:
    action: Action
    score: float
    legal_action_count: int
    decision_type: str
    final_board: Board
    key_hash: str


@dataclass
class FinalTurnProfile:
    seconds: float = 0.0
    exact_enumeration_seconds: float = 0.0
    exact_scoring_seconds: float = 0.0
    policy_decision_seconds: float = 0.0
    legal_action_generation_seconds: float = 0.0
    hand_eval_seconds: float = 0.0
    royalty_scoring_seconds: float = 0.0
    foul_check_seconds: float = 0.0
    cache_lookup_seconds: float = 0.0
    cache_update_seconds: float = 0.0
    other_seconds: float = 0.0
    cache_hit: bool = False
    legal_action_count: int = 0
    key_hash: str = ""
    final_action: Action | None = None
    score: float = 0.0


class FinalTurnDecisionCache:
    def __init__(self, max_size: int = 200_000) -> None:
        self.max_size = max(0, int(max_size))
        self._items: OrderedDict[tuple[Any, ...], FinalTurnDecision] = OrderedDict()

    def get(self, key: tuple[Any, ...]) -> FinalTurnDecision | None:
        if self.max_size <= 0:
            return None
        value = self._items.get(key)
        if value is None:
            return None
        self._items.move_to_end(key)
        return value

    def put(self, key: tuple[Any, ...], value: FinalTurnDecision) -> None:
        if self.max_size <= 0:
            return
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self.max_size:
            self._items.popitem(last=False)

    def __len__(self) -> int:
        return len(self._items)

    def memory_estimate_bytes(self) -> int:
        # Intentionally rough; enough for profile trend checks without importing
        # platform-specific object sizing logic.
        return len(self._items) * 640


def final_turn_canonical_key(
    *,
    board: Board,
    opponent_board: Board,
    dealt_cards: Iterable[str],
    dead_cards: Iterable[str],
    actor: str,
    seat: str,
    to_act_order: str,
    fl_ev: dict[int, float] | None = None,
) -> tuple[Any, ...]:
    fl = fl_ev or DEFAULT_FL_EV
    return (
        FINAL_TURN_CACHE_VERSION,
        "regular_ofc_v1",
        "regular_ofc_v1",
        _board_key(board),
        _board_key(opponent_board),
        _cards_key(dealt_cards),
        _cards_key(dead_cards),
        "final_turn",
        actor,
        seat,
        to_act_order,
        bool(opponent_board.is_complete()),
        tuple(sorted((int(k), float(v)) for k, v in fl.items())),
    )


def final_turn_key_hash(key: tuple[Any, ...]) -> str:
    payload = json.dumps(_jsonable(key), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def decide_final_turn_exact(
    *,
    board: Board,
    dealt_cards: Iterable[str],
    opponent_board: Board,
    dead_cards: Iterable[str],
    actor: str,
    seat: str,
    to_act_order: str,
    cache: FinalTurnDecisionCache | None = None,
    use_cache: bool = True,
    fl_ev: dict[int, float] | None = None,
) -> tuple[FinalTurnDecision | None, FinalTurnProfile, dict[str, Any]]:
    fl_ev = fl_ev or DEFAULT_FL_EV
    dealt = tuple(dealt_cards)
    dead = tuple(dead_cards)
    key = final_turn_canonical_key(
        board=board,
        opponent_board=opponent_board,
        dealt_cards=dealt,
        dead_cards=dead,
        actor=actor,
        seat=seat,
        to_act_order=to_act_order,
        fl_ev=fl_ev,
    )
    key_hash = final_turn_key_hash(key)
    started_at = time.perf_counter()
    profile = FinalTurnProfile(key_hash=key_hash)

    lookup_started_at = time.perf_counter()
    cached = cache.get(key) if use_cache and cache is not None else None
    profile.cache_lookup_seconds = time.perf_counter() - lookup_started_at
    if cached is not None:
        profile.cache_hit = True
        profile.legal_action_count = cached.legal_action_count
        profile.final_action = cached.action
        profile.score = cached.score
        profile.seconds = time.perf_counter() - started_at
        return cached, profile, {
            "key_hash": key_hash,
            "canonical_key": _jsonable(key),
            "cache_hit": True,
        }

    legal_started_at = time.perf_counter()
    actions = generate_turn_actions(board, dealt)
    profile.legal_action_generation_seconds = time.perf_counter() - legal_started_at
    profile.legal_action_count = len(actions)
    if not actions:
        profile.seconds = time.perf_counter() - started_at
        return None, profile, {
            "key_hash": key_hash,
            "canonical_key": _jsonable(key),
            "cache_hit": False,
        }

    enumeration_seconds = 0.0
    scoring_seconds = 0.0
    hand_eval_seconds = 0.0
    royalty_seconds = 0.0
    foul_seconds = 0.0
    best: tuple[float, Action, Board] | None = None
    terminal_opponent = opponent_board if opponent_board.is_complete() else None
    opponent_score: BoardScore | None = None
    if terminal_opponent is not None:
        opponent_started_at = time.perf_counter()
        opponent_score, opponent_parts = _score_board_profiled(terminal_opponent)
        scoring_seconds += time.perf_counter() - opponent_started_at
        hand_eval_seconds += opponent_parts["hand_eval_seconds"]
        royalty_seconds += opponent_parts["royalty_scoring_seconds"]
        foul_seconds += opponent_parts["foul_check_seconds"]
    for action in actions:
        enum_started_at = time.perf_counter()
        next_board = board.place(action.placements)
        if not next_board.is_complete():
            enumeration_seconds += time.perf_counter() - enum_started_at
            continue
        enumeration_seconds += time.perf_counter() - enum_started_at
        score_started_at = time.perf_counter()
        score, score_parts = _terminal_score_profiled(
            next_board,
            terminal_opponent,
            fl_ev,
            opponent_score=opponent_score,
        )
        scoring_seconds += time.perf_counter() - score_started_at
        hand_eval_seconds += score_parts["hand_eval_seconds"]
        royalty_seconds += score_parts["royalty_scoring_seconds"]
        foul_seconds += score_parts["foul_check_seconds"]
        if best is None or score > best[0]:
            best = (float(score), action, next_board)

    profile.exact_enumeration_seconds = enumeration_seconds
    profile.exact_scoring_seconds = scoring_seconds
    profile.hand_eval_seconds = hand_eval_seconds
    profile.royalty_scoring_seconds = royalty_seconds
    profile.foul_check_seconds = foul_seconds
    if best is None:
        profile.seconds = time.perf_counter() - started_at
        profile.policy_decision_seconds = profile.seconds
        profile.other_seconds = _profile_other_seconds(profile)
        return None, profile, {
            "key_hash": key_hash,
            "canonical_key": _jsonable(key),
            "cache_hit": False,
        }

    decision = FinalTurnDecision(
        action=best[1],
        score=best[0],
        legal_action_count=len(actions),
        decision_type="exact_final_turn",
        final_board=best[2],
        key_hash=key_hash,
    )
    update_started_at = time.perf_counter()
    if use_cache and cache is not None:
        cache.put(key, decision)
    profile.cache_update_seconds = time.perf_counter() - update_started_at
    profile.final_action = decision.action
    profile.score = decision.score
    profile.seconds = time.perf_counter() - started_at
    profile.policy_decision_seconds = profile.seconds
    profile.other_seconds = _profile_other_seconds(profile)
    return decision, profile, {
        "key_hash": key_hash,
        "canonical_key": _jsonable(key),
        "cache_hit": False,
    }


def final_turn_slow_state_record(
    *,
    profile: FinalTurnProfile,
    board: Board,
    opponent_board: Board,
    dealt_cards: Iterable[str],
    dead_cards: Iterable[str],
    actor: str,
    seat: str,
    to_act_order: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "key_hash": profile.key_hash,
        "seconds": profile.seconds,
        "legal_action_count": profile.legal_action_count,
        "exact_enumeration_seconds": profile.exact_enumeration_seconds,
        "exact_scoring_seconds": profile.exact_scoring_seconds,
        "policy_decision_seconds": profile.policy_decision_seconds,
        "legal_action_generation_seconds": profile.legal_action_generation_seconds,
        "hand_eval_seconds": profile.hand_eval_seconds,
        "royalty_scoring_seconds": profile.royalty_scoring_seconds,
        "foul_check_seconds": profile.foul_check_seconds,
        "cache_hit": profile.cache_hit,
        "actor": actor,
        "seat": seat,
        "to_act_order": to_act_order,
        "hero_board": _board_json(board),
        "opponent_board": _board_json(opponent_board),
        "cards_to_place": list(dealt_cards),
        "dead_cards": list(dead_cards),
        "final_action": _action_json(profile.final_action),
        "score": profile.score,
        "metadata": metadata or {},
    }


def _terminal_score_profiled(
    board: Board,
    opponent_board: Board | None,
    fl_ev: dict[int, float],
    *,
    opponent_score: BoardScore | None = None,
) -> tuple[float, dict[str, float]]:
    if opponent_board is not None and set(board.all_cards()) & set(opponent_board.all_cards()):
        raise ValueError("hero and opponent boards overlap")
    own, own_parts = _score_board_profiled(board)
    opp = opponent_score
    opp_parts = {"hand_eval_seconds": 0.0, "royalty_scoring_seconds": 0.0, "foul_check_seconds": 0.0}
    if opponent_board is not None and opp is None:
        opp, opp_parts = _score_board_profiled(opponent_board)
    if opp is None:
        score = _standalone_terminal_score(own, fl_ev)
    else:
        score = _heads_up_terminal_score(own, opp, fl_ev)
    return score, {
        "hand_eval_seconds": own_parts["hand_eval_seconds"] + opp_parts["hand_eval_seconds"],
        "royalty_scoring_seconds": own_parts["royalty_scoring_seconds"] + opp_parts["royalty_scoring_seconds"],
        "foul_check_seconds": own_parts["foul_check_seconds"] + opp_parts["foul_check_seconds"],
    }


def _score_board_profiled(board: Board) -> tuple[BoardScore, dict[str, float]]:
    top_key = _cards_key(board.top)
    middle_key = _cards_key(board.middle)
    bottom_key = _cards_key(board.bottom)

    hand_started_at = time.perf_counter()
    top_value = _evaluate_3_cached(top_key)
    middle_value = _evaluate_5_cached(middle_key)
    bottom_value = _evaluate_5_cached(bottom_key)
    hand_eval_seconds = time.perf_counter() - hand_started_at

    foul_started_at = time.perf_counter()
    busted = top_value > middle_value or middle_value > bottom_value
    foul_check_seconds = time.perf_counter() - foul_started_at

    royalty_started_at = time.perf_counter()
    top_royalty = middle_royalty = bottom_royalty = 0
    fl_entry = FantasylandEntry(False, 0, None)
    if not busted:
        top_royalty = _top_royalty_from_value(top_value)
        middle_royalty = _middle_royalty_from_value(middle_value)
        bottom_royalty = _bottom_royalty_from_value(bottom_value)
        fl_entry = _fl_entry_cached(top_key)
    royalty_scoring_seconds = time.perf_counter() - royalty_started_at

    score = BoardScore(
        busted=busted,
        top_value=top_value,
        middle_value=middle_value,
        bottom_value=bottom_value,
        top_royalty=top_royalty,
        middle_royalty=middle_royalty,
        bottom_royalty=bottom_royalty,
        total_royalty=top_royalty + middle_royalty + bottom_royalty,
        fl_entry=fl_entry,
    )
    return score, {
        "hand_eval_seconds": hand_eval_seconds,
        "royalty_scoring_seconds": royalty_scoring_seconds,
        "foul_check_seconds": foul_check_seconds,
    }


@lru_cache(maxsize=500_000)
def _evaluate_3_cached(cards_key: tuple[str, ...]) -> tuple[int, tuple[int, ...]]:
    return evaluate_3_card(cards_key)


@lru_cache(maxsize=500_000)
def _evaluate_5_cached(cards_key: tuple[str, ...]) -> tuple[int, tuple[int, ...]]:
    return evaluate_5_card(cards_key)


@lru_cache(maxsize=200_000)
def _fl_entry_cached(top_key: tuple[str, ...]) -> FantasylandEntry:
    ranks = [card[0] for card in top_key]
    counts = Counter(ranks)

    if any(count >= 3 for count in counts.values()):
        entry_type = "trips"
    elif counts.get("A", 0) >= 2:
        entry_type = "aa"
    elif counts.get("K", 0) >= 2:
        entry_type = "kk"
    elif counts.get("Q", 0) >= 2:
        entry_type = "qq"
    else:
        return FantasylandEntry(False, 0, None)
    return FantasylandEntry(True, int(REGULAR_RULES.fl_entry_cards[entry_type]), entry_type)


def _top_royalty_from_value(value: tuple[int, tuple[int, ...]]) -> int:
    category, ranks = value
    if category == HAND_TRIPS:
        return 10 + (ranks[0] - 2)
    if category == HAND_PAIR and ranks[0] >= RANK_VALUE["6"]:
        return ranks[0] - 5
    return 0


def _middle_royalty_from_value(value: tuple[int, tuple[int, ...]]) -> int:
    category, ranks = value
    if category == HAND_STRAIGHT_FLUSH and ranks[0] == 14:
        return 50
    if category == HAND_STRAIGHT_FLUSH:
        return 30
    if category == HAND_QUADS:
        return 20
    if category == HAND_FULL_HOUSE:
        return 12
    if category == HAND_FLUSH:
        return 8
    if category == HAND_STRAIGHT:
        return 4
    if category == HAND_TRIPS:
        return 2
    return 0


def _bottom_royalty_from_value(value: tuple[int, tuple[int, ...]]) -> int:
    category, ranks = value
    if category == HAND_STRAIGHT_FLUSH and ranks[0] == 14:
        return 25
    if category == HAND_STRAIGHT_FLUSH:
        return 15
    if category == HAND_QUADS:
        return 10
    if category == HAND_FULL_HOUSE:
        return 6
    if category == HAND_FLUSH:
        return 4
    if category == HAND_STRAIGHT:
        return 2
    return 0


def _standalone_terminal_score(board_score: BoardScore, fl_ev: dict[int, float]) -> float:
    if board_score.busted:
        return 0.0
    fl_bonus = fl_ev.get(board_score.fl_entry.card_count, 0.0)
    return float(board_score.total_royalty) + fl_bonus


def _heads_up_terminal_score(own: BoardScore, opp: BoardScore, fl_ev: dict[int, float]) -> float:
    own_royalty = 0 if own.busted else own.total_royalty
    opp_royalty = 0 if opp.busted else opp.total_royalty
    own_fl = 0.0 if own.busted else fl_ev.get(own.fl_entry.card_count, 0.0)
    opp_fl = 0.0 if opp.busted else fl_ev.get(opp.fl_entry.card_count, 0.0)
    if own.busted and opp.busted:
        return 0.0
    if own.busted:
        return -6.0 - opp_royalty - opp_fl
    if opp.busted:
        return 6.0 + own_royalty + own_fl
    row_points = 0
    for own_value, opp_value in (
        (own.top_value, opp.top_value),
        (own.middle_value, opp.middle_value),
        (own.bottom_value, opp.bottom_value),
    ):
        if own_value > opp_value:
            row_points += 1
        elif own_value < opp_value:
            row_points -= 1
    if row_points == 3:
        row_points += 3
    elif row_points == -3:
        row_points -= 3
    return float(row_points + own_royalty - opp_royalty + own_fl - opp_fl)


def _profile_other_seconds(profile: FinalTurnProfile) -> float:
    known = (
        profile.exact_enumeration_seconds
        + profile.exact_scoring_seconds
        + profile.legal_action_generation_seconds
        + profile.cache_lookup_seconds
        + profile.cache_update_seconds
    )
    return max(0.0, profile.seconds - known)


def _board_key(board: Board) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    return (
        _cards_key(board.top),
        _cards_key(board.middle),
        _cards_key(board.bottom),
    )


def _cards_key(cards: Iterable[str]) -> tuple[str, ...]:
    values = tuple(cards)
    validate_cards(values)
    return tuple(sorted(values, key=_card_sort_index))


def _card_sort_index(card: str) -> int:
    try:
        return ALL_CARDS.index(card)
    except ValueError:
        return 999


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _board_json(board: Board) -> dict[str, list[str]]:
    return {
        "top": list(board.top),
        "middle": list(board.middle),
        "bottom": list(board.bottom),
    }


def _action_json(action: Action | None) -> dict[str, Any] | None:
    if action is None:
        return None
    return {
        "placements": [list(item) for item in action.placements],
        "discards": list(action.discards),
    }
