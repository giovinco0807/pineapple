"""Grading a Fantasyland setting: rank every legal 13-of-14 placement.

`ofc_regular.fantasyland.solve_fantasyland` answers "what is the best setting"
and stops there.  Grading a hand somebody actually played needs the rest of the
list: where their setting came in, and what it cost against the top.  So this
enumerates the same space and keeps a ranking.

The objective is the solver's -- total royalty plus the value of another
Fantasyland when the setting stays -- and it is deliberately opponent-blind.
A Fantasyland hand is set face down before anything opposite is visible, so the
decision genuinely has no opponent in it; the head-to-head result is reported
alongside as an outcome, never folded into the grade.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ofc_regular.cards import validate_cards
from ofc_regular.evaluator import (
    HAND_QUADS,
    HAND_TRIPS,
    evaluate_3_card,
    evaluate_5_card,
    get_bottom_royalty,
    get_middle_royalty,
    get_top_royalty,
)

FL_CARDS = 14
ROWS = ("top", "middle", "bottom")


def _row_key(cards: Sequence[str]) -> str:
    return ",".join(sorted(cards))


def board_key(top: Sequence[str], middle: Sequence[str], bottom: Sequence[str]) -> str:
    return "|".join((_row_key(top), _row_key(middle), _row_key(bottom)))


def score_setting(
    top: Sequence[str],
    middle: Sequence[str],
    bottom: Sequence[str],
    stay_bonus: float,
) -> Optional[Dict[str, Any]]:
    """Royalty, stay and objective for one setting, or None if it fouls."""
    top_value = evaluate_3_card(tuple(top))
    middle_value = evaluate_5_card(tuple(middle))
    bottom_value = evaluate_5_card(tuple(bottom))
    if top_value > middle_value or middle_value > bottom_value:
        return None
    top_royalty = get_top_royalty(tuple(top))
    middle_royalty = get_middle_royalty(tuple(middle))
    bottom_royalty = get_bottom_royalty(tuple(bottom))
    royalty = top_royalty + middle_royalty + bottom_royalty
    stay = top_value[0] == HAND_TRIPS or bottom_value[0] >= HAND_QUADS
    return {
        "top_royalty": top_royalty,
        "middle_royalty": middle_royalty,
        "bottom_royalty": bottom_royalty,
        "royalty": royalty,
        "stay": stay,
        "score": float(royalty) + (stay_bonus if stay else 0.0),
    }


def is_foul(board: Dict[str, Sequence[str]]) -> bool:
    """Does this 13-card setting break the top <= middle <= bottom order?"""
    return (
        evaluate_3_card(tuple(board["top"])) > evaluate_5_card(tuple(board["middle"]))
        or evaluate_5_card(tuple(board["middle"])) > evaluate_5_card(tuple(board["bottom"]))
    )


def rank_settings(
    cards: Sequence[str],
    *,
    stay_bonus: float,
    played: Optional[Dict[str, Sequence[str]]] = None,
    top_n: int = 20,
) -> Dict[str, Any]:
    """Rank every legal setting of `cards` (14) by royalty + stay value.

    Returns the top `top_n`, plus the rank and score of `played` when given.
    Ties share a rank: with fourteen cards a dozen settings routinely reach the
    same royalty, and calling the one the player happened to pick "12th" when
    eleven of those are the identical score would be a lie about their decision.
    """
    if len(cards) != FL_CARDS:
        raise ValueError(f"Fantasyland は14枚です（{len(cards)}枚）")
    validate_cards(list(cards))

    cards_tuple = tuple(cards)
    all_mask = (1 << FL_CARDS) - 1

    # Same precomputation as the solver: every 3-card and 5-card subset scored
    # once, then assembled.  The assembly is ~1M partitions before pruning.
    combo3: List[Tuple[int, Tuple[str, ...], tuple, int, bool]] = []
    for idxs in combinations(range(FL_CARDS), 3):
        hand = tuple(cards_tuple[i] for i in idxs)
        value = evaluate_3_card(tuple(sorted(hand)))
        mask = 0
        for i in idxs:
            mask |= 1 << i
        combo3.append((mask, hand, value, get_top_royalty(hand), value[0] == HAND_TRIPS))

    top_by_remaining: Dict[int, list] = {}
    for idxs in combinations(range(FL_CARDS), 4):
        mask = 0
        for i in idxs:
            mask |= 1 << i
        top_by_remaining[mask] = [e for e in combo3 if e[0] & mask == e[0]]

    combo5: List[Tuple[int, Tuple[str, ...], tuple, int, int, bool]] = []
    for idxs in combinations(range(FL_CARDS), 5):
        hand = tuple(cards_tuple[i] for i in idxs)
        value = evaluate_5_card(tuple(sorted(hand)))
        mask = 0
        for i in idxs:
            mask |= 1 << i
        combo5.append(
            (mask, hand, value, get_middle_royalty(hand), get_bottom_royalty(hand), value[0] >= HAND_QUADS)
        )

    played_key = None
    if played:
        played_key = board_key(played["top"], played["middle"], played["bottom"])

    settings: List[Dict[str, Any]] = []
    played_entry: Optional[Dict[str, Any]] = None
    legal = 0

    for bottom_mask, bottom, bottom_value, _mid_roy_as_bottom, bottom_royalty, bottom_stay in combo5:
        after_bottom = all_mask ^ bottom_mask
        for middle_mask, middle, middle_value, middle_royalty, _bot_roy_as_middle, _ in combo5:
            if middle_mask & bottom_mask:
                continue
            if middle_value > bottom_value:
                continue
            after_middle = after_bottom ^ middle_mask
            for top_mask, top, top_value, top_royalty, top_stay in top_by_remaining[after_middle]:
                if top_value > middle_value:
                    continue
                legal += 1
                stay = top_stay or bottom_stay
                royalty = top_royalty + middle_royalty + bottom_royalty
                score = float(royalty) + (stay_bonus if stay else 0.0)
                discard_mask = after_middle ^ top_mask
                entry = {
                    "top": list(top),
                    "middle": list(middle),
                    "bottom": list(bottom),
                    "discard": next(
                        cards_tuple[i] for i in range(FL_CARDS) if discard_mask & (1 << i)
                    ),
                    "royalty": royalty,
                    "top_royalty": top_royalty,
                    "middle_royalty": middle_royalty,
                    "bottom_royalty": bottom_royalty,
                    "stay": stay,
                    "score": score,
                }
                settings.append(entry)
                if played_key and played_entry is None:
                    if board_key(top, middle, bottom) == played_key:
                        played_entry = entry

    settings.sort(key=lambda e: (-e["score"], -e["royalty"], e["discard"]))

    rank = None
    if played_entry is not None:
        better = sum(1 for e in settings if e["score"] > played_entry["score"] + 1e-9)
        rank = better + 1

    best = settings[0] if settings else None
    return {
        "legal_count": legal,
        "best": best,
        "candidates": settings[:top_n],
        "played": played_entry,
        "rank": rank,
        "tied_with_played": (
            sum(1 for e in settings if played_entry and abs(e["score"] - played_entry["score"]) <= 1e-9)
            if played_entry
            else 0
        ),
        "ev_loss": (
            max(0.0, best["score"] - played_entry["score"])
            if best and played_entry
            else None
        ),
        "stay_bonus": stay_bonus,
    }


def rank_arrangements(
    placed: Dict[str, Sequence[str]],
    *,
    stay_bonus: float,
    top_n: int = 20,
) -> Dict[str, Any]:
    """Rank the arrangements of an already-known 13 cards.

    For the opponent's Fantasyland only the thirteen cards they set are ever
    visible; the fourteenth went face down.  Their real decision was 14-choose-13
    and cannot be reconstructed, but "given these thirteen, was that the best
    arrangement of them" is still a question worth answering, and it is a
    different one -- so it is a different function with a different name.
    """
    cards = [*placed["top"], *placed["middle"], *placed["bottom"]]
    if len(cards) != 13:
        raise ValueError(f"配置は13枚です（{len(cards)}枚）")
    validate_cards(cards)

    played_key = board_key(placed["top"], placed["middle"], placed["bottom"])
    settings: List[Dict[str, Any]] = []
    played_entry = None
    for bottom in combinations(cards, 5):
        rest9 = [c for c in cards if c not in bottom]
        for middle in combinations(rest9, 5):
            top = tuple(c for c in rest9 if c not in middle)
            scored = score_setting(top, middle, bottom, stay_bonus)
            if scored is None:
                continue
            entry = {"top": list(top), "middle": list(middle), "bottom": list(bottom), **scored}
            settings.append(entry)
            if played_entry is None and board_key(top, middle, bottom) == played_key:
                played_entry = entry

    settings.sort(key=lambda e: (-e["score"], -e["royalty"]))
    rank = None
    if played_entry is not None:
        rank = sum(1 for e in settings if e["score"] > played_entry["score"] + 1e-9) + 1
    best = settings[0] if settings else None
    return {
        "legal_count": len(settings),
        "best": best,
        "candidates": settings[:top_n],
        "played": played_entry,
        "rank": rank,
        "ev_loss": (
            max(0.0, best["score"] - played_entry["score"]) if best and played_entry else None
        ),
        "stay_bonus": stay_bonus,
    }
