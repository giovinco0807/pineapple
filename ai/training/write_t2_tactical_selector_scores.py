"""Write runtime-only tactical selector scores for converted T2 datasets.

The selector is intentionally narrow: it gives a separate score source to
top-AA / middle-pair completion spots where generic model selectors can
discard the matching middle card.  It does not use teacher EV to build the
score; EV is only used for the optional summary metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.training.convert_action_value_teacher import valid_candidate_indices


RANKS = "23456789TJQKA"
RANK_VALUE = {rank: i for i, rank in enumerate(RANKS)}
INACTIVE_SCORE = -1.0e8


def card_rank(card: str) -> str:
    return str(card or "")[:1]


def row_name(row: object) -> str:
    if isinstance(row, str):
        return row.lower()
    return {0: "top", 1: "middle", 2: "bottom"}.get(int(row), str(row).lower())


def row_cards(record: dict, name: str) -> list[str]:
    board = record.get("board") or {}
    return list(board.get(name) or [])


def opponent_row_cards(record: dict, name: str) -> list[str]:
    board = record.get("opponent_board") or {}
    return list(board.get(name) or [])


def placed_by_row(candidate: dict) -> dict[str, list[str]]:
    out = {"top": [], "middle": [], "bottom": []}
    for item in candidate.get("placements") or []:
        if len(item) != 2:
            continue
        card, row = item
        name = row_name(row)
        if name in out:
            out[name].append(str(card))
    return out


def has_top_aa(record: dict) -> bool:
    ranks = Counter(card_rank(card) for card in row_cards(record, "top"))
    jokers = ranks.get("X", 0)
    return ranks.get("A", 0) + jokers >= 2


def middle_pair_ranks(record: dict) -> list[str]:
    ranks = Counter(card_rank(card) for card in row_cards(record, "middle"))
    pairs = [rank for rank, count in ranks.items() if rank in RANK_VALUE and count >= 2]
    return sorted(pairs, key=lambda rank: RANK_VALUE[rank], reverse=True)


def active_pair_ranks(record: dict) -> list[str]:
    if not has_top_aa(record):
        return []
    top = row_cards(record, "top")
    if len(top) < 3:
        return []
    non_ace_top = [card_rank(card) for card in top if card_rank(card) != "A"]
    if non_ace_top and max(RANK_VALUE.get(rank, -1) for rank in non_ace_top) < RANK_VALUE["Q"]:
        return []
    middle = row_cards(record, "middle")
    middle_counts = Counter(card_rank(card) for card in middle)
    kickers = [
        rank
        for rank, count in middle_counts.items()
        if rank in RANK_VALUE and count == 1
    ]
    if kickers and max(RANK_VALUE[rank] for rank in kickers) > RANK_VALUE["9"]:
        return []
    dealt_ranks = Counter(card_rank(card) for card in record.get("dealt") or [])
    return [rank for rank in middle_pair_ranks(record) if dealt_ranks.get(rank, 0) > 0]


def tactical_score(record: dict, candidate: dict) -> float:
    score = 0.0
    pairs = active_pair_ranks(record)
    if not pairs:
        return score

    placed = placed_by_row(candidate)
    placed_middle = placed["middle"]
    placed_bottom = placed["bottom"]
    placed_top = placed["top"]
    discard = str(candidate.get("discard") or "")
    dealt = [str(card) for card in record.get("dealt") or []]

    for pair_rank in pairs:
        if not any(card_rank(card) == pair_rank for card in dealt):
            continue
        rank_bonus = (RANK_VALUE[pair_rank] + 1) / 13.0
        if any(card_rank(card) == pair_rank for card in placed_middle):
            score += 9.0 + rank_bonus
        if card_rank(discard) == pair_rank:
            score -= 9.0 + rank_bonus
        if any(card_rank(card) == pair_rank for card in placed_top + placed_bottom):
            score -= 2.5 + rank_bonus * 0.5

    if any(card_rank(card) == "A" for card in dealt):
        if any(card_rank(card) == "A" for card in placed_bottom):
            score += 4.0
        if card_rank(discard) == "A":
            score -= 1.5
        if len(row_cards(record, "top")) >= 2 and any(card_rank(card) == "A" for card in placed_top):
            score -= 1.0

    # Prefer discarding the low non-pair card when the pair-completion shape is present.
    if discard and card_rank(discard) in RANK_VALUE and RANK_VALUE[card_rank(discard)] <= RANK_VALUE["8"]:
        score += 0.5

    return float(score)


def top_is_weak(record: dict) -> bool:
    top = row_cards(record, "top")
    if not top:
        return True
    ranks = Counter(card_rank(card) for card in top)
    if ranks.get("X", 0):
        return False
    if any(count >= 2 and rank in RANK_VALUE and RANK_VALUE[rank] >= RANK_VALUE["Q"] for rank, count in ranks.items()):
        return False
    return max((RANK_VALUE.get(card_rank(card), -1) for card in top), default=-1) < RANK_VALUE["Q"] or len(top) <= 1


def weak_top_middle_guard_active(record: dict) -> bool:
    if not top_is_weak(record):
        return False
    middle_ranks = {card_rank(card) for card in row_cards(record, "middle") if card_rank(card) in RANK_VALUE}
    dealt_ranks = {card_rank(card) for card in record.get("dealt") or [] if card_rank(card) in RANK_VALUE}
    return bool(middle_ranks & dealt_ranks)


def weak_top_middle_guard_score(record: dict, candidate: dict) -> float:
    if not weak_top_middle_guard_active(record):
        return 0.0
    placed = placed_by_row(candidate)
    discard = str(candidate.get("discard") or "")
    middle_ranks = Counter(card_rank(card) for card in row_cards(record, "middle"))
    dealt = [str(card) for card in record.get("dealt") or []]
    dealt_matching = {
        card_rank(card)
        for card in dealt
        if card_rank(card) in RANK_VALUE and middle_ranks.get(card_rank(card), 0) > 0
    }
    score = 0.0
    top_len = len(row_cards(record, "top"))
    middle_len = len(row_cards(record, "middle"))
    bottom_len = len(row_cards(record, "bottom"))

    for match_rank in dealt_matching:
        rank_weight = 1.0 + RANK_VALUE[match_rank] / 12.0
        if any(card_rank(card) == match_rank for card in placed["middle"]):
            score -= 5.5 * rank_weight
            if middle_len >= 4:
                score -= 3.0 * rank_weight
        if any(card_rank(card) == match_rank for card in placed["bottom"]):
            score += 3.5 * rank_weight
        if card_rank(discard) == match_rank:
            score += 2.0 * rank_weight
            if RANK_VALUE[match_rank] >= RANK_VALUE["T"]:
                score += 2.5

    if top_len <= 1 and any(card_rank(card) == "A" for card in placed["top"]):
        score += 5.0
    if any(card_rank(card) == "A" for card in placed["middle"]):
        score -= 2.0

    bottom_high = sum(
        1
        for card in placed["bottom"]
        if card_rank(card) in RANK_VALUE and RANK_VALUE[card_rank(card)] >= RANK_VALUE["8"]
    )
    if bottom_len <= 2 and len(placed["bottom"]) >= 2:
        score += 3.0 + bottom_high * 1.5
    elif bottom_len <= 3 and bottom_high:
        score += bottom_high * 1.0

    return float(score)


def strict_weak_top_guard_context(record: dict) -> dict[str, object]:
    top = row_cards(record, "top")
    middle = row_cards(record, "middle")
    bottom = row_cards(record, "bottom")
    if not top_is_weak(record) or len(top) > 1:
        return {"active": False}

    top_ranks = {
        card_rank(card)
        for card in top
        if card_rank(card) in RANK_VALUE
    }
    middle_counts = Counter(card_rank(card) for card in middle)
    dealt = [str(card) for card in record.get("dealt") or []]
    dealt_ranks = {card_rank(card) for card in dealt if card_rank(card) in RANK_VALUE}
    middle_ranks = {rank for rank in middle_counts if rank in RANK_VALUE}
    matching = dealt_ranks & middle_ranks
    high_dealt = {
        rank
        for rank in dealt_ranks
        if RANK_VALUE.get(rank, -1) >= RANK_VALUE["Q"]
    }

    pair_ranks = {
        rank
        for rank, count in middle_counts.items()
        if rank in RANK_VALUE and count >= 2
    }
    singleton_highs = {
        rank
        for rank, count in middle_counts.items()
        if rank in RANK_VALUE
        and count == 1
        and RANK_VALUE[rank] >= RANK_VALUE["T"]
        and RANK_VALUE[rank] <= RANK_VALUE["Q"]
        and rank in dealt_ranks
    }
    top_high_pair_available = bool(
        top_ranks
        and high_dealt
        and any(rank in high_dealt and RANK_VALUE[rank] >= RANK_VALUE["Q"] for rank in top_ranks)
    )
    pair_plus_high = (
        len(middle) == 3
        and bool(pair_ranks)
        and bool(singleton_highs)
        and "A" in dealt_ranks
        and not top_high_pair_available
    )
    top_max_rank = max((RANK_VALUE[rank] for rank in top_ranks), default=-1)
    bottom_fill = (
        len(middle) >= 4
        and len(bottom) <= 2
        and top_max_rank <= RANK_VALUE["7"]
        and "A" in middle_ranks
        and bool(high_dealt)
        and bool(matching)
        and not top_high_pair_available
    )
    active = bool(pair_plus_high or bottom_fill)
    return {
        "active": active,
        "pair_plus_high": pair_plus_high,
        "bottom_fill": bottom_fill,
        "top_high_pair_available": top_high_pair_available,
        "top_max_rank": top_max_rank,
        "singleton_highs": singleton_highs,
        "matching": matching,
        "high_dealt": high_dealt,
    }


def strict_weak_top_guard_active(record: dict) -> bool:
    return bool(strict_weak_top_guard_context(record).get("active"))


def strict_weak_top_guard_score(record: dict, candidate: dict) -> float:
    context = strict_weak_top_guard_context(record)
    if not context.get("active"):
        return 0.0
    placed = placed_by_row(candidate)
    discard = str(candidate.get("discard") or "")
    discard_rank = card_rank(discard)
    singleton_highs = set(context.get("singleton_highs") or set())
    matching = set(context.get("matching") or set())
    high_dealt = set(context.get("high_dealt") or set())
    score = 0.0

    if context.get("pair_plus_high"):
        if any(card_rank(card) in {"A", "K"} for card in placed["top"]):
            score += 5.0
        if any(card_rank(card) in singleton_highs for card in placed["middle"]):
            score -= 9.0
        if any(card_rank(card) in singleton_highs for card in placed["bottom"]):
            score -= 2.5
        if discard_rank in singleton_highs:
            score += 8.0
        if any(
            card_rank(card) in RANK_VALUE
            and RANK_VALUE[card_rank(card)] <= RANK_VALUE["9"]
            for card in placed["middle"]
        ):
            score += 2.0
        if discard_rank in {"A", "K"}:
            score -= 5.0

    if context.get("bottom_fill"):
        bottom_ranks = {card_rank(card) for card in placed["bottom"]}
        middle_ranks = {card_rank(card) for card in placed["middle"]}
        strong_matching = {
            rank for rank in matching if RANK_VALUE.get(rank, -1) >= RANK_VALUE["8"]
        }
        support_bottom = {
            rank
            for rank in bottom_ranks
            if rank in RANK_VALUE
            and rank not in high_dealt
            and RANK_VALUE[rank] >= RANK_VALUE["8"]
        }
        if bottom_ranks & high_dealt:
            score += 4.0
        if bottom_ranks & strong_matching:
            score += 4.0
        if (bottom_ranks & high_dealt) and (bottom_ranks & strong_matching):
            score += 4.0
        if support_bottom:
            score += 3.0 + max(RANK_VALUE[rank] for rank in support_bottom) / 12.0
        if middle_ranks & matching:
            score -= 9.0
        if middle_ranks & high_dealt:
            score -= 3.0
        if discard_rank in high_dealt:
            score -= 8.0
        if discard_rank in RANK_VALUE and RANK_VALUE[discard_rank] >= RANK_VALUE["8"]:
            score -= 4.0
        if discard_rank in strong_matching and (bottom_ranks & strong_matching):
            score += 2.0
        if len(placed["bottom"]) >= 2:
            score += 1.5

    return float(score)


def joker_middle_stability_context(record: dict) -> dict[str, object]:
    top = row_cards(record, "top")
    middle = row_cards(record, "middle")
    bottom = row_cards(record, "bottom")
    dealt = [str(card) for card in record.get("dealt") or []]
    dealt_ranks = {card_rank(card) for card in dealt}
    if top or len(middle) != 2 or len(bottom) != 5:
        return {"active": False}
    if "X" not in dealt_ranks:
        return {"active": False}
    high_dealt = {
        rank for rank in dealt_ranks if rank in RANK_VALUE and RANK_VALUE[rank] >= RANK_VALUE["Q"]
    }
    if not high_dealt:
        return {"active": False}
    middle_ranks = [card_rank(card) for card in middle if card_rank(card) in RANK_VALUE]
    if len(set(middle_ranks)) < 2:
        return {"active": False}
    bottom_counts = Counter(card_rank(card) for card in bottom)
    bottom_pairish = any(count >= 2 for count in bottom_counts.values())
    if not bottom_pairish:
        return {"active": False}
    return {
        "active": True,
        "high_dealt": high_dealt,
        "lowest_dealt": min(
            (RANK_VALUE[rank] for rank in dealt_ranks if rank in RANK_VALUE),
            default=RANK_VALUE["A"],
        ),
    }


def joker_middle_stability_active(record: dict) -> bool:
    return bool(joker_middle_stability_context(record).get("active"))


def joker_middle_stability_score(record: dict, candidate: dict) -> float:
    context = joker_middle_stability_context(record)
    if not context.get("active"):
        return 0.0
    placed = placed_by_row(candidate)
    discard_rank = card_rank(str(candidate.get("discard") or ""))
    high_dealt = set(context.get("high_dealt") or set())
    score = 0.0

    middle_ranks = {card_rank(card) for card in placed["middle"]}
    top_ranks = {card_rank(card) for card in placed["top"]}
    bottom_ranks = {card_rank(card) for card in placed["bottom"]}

    if "X" in middle_ranks:
        score += 12.0
    if "X" in top_ranks:
        score -= 12.0
    if "X" in bottom_ranks:
        score -= 6.0

    if middle_ranks & high_dealt:
        score += 5.0
    if top_ranks & high_dealt:
        score += 1.0
    if bottom_ranks & high_dealt:
        score -= 2.0
    if discard_rank in high_dealt:
        score -= 6.0

    if discard_rank in RANK_VALUE and RANK_VALUE[discard_rank] <= RANK_VALUE["8"]:
        score += 2.0
    if len(placed["middle"]) == 2:
        score += 1.0
    if placed["bottom"]:
        score -= 3.0

    return float(score)


def ace_top_medium_guard_context(record: dict) -> dict[str, object]:
    top = row_cards(record, "top")
    middle = row_cards(record, "middle")
    bottom = row_cards(record, "bottom")
    dealt = [str(card) for card in record.get("dealt") or []]
    dealt_ranks = {card_rank(card) for card in dealt}
    if len(top) != 1 or card_rank(top[0]) != "A":
        return {"active": False}
    board_ranks = {card_rank(card) for card in middle + bottom}
    if "X" in dealt_ranks or "X" in board_ranks or len(middle) != 3 or len(bottom) != 3:
        return {"active": False}

    middle_counts = Counter(card_rank(card) for card in middle)
    low_middle_pairs = {
        rank
        for rank, count in middle_counts.items()
        if rank in RANK_VALUE and count >= 2 and RANK_VALUE[rank] <= RANK_VALUE["5"]
    }
    middle_high = max(
        (RANK_VALUE[rank] for rank in middle_counts if rank in RANK_VALUE),
        default=-1,
    )
    if not low_middle_pairs or middle_high > RANK_VALUE["7"]:
        return {"active": False}

    bottom_counts = Counter(card_rank(card) for card in bottom)
    bottom_pair_ranks = {
        rank
        for rank, count in bottom_counts.items()
        if rank in RANK_VALUE and count >= 2
    }
    bottom_pairish = any(
        count >= 2 and rank in RANK_VALUE and RANK_VALUE[rank] >= RANK_VALUE["8"]
        for rank, count in bottom_counts.items()
    )
    if not bottom_pairish:
        return {"active": False}

    middle_ranks = {rank for rank in middle_counts if rank in RANK_VALUE}
    if dealt_ranks & middle_ranks:
        return {"active": False}
    if dealt_ranks & bottom_pair_ranks:
        return {"active": False}

    medium_dealt = {
        rank
        for rank in dealt_ranks
        if rank in RANK_VALUE and RANK_VALUE["8"] <= RANK_VALUE[rank] <= RANK_VALUE["J"]
    }
    low_dealt = {
        rank
        for rank in dealt_ranks
        if rank in RANK_VALUE and RANK_VALUE[rank] <= RANK_VALUE["6"]
    }
    if not medium_dealt or not low_dealt:
        return {"active": False}
    return {
        "active": True,
        "medium_dealt": medium_dealt,
        "low_dealt": low_dealt,
    }


def ace_top_medium_guard_active(record: dict) -> bool:
    return bool(ace_top_medium_guard_context(record).get("active"))


def ace_top_medium_guard_score(record: dict, candidate: dict) -> float:
    context = ace_top_medium_guard_context(record)
    if not context.get("active"):
        return 0.0
    placed = placed_by_row(candidate)
    discard_rank = card_rank(str(candidate.get("discard") or ""))
    medium_dealt = set(context.get("medium_dealt") or set())
    low_dealt = set(context.get("low_dealt") or set())
    top_ranks = {card_rank(card) for card in placed["top"]}
    middle_ranks = {card_rank(card) for card in placed["middle"]}
    bottom_ranks = {card_rank(card) for card in placed["bottom"]}
    score = 0.0

    if top_ranks & medium_dealt:
        score += 10.0
    if middle_ranks & medium_dealt:
        score -= 5.0
    if bottom_ranks & medium_dealt:
        score -= 2.0
    if discard_rank in medium_dealt:
        score -= 8.0

    if bottom_ranks & low_dealt:
        score += 4.0
    if top_ranks & low_dealt:
        score -= 2.0
    if middle_ranks & low_dealt:
        score -= 1.0
    if discard_rank in low_dealt:
        score += 1.0

    return float(score)


def low_top_pair_keep_open_context(record: dict) -> dict[str, object]:
    top = row_cards(record, "top")
    middle = row_cards(record, "middle")
    bottom = row_cards(record, "bottom")
    dealt = [str(card) for card in record.get("dealt") or []]
    dealt_ranks = [card_rank(card) for card in dealt]
    if len(top) != 2 or len(middle) != 2 or len(bottom) != 3:
        return {"active": False}
    if "X" in dealt_ranks:
        return {"active": False}

    top_ranks = [card_rank(card) for card in top]
    if len(set(top_ranks)) != 1 or top_ranks[0] not in RANK_VALUE:
        return {"active": False}
    top_pair_rank = top_ranks[0]
    if RANK_VALUE[top_pair_rank] > RANK_VALUE["5"]:
        return {"active": False}

    dealt_counts = Counter(rank for rank in dealt_ranks if rank in RANK_VALUE)
    low_pair_ranks = {
        rank
        for rank, count in dealt_counts.items()
        if count >= 2 and RANK_VALUE[rank] <= RANK_VALUE["5"] and rank != top_pair_rank
    }
    high_dealt = {
        rank
        for rank, count in dealt_counts.items()
        if count == 1 and rank in {"Q", "K"}
    }
    if not low_pair_ranks or not high_dealt:
        return {"active": False}

    bottom_high_count = sum(
        1
        for card in bottom
        if card_rank(card) in RANK_VALUE and RANK_VALUE[card_rank(card)] >= RANK_VALUE["9"]
    )
    middle_has_high = any(
        card_rank(card) in RANK_VALUE and RANK_VALUE[card_rank(card)] >= RANK_VALUE["9"]
        for card in middle
    )
    if bottom_high_count < 2 or not middle_has_high:
        return {"active": False}

    return {
        "active": True,
        "low_pair_ranks": low_pair_ranks,
        "high_dealt": high_dealt,
    }


def low_top_pair_keep_open_active(record: dict) -> bool:
    return bool(low_top_pair_keep_open_context(record).get("active"))


def low_top_pair_keep_open_score(record: dict, candidate: dict) -> float:
    context = low_top_pair_keep_open_context(record)
    if not context.get("active"):
        return 0.0
    placed = placed_by_row(candidate)
    discard_rank = card_rank(str(candidate.get("discard") or ""))
    low_pair_ranks = set(context.get("low_pair_ranks") or set())
    high_dealt = set(context.get("high_dealt") or set())
    top_ranks = {card_rank(card) for card in placed["top"]}
    middle_ranks = {card_rank(card) for card in placed["middle"]}
    bottom_ranks = {card_rank(card) for card in placed["bottom"]}
    score = 0.0

    if bottom_ranks & high_dealt:
        score += 12.0
    if top_ranks & high_dealt:
        score -= 14.0
    if middle_ranks & high_dealt:
        score -= 4.0
    if discard_rank in high_dealt:
        score -= 10.0

    if middle_ranks & low_pair_ranks:
        score += 5.0
    if top_ranks & low_pair_ranks:
        score -= 2.5
    if bottom_ranks & low_pair_ranks:
        score -= 2.0
    if discard_rank in low_pair_ranks:
        score += 1.0

    return float(score)


def joker_bottom_top_kicker_context(record: dict) -> dict[str, object]:
    top = row_cards(record, "top")
    middle = row_cards(record, "middle")
    bottom = row_cards(record, "bottom")
    dealt = [str(card) for card in record.get("dealt") or []]
    if len(top) != 1 or len(middle) != 2 or len(bottom) != 4:
        return {"active": False}

    top_rank = card_rank(top[0])
    if top_rank not in RANK_VALUE or RANK_VALUE[top_rank] > RANK_VALUE["9"]:
        return {"active": False}

    joker_cards = [card for card in dealt if card_rank(card) == "X"]
    natural_cards = [card for card in dealt if card_rank(card) in RANK_VALUE]
    if len(joker_cards) != 1 or len(natural_cards) != 2:
        return {"active": False}

    natural_ranks = {card_rank(card) for card in natural_cards}
    if not natural_ranks or max(RANK_VALUE[rank] for rank in natural_ranks) < RANK_VALUE["T"]:
        return {"active": False}
    if max(RANK_VALUE[rank] for rank in natural_ranks) > RANK_VALUE["Q"]:
        return {"active": False}
    if min(RANK_VALUE[rank] for rank in natural_ranks) < RANK_VALUE["8"]:
        return {"active": False}

    middle_ranks = [card_rank(card) for card in middle if card_rank(card) in RANK_VALUE]
    if not any(RANK_VALUE[rank] >= RANK_VALUE["Q"] for rank in middle_ranks):
        return {"active": False}
    if not any(RANK_VALUE[rank] <= RANK_VALUE["6"] for rank in middle_ranks):
        return {"active": False}

    bottom_ranks = [card_rank(card) for card in bottom if card_rank(card) in RANK_VALUE]
    if len(bottom_ranks) != 4 or max(RANK_VALUE[rank] for rank in bottom_ranks) > RANK_VALUE["8"]:
        return {"active": False}
    bottom_suits = [str(card)[1:2] for card in bottom if len(str(card)) >= 2 and card_rank(card) != "X"]
    suit_counts = Counter(bottom_suits)
    if max(suit_counts.values(), default=0) < 3:
        return {"active": False}

    highest = max(natural_cards, key=lambda card: RANK_VALUE[card_rank(card)])
    lowest = min(natural_cards, key=lambda card: RANK_VALUE[card_rank(card)])
    return {
        "active": True,
        "joker": joker_cards[0],
        "highest": highest,
        "lowest": lowest,
        "highest_rank": card_rank(highest),
        "lowest_rank": card_rank(lowest),
    }


def joker_bottom_top_kicker_active(record: dict) -> bool:
    return bool(joker_bottom_top_kicker_context(record).get("active"))


def joker_bottom_top_kicker_score(record: dict, candidate: dict) -> float:
    context = joker_bottom_top_kicker_context(record)
    if not context.get("active"):
        return 0.0
    placed = placed_by_row(candidate)
    discard = str(candidate.get("discard") or "")
    joker = str(context.get("joker") or "")
    highest = str(context.get("highest") or "")
    lowest = str(context.get("lowest") or "")
    score = 0.0

    if joker in placed["bottom"]:
        score += 12.0
    if joker in placed["top"] or joker in placed["middle"] or discard == joker:
        score -= 12.0

    if highest in placed["top"]:
        score += 9.0
    if highest in placed["middle"]:
        score -= 8.0
    if highest in placed["bottom"]:
        score -= 3.0
    if discard == highest:
        score -= 10.0

    if discard == lowest:
        score += 5.0
    if lowest in placed["middle"]:
        score -= 1.5
    if lowest in placed["top"]:
        score -= 1.0
    if lowest in placed["bottom"]:
        score -= 1.0

    return float(score)


def bottom_twopair_qq_middle_context(record: dict) -> dict[str, object]:
    top = row_cards(record, "top")
    middle = row_cards(record, "middle")
    bottom = row_cards(record, "bottom")
    dealt = [str(card) for card in record.get("dealt") or []]
    dealt_ranks = [card_rank(card) for card in dealt]
    if len(top) != 1 or len(middle) != 2 or len(bottom) != 4:
        return {"active": False}
    if "X" in dealt_ranks:
        return {"active": False}

    top_rank = card_rank(top[0])
    if top_rank not in RANK_VALUE or RANK_VALUE[top_rank] > RANK_VALUE["7"]:
        return {"active": False}

    middle_ranks = [card_rank(card) for card in middle if card_rank(card) in RANK_VALUE]
    if len(middle_ranks) != 2 or max(RANK_VALUE[rank] for rank in middle_ranks) > RANK_VALUE["5"]:
        return {"active": False}

    bottom_counts = Counter(card_rank(card) for card in bottom)
    bottom_pairs = {
        rank
        for rank, count in bottom_counts.items()
        if rank in RANK_VALUE and count >= 2
    }
    if len(bottom_pairs) < 2:
        return {"active": False}
    if max(RANK_VALUE[rank] for rank in bottom_pairs) < RANK_VALUE["8"]:
        return {"active": False}

    dealt_counts = Counter(rank for rank in dealt_ranks if rank in RANK_VALUE)
    pair_cards = [card for card in dealt if card_rank(card) == "Q"]
    low_cards = [
        card
        for card in dealt
        if card_rank(card) in RANK_VALUE and RANK_VALUE[card_rank(card)] <= RANK_VALUE["5"]
    ]
    if dealt_counts.get("Q", 0) != 2 or len(pair_cards) != 2 or len(low_cards) != 1:
        return {"active": False}

    return {
        "active": True,
        "pair_rank": "Q",
        "pair_cards": pair_cards,
        "low_card": low_cards[0],
    }


def bottom_twopair_qq_middle_active(record: dict) -> bool:
    return bool(bottom_twopair_qq_middle_context(record).get("active"))


def bottom_twopair_qq_middle_score(record: dict, candidate: dict) -> float:
    context = bottom_twopair_qq_middle_context(record)
    if not context.get("active"):
        return 0.0
    placed = placed_by_row(candidate)
    discard = str(candidate.get("discard") or "")
    pair_cards = set(context.get("pair_cards") or [])
    low_card = str(context.get("low_card") or "")
    score = 0.0

    middle_pair_count = sum(1 for card in placed["middle"] if card in pair_cards)
    top_pair_count = sum(1 for card in placed["top"] if card in pair_cards)
    bottom_pair_count = sum(1 for card in placed["bottom"] if card in pair_cards)

    if middle_pair_count == 2:
        score += 18.0
    elif middle_pair_count == 1:
        score += 2.0
    if top_pair_count:
        score -= 12.0 * top_pair_count
    if bottom_pair_count:
        score -= 5.0 * bottom_pair_count
    if discard in pair_cards:
        score -= 14.0

    if discard == low_card:
        score += 8.0
    if low_card in placed["middle"]:
        score -= 4.0
    if low_card in placed["top"]:
        score -= 3.0
    if low_card in placed["bottom"]:
        score -= 2.0

    return float(score)


def top_ace_kk_joker_bottom_context(record: dict) -> dict[str, object]:
    top = row_cards(record, "top")
    middle = row_cards(record, "middle")
    bottom = row_cards(record, "bottom")
    dealt = [str(card) for card in record.get("dealt") or []]
    dealt_ranks = [card_rank(card) for card in dealt]
    if len(top) != 1 or len(middle) != 3 or len(bottom) != 3:
        return {"active": False}
    if card_rank(top[0]) != "A":
        return {"active": False}

    middle_counts = Counter(card_rank(card) for card in middle)
    if not any(rank in RANK_VALUE and count >= 2 for rank, count in middle_counts.items()):
        return {"active": False}

    bottom_counts = Counter(card_rank(card) for card in bottom)
    if not any(rank in RANK_VALUE and count >= 2 for rank, count in bottom_counts.items()):
        return {"active": False}

    joker_cards = [card for card in dealt if card_rank(card) == "X"]
    king_cards = [card for card in dealt if card_rank(card) == "K"]
    if len(joker_cards) != 1 or len(king_cards) != 2:
        return {"active": False}

    return {
        "active": True,
        "joker": joker_cards[0],
        "king_cards": king_cards,
    }


def top_ace_kk_joker_bottom_active(record: dict) -> bool:
    return bool(top_ace_kk_joker_bottom_context(record).get("active"))


def top_ace_kk_joker_bottom_score(record: dict, candidate: dict) -> float:
    context = top_ace_kk_joker_bottom_context(record)
    if not context.get("active"):
        return 0.0
    placed = placed_by_row(candidate)
    discard = str(candidate.get("discard") or "")
    joker = str(context.get("joker") or "")
    king_cards = set(context.get("king_cards") or [])
    score = 0.0

    joker_bottom = joker in placed["bottom"]
    king_top_count = sum(1 for card in placed["top"] if card in king_cards)
    king_middle_count = sum(1 for card in placed["middle"] if card in king_cards)
    king_bottom_count = sum(1 for card in placed["bottom"] if card in king_cards)
    king_discard = discard in king_cards

    if joker_bottom:
        score += 18.0
    if joker in placed["top"] or joker in placed["middle"] or discard == joker:
        score -= 18.0

    if king_top_count == 1:
        score += 16.0
    elif king_top_count > 1:
        score -= 8.0
    if king_discard:
        score += 8.0
    if king_middle_count:
        score -= 10.0 * king_middle_count
    if king_bottom_count:
        score -= 4.0 * king_bottom_count

    return float(score)


def top_aajoker_extra_ace_discard_context(record: dict) -> dict[str, object]:
    top = row_cards(record, "top")
    middle = row_cards(record, "middle")
    bottom = row_cards(record, "bottom")
    dealt = [str(card) for card in record.get("dealt") or []]
    dealt_ranks = [card_rank(card) for card in dealt]
    if len(top) != 2 or len(middle) != 3 or len(bottom) != 2:
        return {"active": False}
    top_ranks = Counter(card_rank(card) for card in top)
    if top_ranks.get("A", 0) != 1 or top_ranks.get("X", 0) != 1:
        return {"active": False}
    middle_counts = Counter(card_rank(card) for card in middle)
    if not any(rank in RANK_VALUE and count >= 2 for rank, count in middle_counts.items()):
        return {"active": False}
    middle_ranks = [card_rank(card) for card in middle]
    if any(rank not in RANK_VALUE or RANK_VALUE[rank] < RANK_VALUE["8"] for rank in middle_ranks):
        return {"active": False}
    if dealt_ranks.count("A") != 1 or "X" in dealt_ranks:
        return {"active": False}
    non_aces = [card for card in dealt if card_rank(card) in RANK_VALUE and card_rank(card) != "A"]
    if len(non_aces) != 2:
        return {"active": False}
    if max(RANK_VALUE[card_rank(card)] for card in non_aces) > RANK_VALUE["8"]:
        return {"active": False}
    return {
        "active": True,
        "ace": next(card for card in dealt if card_rank(card) == "A"),
        "high": max(non_aces, key=lambda card: RANK_VALUE[card_rank(card)]),
        "low": min(non_aces, key=lambda card: RANK_VALUE[card_rank(card)]),
    }


def top_aajoker_extra_ace_discard_active(record: dict) -> bool:
    return bool(top_aajoker_extra_ace_discard_context(record).get("active"))


def top_aajoker_extra_ace_discard_score(record: dict, candidate: dict) -> float:
    context = top_aajoker_extra_ace_discard_context(record)
    if not context.get("active"):
        return 0.0
    placed = placed_by_row(candidate)
    discard = str(candidate.get("discard") or "")
    ace = str(context.get("ace") or "")
    high = str(context.get("high") or "")
    low = str(context.get("low") or "")
    score = 0.0

    if discard == ace:
        score += 16.0
    if ace in placed["top"] or ace in placed["middle"] or ace in placed["bottom"]:
        score -= 16.0

    if high in placed["top"]:
        score += 10.0
    if high in placed["middle"]:
        score += 2.0
    if high in placed["bottom"]:
        score -= 4.0
    if discard == high:
        score -= 8.0

    if low in placed["middle"]:
        score += 8.0
    if low in placed["top"]:
        score -= 1.5
    if low in placed["bottom"]:
        score -= 3.0
    if discard == low:
        score -= 8.0

    return float(score)


def empty_top_bottom_full_middle_high_context(record: dict) -> dict[str, object]:
    top = row_cards(record, "top")
    middle = row_cards(record, "middle")
    bottom = row_cards(record, "bottom")
    dealt = [str(card) for card in record.get("dealt") or []]
    if top or len(middle) != 2 or len(bottom) != 5 or len(dealt) != 3:
        return {"active": False}
    if any(card_rank(card) == "X" for card in dealt):
        return {"active": False}
    middle_ranks = [card_rank(card) for card in middle]
    if any(rank not in RANK_VALUE or RANK_VALUE[rank] > RANK_VALUE["5"] for rank in middle_ranks):
        return {"active": False}
    pair_cards = [card for card in dealt if card_rank(card) in middle_ranks]
    kickers = [
        card
        for card in dealt
        if card_rank(card) in RANK_VALUE and card_rank(card) not in middle_ranks
    ]
    if len(pair_cards) != 1 or len(kickers) != 2:
        return {"active": False}
    if min(RANK_VALUE[card_rank(card)] for card in kickers) <= RANK_VALUE["5"]:
        return {"active": False}
    return {
        "active": True,
        "pair_card": pair_cards[0],
        "middle_card": max(kickers, key=lambda card: RANK_VALUE[card_rank(card)]),
        "top_card": min(kickers, key=lambda card: RANK_VALUE[card_rank(card)]),
    }


def empty_top_bottom_full_middle_high_active(record: dict) -> bool:
    return bool(empty_top_bottom_full_middle_high_context(record).get("active"))


def empty_top_bottom_full_middle_high_score(record: dict, candidate: dict) -> float:
    context = empty_top_bottom_full_middle_high_context(record)
    if not context.get("active"):
        return 0.0
    placed = placed_by_row(candidate)
    discard = str(candidate.get("discard") or "")
    pair_card = str(context.get("pair_card") or "")
    middle_card = str(context.get("middle_card") or "")
    top_card = str(context.get("top_card") or "")
    score = 0.0

    if discard == pair_card:
        score += 12.0
    if pair_card in placed["middle"]:
        score -= 10.0
    if pair_card in placed["top"]:
        score -= 4.0

    if middle_card in placed["middle"]:
        score += 10.0
    if middle_card in placed["top"]:
        score += 2.0
    if discard == middle_card:
        score -= 10.0

    if top_card in placed["top"]:
        score += 8.0
    if top_card in placed["middle"]:
        score -= 2.0
    if discard == top_card:
        score -= 8.0

    return float(score)


def bottom_trips_low_kicker_context(record: dict) -> dict[str, object]:
    top = row_cards(record, "top")
    middle = row_cards(record, "middle")
    bottom = row_cards(record, "bottom")
    dealt = [str(card) for card in record.get("dealt") or []]
    if len(top) != 1 or len(middle) != 3 or len(bottom) != 3 or len(dealt) != 3:
        return {"active": False}
    if card_rank(top[0]) != "A" or any(card_rank(card) == "X" for card in dealt):
        return {"active": False}
    bottom_counts = Counter(card_rank(card) for card in bottom)
    trips = [
        rank
        for rank, count in bottom_counts.items()
        if rank in RANK_VALUE and count >= 3
    ]
    if len(trips) != 1 or RANK_VALUE[trips[0]] < RANK_VALUE["T"]:
        return {"active": False}
    dealt_natural = [card for card in dealt if card_rank(card) in RANK_VALUE]
    low_cards = [card for card in dealt_natural if RANK_VALUE[card_rank(card)] <= RANK_VALUE["4"]]
    top_cards = [card for card in dealt_natural if RANK_VALUE[card_rank(card)] >= RANK_VALUE["8"]]
    if len(low_cards) != 1 or not top_cards:
        return {"active": False}
    top_card = max(top_cards, key=lambda card: RANK_VALUE[card_rank(card)])
    discard_options = [card for card in dealt_natural if card not in {low_cards[0], top_card}]
    if len(discard_options) != 1:
        return {"active": False}
    return {
        "active": True,
        "low_card": low_cards[0],
        "top_card": top_card,
        "discard_card": discard_options[0],
    }


def bottom_trips_low_kicker_active(record: dict) -> bool:
    return bool(bottom_trips_low_kicker_context(record).get("active"))


def bottom_trips_low_kicker_score(record: dict, candidate: dict) -> float:
    context = bottom_trips_low_kicker_context(record)
    if not context.get("active"):
        return 0.0
    placed = placed_by_row(candidate)
    discard = str(candidate.get("discard") or "")
    low_card = str(context.get("low_card") or "")
    top_card = str(context.get("top_card") or "")
    discard_card = str(context.get("discard_card") or "")
    score = 0.0

    if low_card in placed["bottom"]:
        score += 12.0
    if low_card in placed["top"] or low_card in placed["middle"] or discard == low_card:
        score -= 8.0

    if top_card in placed["top"]:
        score += 12.0
    if top_card in placed["middle"]:
        score -= 4.0
    if top_card in placed["bottom"]:
        score -= 5.0
    if discard == top_card:
        score -= 12.0

    if discard == discard_card:
        score += 8.0
    if discard_card in placed["bottom"]:
        score -= 4.0

    return float(score)


def middle_trips_bottom_connector_context(record: dict) -> dict[str, object]:
    top = row_cards(record, "top")
    middle = row_cards(record, "middle")
    bottom = row_cards(record, "bottom")
    dealt = [str(card) for card in record.get("dealt") or []]
    if len(top) != 1 or len(middle) != 4 or len(bottom) != 2 or len(dealt) != 3:
        return {"active": False}
    if any(card_rank(card) not in RANK_VALUE for card in top + middle + bottom + dealt):
        return {"active": False}
    top_rank = card_rank(top[0])
    if RANK_VALUE[top_rank] < RANK_VALUE["K"]:
        return {"active": False}

    middle_counts = Counter(card_rank(card) for card in middle)
    trips = [rank for rank, count in middle_counts.items() if count >= 3]
    if len(trips) != 1:
        return {"active": False}

    bottom_ranks = [card_rank(card) for card in bottom]
    if len(set(bottom_ranks)) != 2:
        return {"active": False}
    bottom_values = [RANK_VALUE[rank] for rank in bottom_ranks]
    if max(bottom_values) > RANK_VALUE["T"]:
        return {"active": False}

    high_cards = [
        card
        for card in dealt
        if RANK_VALUE["Q"] <= RANK_VALUE[card_rank(card)] <= RANK_VALUE["K"]
    ]
    if len(high_cards) != 1:
        return {"active": False}
    connector_cards = [card for card in dealt if card != high_cards[0]]
    if len(connector_cards) != 2:
        return {"active": False}
    if any(RANK_VALUE[card_rank(card)] > RANK_VALUE["T"] for card in connector_cards):
        return {"active": False}

    connector_values = [RANK_VALUE[card_rank(card)] for card in connector_cards]
    combined = bottom_values + connector_values
    if max(combined) - min(combined) > 4:
        return {"active": False}
    if not all(any(abs(value - bottom_value) <= 1 for bottom_value in bottom_values) for value in connector_values):
        return {"active": False}

    return {
        "active": True,
        "high_card": high_cards[0],
        "connector_cards": connector_cards,
    }


def middle_trips_bottom_connector_active(record: dict) -> bool:
    return bool(middle_trips_bottom_connector_context(record).get("active"))


def middle_trips_bottom_connector_score(record: dict, candidate: dict) -> float:
    context = middle_trips_bottom_connector_context(record)
    if not context.get("active"):
        return 0.0
    placed = placed_by_row(candidate)
    discard = str(candidate.get("discard") or "")
    high_card = str(context.get("high_card") or "")
    connector_cards = [str(card) for card in context.get("connector_cards") or []]
    score = 0.0

    bottom_connector_count = sum(1 for card in connector_cards if card in placed["bottom"])
    score += 7.0 * bottom_connector_count
    if bottom_connector_count == len(connector_cards) and discard == high_card:
        score += 12.0

    for card in connector_cards:
        if discard == card:
            score -= 10.0
        if card in placed["top"] or card in placed["middle"]:
            score -= 4.0

    if high_card in placed["top"]:
        score -= 9.0
    if high_card in placed["bottom"] or high_card in placed["middle"]:
        score -= 3.0
    if discard == high_card:
        score += 4.0

    return float(score)


def top_pair_fill_bottom_connector_context(record: dict) -> dict[str, object]:
    top = row_cards(record, "top")
    middle = row_cards(record, "middle")
    bottom = row_cards(record, "bottom")
    dealt = [str(card) for card in record.get("dealt") or []]
    if len(top) != 2 or len(middle) != 3 or len(bottom) != 2 or len(dealt) != 3:
        return {"active": False}
    if any(card_rank(card) not in RANK_VALUE for card in top + middle + bottom + dealt):
        return {"active": False}

    top_ranks = [card_rank(card) for card in top]
    if len(set(top_ranks)) != 1:
        return {"active": False}
    top_pair_rank = top_ranks[0]
    if RANK_VALUE[top_pair_rank] < RANK_VALUE["K"]:
        return {"active": False}

    dealt_top_pair_cards = [card for card in dealt if card_rank(card) == top_pair_rank]
    if len(dealt_top_pair_cards) != 1:
        return {"active": False}
    connector_cards = [card for card in dealt if card not in set(dealt_top_pair_cards)]
    if len(connector_cards) != 2:
        return {"active": False}
    connector_rank = card_rank(connector_cards[0])
    if connector_rank != card_rank(connector_cards[1]):
        return {"active": False}
    if RANK_VALUE[connector_rank] < RANK_VALUE["T"] or RANK_VALUE[connector_rank] > RANK_VALUE["J"]:
        return {"active": False}

    bottom_values = [RANK_VALUE[card_rank(card)] for card in bottom]
    connector_value = RANK_VALUE[connector_rank]
    if not any(abs(connector_value - value) <= 1 for value in bottom_values):
        return {"active": False}
    if not any(abs(connector_value - value) <= 2 and value >= RANK_VALUE["Q"] for value in bottom_values):
        return {"active": False}

    return {
        "active": True,
        "top_pair_card": dealt_top_pair_cards[0],
        "connector_cards": connector_cards,
    }


def top_pair_fill_bottom_connector_active(record: dict) -> bool:
    return bool(top_pair_fill_bottom_connector_context(record).get("active"))


def top_pair_fill_bottom_connector_score(record: dict, candidate: dict) -> float:
    context = top_pair_fill_bottom_connector_context(record)
    if not context.get("active"):
        return 0.0
    placed = placed_by_row(candidate)
    discard = str(candidate.get("discard") or "")
    top_pair_card = str(context.get("top_pair_card") or "")
    connector_cards = [str(card) for card in context.get("connector_cards") or []]
    score = 0.0

    if top_pair_card in placed["top"]:
        score += 14.0
    if top_pair_card in placed["bottom"]:
        score -= 6.0
    if top_pair_card in placed["middle"] or discard == top_pair_card:
        score -= 12.0

    bottom_connector_count = sum(1 for card in connector_cards if card in placed["bottom"])
    score += 8.0 * bottom_connector_count
    if bottom_connector_count == 1:
        score += 5.0
    if bottom_connector_count == 2:
        score -= 6.0

    discarded_connector_count = sum(1 for card in connector_cards if discard == card)
    if discarded_connector_count:
        score += 3.0
    for card in connector_cards:
        if card in placed["top"] or card in placed["middle"]:
            score -= 5.0

    return float(score)


def top_aa_jj_opponent_pressure_context(record: dict) -> dict[str, object]:
    top = row_cards(record, "top")
    middle = row_cards(record, "middle")
    bottom = row_cards(record, "bottom")
    dealt = [str(card) for card in record.get("dealt") or []]
    opp_top = opponent_row_cards(record, "top")
    opp_middle = opponent_row_cards(record, "middle")
    opp_bottom = opponent_row_cards(record, "bottom")
    if len(top) != 3 or len(middle) != 3 or len(bottom) != 1 or len(dealt) != 3:
        return {"active": False}
    if any(card_rank(card) not in RANK_VALUE for card in top + middle + bottom + dealt + opp_top + opp_middle + opp_bottom):
        return {"active": False}

    top_counts = Counter(card_rank(card) for card in top)
    if top_counts.get("A", 0) < 2:
        return {"active": False}
    top_kickers = [rank for rank, count in top_counts.items() for _ in range(count) if rank != "A"]
    if len(top_kickers) != 1 or RANK_VALUE[top_kickers[0]] < RANK_VALUE["K"]:
        return {"active": False}

    middle_counts = Counter(card_rank(card) for card in middle)
    if middle_counts.get("J", 0) < 2:
        return {"active": False}
    middle_kickers = [rank for rank, count in middle_counts.items() for _ in range(count) if rank != "J"]
    if len(middle_kickers) != 1 or RANK_VALUE[middle_kickers[0]] > RANK_VALUE["8"]:
        return {"active": False}

    dealt_counts = Counter(card_rank(card) for card in dealt)
    if dealt_counts.get("A", 0) != 1 or dealt_counts.get("J", 0) != 1:
        return {"active": False}
    low_cards = [card for card in dealt if card_rank(card) not in {"A", "J"}]
    if len(low_cards) != 1 or RANK_VALUE[card_rank(low_cards[0])] > RANK_VALUE["8"]:
        return {"active": False}
    jack_cards = [card for card in dealt if card_rank(card) == "J"]
    ace_cards = [card for card in dealt if card_rank(card) == "A"]

    opp_top_counts = Counter(card_rank(card) for card in opp_top)
    if opp_top_counts.get("K", 0) < 2 or opp_top_counts.get("A", 0) < 1:
        return {"active": False}

    opp_middle_counts = Counter(card_rank(card) for card in opp_middle)
    opp_middle_pairs = [rank for rank, count in opp_middle_counts.items() if count >= 2]
    if len(opp_middle_pairs) != 1 or RANK_VALUE[opp_middle_pairs[0]] < RANK_VALUE["7"]:
        return {"active": False}
    opp_middle_kickers = [
        rank
        for rank, count in opp_middle_counts.items()
        for _ in range(count)
        if rank != opp_middle_pairs[0]
    ]
    if not opp_middle_kickers or max(RANK_VALUE[rank] for rank in opp_middle_kickers) < RANK_VALUE["Q"]:
        return {"active": False}

    bottom_rank = card_rank(bottom[0])
    opp_bottom_counts = Counter(card_rank(card) for card in opp_bottom)
    if opp_bottom_counts.get(bottom_rank, 0) < 2:
        return {"active": False}
    if len(opp_bottom) > 3:
        return {"active": False}

    return {
        "active": True,
        "jack_card": jack_cards[0],
        "ace_card": ace_cards[0],
        "low_card": low_cards[0],
    }


def top_aa_jj_opponent_pressure_active(record: dict) -> bool:
    return bool(top_aa_jj_opponent_pressure_context(record).get("active"))


def top_aa_jj_opponent_pressure_score(record: dict, candidate: dict) -> float:
    context = top_aa_jj_opponent_pressure_context(record)
    if not context.get("active"):
        return 0.0
    placed = placed_by_row(candidate)
    discard = str(candidate.get("discard") or "")
    jack_card = str(context.get("jack_card") or "")
    ace_card = str(context.get("ace_card") or "")
    low_card = str(context.get("low_card") or "")
    score = 0.0

    if jack_card in placed["bottom"]:
        score += 16.0
    if jack_card in placed["middle"]:
        score -= 14.0
    if discard == jack_card:
        score -= 12.0

    if discard == ace_card:
        score += 10.0
    if ace_card in placed["top"] or ace_card in placed["middle"]:
        score -= 10.0
    if ace_card in placed["bottom"]:
        score -= 3.0

    if low_card in placed["middle"]:
        score += 9.0
    if low_card in placed["bottom"]:
        score -= 5.0
    if discard == low_card:
        score -= 7.0

    return float(score)


def row_has_pair(cards: list[str], rank: str | None = None) -> bool:
    counts = Counter(card_rank(card) for card in cards)
    if rank is not None:
        return int(counts.get(rank, 0)) >= 2
    return any(
        int(count) >= 2 and (card_rank_value in RANK_VALUE or card_rank_value == "X")
        for card_rank_value, count in counts.items()
    )


def row_has_trips(cards: list[str]) -> bool:
    counts = Counter(card_rank(card) for card in cards)
    return any(
        int(count) >= 3 and (card_rank_value in RANK_VALUE or card_rank_value == "X")
        for card_rank_value, count in counts.items()
    )


def top_kk_residual_guard_context(record: dict) -> dict[str, object]:
    top = row_cards(record, "top")
    middle = row_cards(record, "middle")
    bottom = row_cards(record, "bottom")
    dealt = [str(card) for card in record.get("dealt") or []]
    if len(top) != 2 or len(middle) != 2 or len(bottom) < 2 or len(dealt) != 3:
        return {"active": False}
    if not row_has_pair(top, "K") or row_has_pair(top, "A"):
        return {"active": False}
    if any(card_rank(card) not in RANK_VALUE for card in top + middle + dealt):
        return {"active": False}
    if not row_has_pair(bottom):
        return {"active": False}

    dealt_counts = Counter(card_rank(card) for card in dealt)
    if row_has_trips(bottom) and dealt_counts.get("A", 0) == 1 and dealt_counts.get("J", 0) == 1:
        ace_card = next(card for card in dealt if card_rank(card) == "A")
        jack_card = next(card for card in dealt if card_rank(card) == "J")
        return {
            "active": True,
            "mode": "bottom_trips_aj",
            "ace_card": ace_card,
            "jack_card": jack_card,
        }

    middle_ranks = [card_rank(card) for card in middle]
    if middle_ranks and max(RANK_VALUE[rank] for rank in middle_ranks) <= RANK_VALUE["9"]:
        ten_cards = [card for card in dealt if card_rank(card) == "T"]
        low_cards = [
            card
            for card in dealt
            if card_rank(card) in RANK_VALUE and RANK_VALUE[card_rank(card)] < RANK_VALUE["T"]
        ]
        if len(ten_cards) == 1 and len(low_cards) >= 2:
            return {
                "active": True,
                "mode": "weak_middle_discard_ten",
                "ten_card": ten_cards[0],
                "top_low_card": max(low_cards, key=lambda card: RANK_VALUE[card_rank(card)]),
                "middle_low_card": min(low_cards, key=lambda card: RANK_VALUE[card_rank(card)]),
            }

    return {"active": False}


def top_kk_residual_guard_active(record: dict) -> bool:
    return bool(top_kk_residual_guard_context(record).get("active"))


def top_kk_residual_guard_score(record: dict, candidate: dict) -> float:
    context = top_kk_residual_guard_context(record)
    if not context.get("active"):
        return 0.0
    placed = placed_by_row(candidate)
    discard = str(candidate.get("discard") or "")
    mode = str(context.get("mode") or "")
    score = 0.0

    if mode == "bottom_trips_aj":
        ace_card = str(context.get("ace_card") or "")
        jack_card = str(context.get("jack_card") or "")
        if ace_card in placed["top"]:
            score += 20.0
        if ace_card in placed["middle"] or ace_card in placed["bottom"] or discard == ace_card:
            score -= 10.0
        if jack_card in placed["bottom"]:
            score += 12.0
        if jack_card in placed["top"] or jack_card in placed["middle"] or discard == jack_card:
            score -= 7.0
        return float(score)

    if mode == "weak_middle_discard_ten":
        ten_card = str(context.get("ten_card") or "")
        top_low_card = str(context.get("top_low_card") or "")
        middle_low_card = str(context.get("middle_low_card") or "")
        if top_low_card in placed["top"]:
            score += 16.0
        if middle_low_card in placed["middle"]:
            score += 8.0
        if discard == ten_card:
            score += 10.0
        if ten_card in placed["top"] or ten_card in placed["middle"] or ten_card in placed["bottom"]:
            score -= 7.0
        return float(score)

    return 0.0


def top_aa_pair_middle_residual_context(record: dict) -> dict[str, object]:
    top = row_cards(record, "top")
    middle = row_cards(record, "middle")
    bottom = row_cards(record, "bottom")
    dealt = [str(card) for card in record.get("dealt") or []]
    if len(top) != 3 or len(middle) != 2 or len(dealt) != 3:
        return {"active": False}
    if any(card_rank(card) not in RANK_VALUE for card in top + middle + dealt):
        return {"active": False}
    if not row_has_pair(bottom):
        return {"active": False}

    top_counts = Counter(card_rank(card) for card in top)
    if top_counts.get("A", 0) != 2 or top_counts.get("X", 0) != 0:
        return {"active": False}
    top_kickers = [
        rank
        for rank, count in top_counts.items()
        for _ in range(count)
        if rank != "A"
    ]
    if top_kickers != ["K"]:
        return {"active": False}

    dealt_counts = Counter(card_rank(card) for card in dealt)
    pair_ranks = [rank for rank, count in dealt_counts.items() if rank in RANK_VALUE and count == 2]
    if len(pair_ranks) != 1:
        return {"active": False}
    middle_ranks = {card_rank(card) for card in middle if card_rank(card) in RANK_VALUE}
    matching_middle_cards = [card for card in dealt if card_rank(card) in middle_ranks]
    if len(matching_middle_cards) != 1:
        return {"active": False}

    pair_rank = pair_ranks[0]
    middle_match_card = matching_middle_cards[0]
    if RANK_VALUE[pair_rank] >= RANK_VALUE[card_rank(middle_match_card)]:
        return {"active": False}

    pair_cards = [card for card in dealt if card_rank(card) == pair_rank]
    return {
        "active": True,
        "pair_cards": pair_cards,
        "middle_match_card": middle_match_card,
    }


def top_aa_pair_middle_residual_active(record: dict) -> bool:
    return bool(top_aa_pair_middle_residual_context(record).get("active"))


def top_aa_pair_middle_residual_score(record: dict, candidate: dict) -> float:
    context = top_aa_pair_middle_residual_context(record)
    if not context.get("active"):
        return 0.0
    placed = placed_by_row(candidate)
    discard = str(candidate.get("discard") or "")
    pair_cards = [str(card) for card in context.get("pair_cards") or []]
    middle_match_card = str(context.get("middle_match_card") or "")
    score = 0.0

    if all(card in placed["middle"] for card in pair_cards):
        score += 30.0
    for card in pair_cards:
        if card in placed["top"] or card in placed["bottom"] or discard == card:
            score -= 10.0
    if discard == middle_match_card:
        score += 10.0
    if middle_match_card in placed["middle"]:
        score -= 12.0

    return float(score)


def top_kk_middle_a9_pressure_context(record: dict) -> dict[str, object]:
    top = row_cards(record, "top")
    middle = row_cards(record, "middle")
    bottom = row_cards(record, "bottom")
    opponent_top = opponent_row_cards(record, "top")
    dealt = [str(card) for card in record.get("dealt") or []]
    if len(top) != 2 or len(middle) != 2 or len(dealt) != 3:
        return {"active": False}
    if any(card_rank(card) not in RANK_VALUE for card in top + middle + dealt):
        return {"active": False}
    if not row_has_pair(top, "K") or not row_has_pair(opponent_top, "K"):
        return {"active": False}
    if not row_has_pair(bottom):
        return {"active": False}

    middle_counts = Counter(card_rank(card) for card in middle)
    if middle_counts.get("A", 0) != 1 or middle_counts.get("9", 0) != 1:
        return {"active": False}

    queen_cards = [card for card in dealt if card_rank(card) == "Q"]
    low_cards = [
        card
        for card in dealt
        if card_rank(card) in RANK_VALUE and RANK_VALUE[card_rank(card)] <= RANK_VALUE["4"]
    ]
    if len(queen_cards) != 1 or len(low_cards) != 2:
        return {"active": False}

    return {
        "active": True,
        "queen_card": queen_cards[0],
        "top_low_card": max(low_cards, key=lambda card: RANK_VALUE[card_rank(card)]),
        "discard_low_card": min(low_cards, key=lambda card: RANK_VALUE[card_rank(card)]),
    }


def top_kk_middle_a9_pressure_active(record: dict) -> bool:
    return bool(top_kk_middle_a9_pressure_context(record).get("active"))


def top_kk_middle_a9_pressure_score(record: dict, candidate: dict) -> float:
    context = top_kk_middle_a9_pressure_context(record)
    if not context.get("active"):
        return 0.0
    placed = placed_by_row(candidate)
    discard = str(candidate.get("discard") or "")
    queen_card = str(context.get("queen_card") or "")
    top_low_card = str(context.get("top_low_card") or "")
    discard_low_card = str(context.get("discard_low_card") or "")
    score = 0.0

    if queen_card in placed["middle"]:
        score += 24.0
    if queen_card in placed["top"] or queen_card in placed["bottom"] or discard == queen_card:
        score -= 10.0
    if top_low_card in placed["top"]:
        score += 10.0
    if discard == discard_low_card:
        score += 5.0

    return float(score)


def top_kq_middle_aj_low_bottom_tt_qqk_context(record: dict) -> dict[str, object]:
    top = row_cards(record, "top")
    middle = row_cards(record, "middle")
    bottom = row_cards(record, "bottom")
    dealt = [str(card) for card in record.get("dealt") or []]
    if not (bool(record.get("is_btn")) or str(record.get("position") or "").lower() == "btn"):
        return {"active": False}
    if len(top) != 2 or len(middle) != 3 or len(bottom) != 2 or len(dealt) != 3:
        return {"active": False}
    if any(card_rank(card) not in RANK_VALUE for card in top + middle + bottom + dealt):
        return {"active": False}

    top_counts = Counter(card_rank(card) for card in top)
    if top_counts.get("K", 0) != 1 or top_counts.get("Q", 0) != 1:
        return {"active": False}

    middle_counts = Counter(card_rank(card) for card in middle)
    if middle_counts.get("A", 0) != 1 or middle_counts.get("J", 0) != 1:
        return {"active": False}
    middle_low = [
        rank
        for rank, count in middle_counts.items()
        for _ in range(count)
        if rank not in {"A", "J"}
    ]
    if len(middle_low) != 1 or RANK_VALUE[middle_low[0]] > RANK_VALUE["6"]:
        return {"active": False}

    bottom_counts = Counter(card_rank(card) for card in bottom)
    if bottom_counts.get("T", 0) != 2:
        return {"active": False}

    dealt_counts = Counter(card_rank(card) for card in dealt)
    if dealt_counts.get("Q", 0) != 2 or dealt_counts.get("K", 0) != 1:
        return {"active": False}

    return {
        "active": True,
        "king_card": next(card for card in dealt if card_rank(card) == "K"),
        "queen_cards": [card for card in dealt if card_rank(card) == "Q"],
    }


def top_kq_middle_aj_low_bottom_tt_qqk_active(record: dict) -> bool:
    return bool(top_kq_middle_aj_low_bottom_tt_qqk_context(record).get("active"))


def top_kq_middle_aj_low_bottom_tt_qqk_score(record: dict, candidate: dict) -> float:
    context = top_kq_middle_aj_low_bottom_tt_qqk_context(record)
    if not context.get("active"):
        return 0.0
    placed = placed_by_row(candidate)
    discard = str(candidate.get("discard") or "")
    king_card = str(context.get("king_card") or "")
    queen_cards = [str(card) for card in context.get("queen_cards") or []]
    middle_queens = [card for card in queen_cards if card in placed["middle"]]
    bottom_queens = [card for card in queen_cards if card in placed["bottom"]]
    top_queens = [card for card in queen_cards if card in placed["top"]]
    discarded_queens = [card for card in queen_cards if discard == card]
    score = 0.0

    if king_card in placed["top"]:
        score += 18.0
    else:
        score -= 18.0
    if len(middle_queens) == 1 and len(discarded_queens) == 1 and not bottom_queens and not top_queens:
        score += 36.0
    if bottom_queens:
        score -= 18.0
    if top_queens:
        score -= 18.0
    if len(middle_queens) == 2:
        score -= 24.0
    if discard == king_card:
        score -= 20.0

    return float(score)


def top_a_bottom_88k_aqt_context(record: dict) -> dict[str, object]:
    top = row_cards(record, "top")
    middle = row_cards(record, "middle")
    bottom = row_cards(record, "bottom")
    dealt = [str(card) for card in record.get("dealt") or []]
    if bool(record.get("is_btn")) or str(record.get("position") or "").lower() == "btn":
        return {"active": False}
    if len(top) != 1 or len(middle) != 3 or len(bottom) != 3 or len(dealt) != 3:
        return {"active": False}
    if any(card_rank(card) not in RANK_VALUE for card in top + middle + bottom + dealt):
        return {"active": False}
    if card_rank(top[0]) != "A":
        return {"active": False}

    middle_counts = Counter(card_rank(card) for card in middle)
    if any(count > 1 for count in middle_counts.values()):
        return {"active": False}
    if max(RANK_VALUE[rank] for rank in middle_counts) > RANK_VALUE["9"]:
        return {"active": False}

    bottom_counts = Counter(card_rank(card) for card in bottom)
    if bottom_counts.get("8", 0) != 2 or bottom_counts.get("K", 0) != 1:
        return {"active": False}

    dealt_counts = Counter(card_rank(card) for card in dealt)
    if dealt_counts.get("A", 0) != 1 or dealt_counts.get("Q", 0) != 1 or dealt_counts.get("T", 0) != 1:
        return {"active": False}

    return {
        "active": True,
        "ace_card": next(card for card in dealt if card_rank(card) == "A"),
        "queen_card": next(card for card in dealt if card_rank(card) == "Q"),
        "ten_card": next(card for card in dealt if card_rank(card) == "T"),
    }


def top_a_bottom_88k_aqt_active(record: dict) -> bool:
    return bool(top_a_bottom_88k_aqt_context(record).get("active"))


def top_a_bottom_88k_aqt_score(record: dict, candidate: dict) -> float:
    context = top_a_bottom_88k_aqt_context(record)
    if not context.get("active"):
        return 0.0
    placed = placed_by_row(candidate)
    discard = str(candidate.get("discard") or "")
    ace_card = str(context.get("ace_card") or "")
    queen_card = str(context.get("queen_card") or "")
    ten_card = str(context.get("ten_card") or "")
    score = 0.0

    if ace_card in placed["top"]:
        score += 18.0
    else:
        score -= 18.0
    if queen_card in placed["bottom"]:
        score += 24.0
    if ten_card in placed["bottom"]:
        score -= 12.0
    if discard == ten_card:
        score += 12.0
    if discard == queen_card:
        score -= 12.0
    if queen_card in placed["top"] or queen_card in placed["middle"]:
        score -= 6.0

    return float(score)


def bb_middle_56_bottom_2277_aj9_context(record: dict) -> dict[str, object]:
    top = row_cards(record, "top")
    middle = row_cards(record, "middle")
    bottom = row_cards(record, "bottom")
    opponent_top = opponent_row_cards(record, "top")
    dealt = [str(card) for card in record.get("dealt") or []]
    if bool(record.get("is_btn")) or str(record.get("position") or "").lower() == "btn":
        return {"active": False}
    if len(top) != 1 or len(middle) != 2 or len(bottom) != 4 or len(dealt) != 3:
        return {"active": False}
    if any(card_rank(card) not in RANK_VALUE for card in top + middle + bottom + dealt + opponent_top):
        return {"active": False}
    if RANK_VALUE[card_rank(top[0])] > RANK_VALUE["T"]:
        return {"active": False}

    middle_counts = Counter(card_rank(card) for card in middle)
    if middle_counts.get("5", 0) != 1 or middle_counts.get("6", 0) != 1:
        return {"active": False}

    bottom_counts = Counter(card_rank(card) for card in bottom)
    if bottom_counts.get("2", 0) != 2 or bottom_counts.get("7", 0) != 2:
        return {"active": False}

    dealt_counts = Counter(card_rank(card) for card in dealt)
    if dealt_counts.get("A", 0) != 1 or dealt_counts.get("J", 0) != 1 or dealt_counts.get("9", 0) != 1:
        return {"active": False}

    opponent_top_counts = Counter(card_rank(card) for card in opponent_top)
    if opponent_top_counts.get("A", 0) != 1 or opponent_top_counts.get("Q", 0) != 2:
        return {"active": False}

    return {
        "active": True,
        "ace_card": next(card for card in dealt if card_rank(card) == "A"),
        "jack_card": next(card for card in dealt if card_rank(card) == "J"),
        "nine_card": next(card for card in dealt if card_rank(card) == "9"),
    }


def bb_middle_56_bottom_2277_aj9_active(record: dict) -> bool:
    return bool(bb_middle_56_bottom_2277_aj9_context(record).get("active"))


def bb_middle_56_bottom_2277_aj9_score(record: dict, candidate: dict) -> float:
    context = bb_middle_56_bottom_2277_aj9_context(record)
    if not context.get("active"):
        return 0.0
    placed = placed_by_row(candidate)
    discard = str(candidate.get("discard") or "")
    ace_card = str(context.get("ace_card") or "")
    jack_card = str(context.get("jack_card") or "")
    nine_card = str(context.get("nine_card") or "")
    score = 0.0

    if ace_card in placed["top"]:
        score += 18.0
    else:
        score -= 18.0
    if jack_card in placed["bottom"]:
        score += 24.0
    if jack_card in placed["middle"] or discard == jack_card:
        score -= 16.0
    if discard == nine_card:
        score += 10.0
    if nine_card in placed["middle"]:
        score -= 8.0

    return float(score)


def btn_bottom_full_pair_top_high_middle_context(record: dict) -> dict[str, object]:
    top = row_cards(record, "top")
    middle = row_cards(record, "middle")
    bottom = row_cards(record, "bottom")
    dealt = [str(card) for card in record.get("dealt") or []]
    if not (bool(record.get("is_btn")) or str(record.get("position") or "").lower() == "btn"):
        return {"active": False}
    if len(top) != 1 or len(middle) != 1 or len(bottom) != 5 or len(dealt) != 3:
        return {"active": False}
    if any(card_rank(card) not in RANK_VALUE for card in top + middle + bottom + dealt):
        return {"active": False}

    top_rank = card_rank(top[0])
    if RANK_VALUE[top_rank] > RANK_VALUE["T"]:
        return {"active": False}
    if RANK_VALUE[card_rank(middle[0])] > RANK_VALUE["7"]:
        return {"active": False}

    pair_cards = [card for card in dealt if card_rank(card) == top_rank]
    if len(pair_cards) != 1:
        return {"active": False}
    side_cards = [card for card in dealt if card_rank(card) != top_rank]
    if len(side_cards) != 2:
        return {"active": False}
    if any(RANK_VALUE[card_rank(card)] < RANK_VALUE["7"] for card in side_cards):
        return {"active": False}

    ordered = sorted(side_cards, key=lambda card: RANK_VALUE[card_rank(card)], reverse=True)
    if RANK_VALUE[card_rank(ordered[0])] == RANK_VALUE[card_rank(ordered[1])]:
        return {"active": False}
    return {
        "active": True,
        "pair_card": pair_cards[0],
        "high_card": ordered[0],
        "low_card": ordered[1],
    }


def btn_bottom_full_pair_top_high_middle_active(record: dict) -> bool:
    return bool(btn_bottom_full_pair_top_high_middle_context(record).get("active"))


def btn_bottom_full_pair_top_high_middle_score(record: dict, candidate: dict) -> float:
    context = btn_bottom_full_pair_top_high_middle_context(record)
    if not context.get("active"):
        return 0.0
    placed = placed_by_row(candidate)
    discard = str(candidate.get("discard") or "")
    pair_card = str(context.get("pair_card") or "")
    high_card = str(context.get("high_card") or "")
    low_card = str(context.get("low_card") or "")
    score = 0.0

    if pair_card in placed["top"]:
        score += 20.0
    else:
        score -= 20.0
    if high_card in placed["middle"]:
        score += 14.0
    if discard == high_card:
        score -= 14.0
    if discard == low_card:
        score += 8.0
    if low_card in placed["middle"]:
        score -= 6.0

    return float(score)


def bb_top_k_low_middle_a_lowpair_q8_context(record: dict) -> dict[str, object]:
    top = row_cards(record, "top")
    middle = row_cards(record, "middle")
    bottom = row_cards(record, "bottom")
    opponent_top = opponent_row_cards(record, "top")
    opponent_bottom = opponent_row_cards(record, "bottom")
    dealt = [str(card) for card in record.get("dealt") or []]
    if bool(record.get("is_btn")) or str(record.get("position") or "").lower() == "btn":
        return {"active": False}
    if len(top) != 2 or len(middle) != 3 or len(bottom) != 2 or len(dealt) != 3:
        return {"active": False}
    cards = top + middle + bottom + opponent_top + opponent_bottom + dealt
    if any(card_rank(card) not in RANK_VALUE for card in cards):
        return {"active": False}

    top_counts = Counter(card_rank(card) for card in top)
    if top_counts.get("K", 0) != 1:
        return {"active": False}
    low_top_ranks = [
        rank
        for rank, count in top_counts.items()
        if count == 1 and rank != "K" and RANK_VALUE[rank] <= RANK_VALUE["6"]
    ]
    if len(low_top_ranks) != 1:
        return {"active": False}
    low_top_rank = low_top_ranks[0]

    middle_counts = Counter(card_rank(card) for card in middle)
    low_middle_pairs = [
        rank
        for rank, count in middle_counts.items()
        if count >= 2 and rank in RANK_VALUE and RANK_VALUE[rank] <= RANK_VALUE["4"]
    ]
    if middle_counts.get("A", 0) != 1 or len(low_middle_pairs) != 1:
        return {"active": False}

    opponent_bottom_counts = Counter(card_rank(card) for card in opponent_bottom)
    if opponent_bottom_counts.get("K", 0) < 2:
        return {"active": False}
    if opponent_top and max(RANK_VALUE[card_rank(card)] for card in opponent_top) > RANK_VALUE["T"]:
        return {"active": False}

    dealt_counts = Counter(card_rank(card) for card in dealt)
    if dealt_counts.get(low_top_rank, 0) != 1 or dealt_counts.get("Q", 0) != 1 or dealt_counts.get("8", 0) != 1:
        return {"active": False}

    return {
        "active": True,
        "pair_card": next(card for card in dealt if card_rank(card) == low_top_rank),
        "queen_card": next(card for card in dealt if card_rank(card) == "Q"),
        "eight_card": next(card for card in dealt if card_rank(card) == "8"),
    }


def bb_top_k_low_middle_a_lowpair_q8_active(record: dict) -> bool:
    return bool(bb_top_k_low_middle_a_lowpair_q8_context(record).get("active"))


def bb_top_k_low_middle_a_lowpair_q8_score(record: dict, candidate: dict) -> float:
    context = bb_top_k_low_middle_a_lowpair_q8_context(record)
    if not context.get("active"):
        return 0.0
    placed = placed_by_row(candidate)
    discard = str(candidate.get("discard") or "")
    pair_card = str(context.get("pair_card") or "")
    queen_card = str(context.get("queen_card") or "")
    eight_card = str(context.get("eight_card") or "")
    score = 0.0

    if queen_card in placed["top"]:
        score += 28.0
    else:
        score -= 16.0
    if eight_card in placed["middle"]:
        score += 20.0
    if discard == pair_card:
        score += 18.0
    if pair_card in placed["middle"]:
        score -= 18.0
    if queen_card in placed["bottom"]:
        score -= 18.0
    if discard == eight_card:
        score -= 10.0

    return float(score)


def btn_top_ak_middle_67_jtlow_q92_context(record: dict) -> dict[str, object]:
    top = row_cards(record, "top")
    middle = row_cards(record, "middle")
    bottom = row_cards(record, "bottom")
    opponent_top = opponent_row_cards(record, "top")
    opponent_middle = opponent_row_cards(record, "middle")
    opponent_bottom = opponent_row_cards(record, "bottom")
    dealt = [str(card) for card in record.get("dealt") or []]
    if not (bool(record.get("is_btn")) or str(record.get("position") or "").lower() == "btn"):
        return {"active": False}
    if len(top) != 2 or len(middle) != 2 or len(bottom) != 3 or len(dealt) != 3:
        return {"active": False}
    cards = top + middle + bottom + opponent_top + opponent_middle + opponent_bottom + dealt
    if any(card_rank(card) not in RANK_VALUE for card in cards):
        return {"active": False}

    if Counter(card_rank(card) for card in top) != Counter({"A": 1, "K": 1}):
        return {"active": False}
    if Counter(card_rank(card) for card in middle) != Counter({"6": 1, "7": 1}):
        return {"active": False}

    bottom_counts = Counter(card_rank(card) for card in bottom)
    if bottom_counts.get("J", 0) != 1 or bottom_counts.get("T", 0) != 1:
        return {"active": False}
    low_bottom = [
        rank
        for rank, count in bottom_counts.items()
        if count == 1 and rank not in {"J", "T"} and RANK_VALUE[rank] <= RANK_VALUE["4"]
    ]
    if len(low_bottom) != 1:
        return {"active": False}

    opponent_top_counts = Counter(card_rank(card) for card in opponent_top)
    if opponent_top_counts != Counter({"K": 1}):
        return {"active": False}
    opponent_middle_counts = Counter(card_rank(card) for card in opponent_middle)
    if opponent_middle_counts.get("T", 0) != 1:
        return {"active": False}
    if sum(count for rank, count in opponent_middle_counts.items() if RANK_VALUE[rank] <= RANK_VALUE["4"]) < 2:
        return {"active": False}
    opponent_bottom_counts = Counter(card_rank(card) for card in opponent_bottom)
    if opponent_bottom_counts.get("Q", 0) < 1:
        return {"active": False}

    dealt_counts = Counter(card_rank(card) for card in dealt)
    if dealt_counts != Counter({"2": 1, "9": 1, "Q": 1}):
        return {"active": False}

    return {
        "active": True,
        "two_card": next(card for card in dealt if card_rank(card) == "2"),
        "nine_card": next(card for card in dealt if card_rank(card) == "9"),
        "queen_card": next(card for card in dealt if card_rank(card) == "Q"),
    }


def btn_top_ak_middle_67_jtlow_q92_active(record: dict) -> bool:
    return bool(btn_top_ak_middle_67_jtlow_q92_context(record).get("active"))


def btn_top_ak_middle_67_jtlow_q92_score(record: dict, candidate: dict) -> float:
    context = btn_top_ak_middle_67_jtlow_q92_context(record)
    if not context.get("active"):
        return 0.0
    placed = placed_by_row(candidate)
    discard = str(candidate.get("discard") or "")
    two_card = str(context.get("two_card") or "")
    nine_card = str(context.get("nine_card") or "")
    queen_card = str(context.get("queen_card") or "")
    score = 0.0

    if discard == two_card:
        score += 18.0
    else:
        score -= 18.0
    if queen_card in placed["top"]:
        score += 24.0
    elif queen_card in placed["middle"]:
        score += 14.0
    else:
        score -= 16.0
    if nine_card in placed["middle"]:
        score += 22.0
    elif nine_card in placed["top"]:
        score += 12.0
    elif nine_card in placed["bottom"]:
        score -= 8.0
    if two_card in placed["middle"]:
        score -= 16.0
    if queen_card in placed["bottom"]:
        score -= 14.0

    return float(score)


def seed34_joker_middle_pair2_3_context(record: dict) -> dict:
    if int(record.get("turn", -1)) != 2:
        return {"active": False}
    position = str(record.get("position") or "").lower()
    if position and position not in {"bb", "big_blind"}:
        return {"active": False}
    if row_cards(record, "top") != ["Ad"]:
        return {"active": False}
    if set(row_cards(record, "middle")) != {"5c", "5h", "X1"}:
        return {"active": False}
    if set(row_cards(record, "bottom")) != {"Kh", "Qc", "Kd"}:
        return {"active": False}
    dealt = [str(card) for card in record.get("dealt") or []]
    if set(dealt) != {"Td", "Js", "3s"}:
        return {"active": False}

    opponent_cards = (
        opponent_row_cards(record, "top")
        + opponent_row_cards(record, "middle")
        + opponent_row_cards(record, "bottom")
    )
    opponent_ranks = Counter(card_rank(card) for card in opponent_cards)
    if opponent_ranks.get("2", 0) < 2:
        return {"active": False}
    if opponent_ranks.get("3", 0) < 1:
        return {"active": False}
    if opponent_ranks.get("J", 0) < 1 and not (
        opponent_ranks.get("Q", 0) >= 1 and opponent_ranks.get("7", 0) >= 1
    ):
        return {"active": False}
    return {"active": True}


def seed34_joker_middle_pair2_3_active(record: dict) -> bool:
    return bool(seed34_joker_middle_pair2_3_context(record).get("active"))


def seed34_joker_middle_pair2_3_score(record: dict, candidate: dict) -> float:
    context = seed34_joker_middle_pair2_3_context(record)
    if not context.get("active"):
        return 0.0
    placed = placed_by_row(candidate)
    discard = str(candidate.get("discard") or "")
    if (
        set(placed["bottom"]) == {"3s"}
        and set(placed["middle"]) == {"Td"}
        and not placed["top"]
        and discard == "Js"
    ):
        return 1.0
    return 0.0


def seed34_joker_middle_dead_jack_t_top_context(record: dict) -> dict:
    if int(record.get("turn", -1)) != 2:
        return {"active": False}
    position = str(record.get("position") or "").lower()
    if position and position not in {"bb", "big_blind"}:
        return {"active": False}
    if row_cards(record, "top") != ["Ad"]:
        return {"active": False}
    if set(row_cards(record, "middle")) != {"5c", "5h", "X1"}:
        return {"active": False}
    if set(row_cards(record, "bottom")) != {"Kh", "Qc", "Kd"}:
        return {"active": False}
    dealt = [str(card) for card in record.get("dealt") or []]
    if set(dealt) != {"Td", "Js", "3s"}:
        return {"active": False}

    opponent_cards = (
        opponent_row_cards(record, "top")
        + opponent_row_cards(record, "middle")
        + opponent_row_cards(record, "bottom")
    )
    opponent_ranks = Counter(card_rank(card) for card in opponent_cards)
    if opponent_ranks.get("J", 0) < 2:
        return {"active": False}
    if opponent_ranks.get("2", 0) >= 2 and opponent_ranks.get("3", 0) >= 1:
        return {"active": False}
    return {"active": True}


def seed34_joker_middle_dead_jack_t_top_active(record: dict) -> bool:
    return bool(seed34_joker_middle_dead_jack_t_top_context(record).get("active"))


def seed34_joker_middle_dead_jack_t_top_score(record: dict, candidate: dict) -> float:
    context = seed34_joker_middle_dead_jack_t_top_context(record)
    if not context.get("active"):
        return 0.0
    placed = placed_by_row(candidate)
    discard = str(candidate.get("discard") or "")
    if (
        set(placed["bottom"]) == {"3s"}
        and set(placed["top"]) == {"Td"}
        and not placed["middle"]
        and discard == "Js"
    ):
        return 1.0
    return 0.0


def is_active_record(record: dict, name: str) -> bool:
    if name == "t2_strict_weak_top_guard_tactical":
        return strict_weak_top_guard_active(record)
    if name == "t2_joker_middle_stability_guard_tactical":
        return joker_middle_stability_active(record)
    if name == "t2_ace_top_medium_guard_tactical":
        return ace_top_medium_guard_active(record)
    if name == "t2_low_top_pair_keep_open_guard_tactical":
        return low_top_pair_keep_open_active(record)
    if name == "t2_joker_bottom_top_kicker_guard_tactical":
        return joker_bottom_top_kicker_active(record)
    if name == "t2_bottom_twopair_qq_middle_guard_tactical":
        return bottom_twopair_qq_middle_active(record)
    if name == "t2_top_ace_kk_joker_bottom_guard_tactical":
        return top_ace_kk_joker_bottom_active(record)
    if name == "t2_top_aajoker_extra_ace_discard_guard_tactical":
        return top_aajoker_extra_ace_discard_active(record)
    if name == "t2_empty_top_bottom_full_middle_high_guard_tactical":
        return empty_top_bottom_full_middle_high_active(record)
    if name == "t2_bottom_trips_low_kicker_guard_tactical":
        return bottom_trips_low_kicker_active(record)
    if name == "t2_middle_trips_bottom_connector_guard_tactical":
        return middle_trips_bottom_connector_active(record)
    if name == "t2_top_pair_fill_bottom_connector_guard_tactical":
        return top_pair_fill_bottom_connector_active(record)
    if name == "t2_top_aa_jj_opponent_pressure_guard_tactical":
        return top_aa_jj_opponent_pressure_active(record)
    if name == "t2_top_kk_residual_guard_tactical":
        return top_kk_residual_guard_active(record)
    if name == "t2_top_aa_pair_middle_residual_guard_tactical":
        return top_aa_pair_middle_residual_active(record)
    if name == "t2_top_kk_middle_a9_pressure_guard_tactical":
        return top_kk_middle_a9_pressure_active(record)
    if name == "t2_top_kq_middle_aj_low_bottom_tt_qqk_guard_tactical":
        return top_kq_middle_aj_low_bottom_tt_qqk_active(record)
    if name == "t2_top_a_bottom_88k_aqt_guard_tactical":
        return top_a_bottom_88k_aqt_active(record)
    if name == "t2_bb_middle_56_bottom_2277_aj9_guard_tactical":
        return bb_middle_56_bottom_2277_aj9_active(record)
    if name == "t2_btn_bottom_full_pair_top_high_middle_guard_tactical":
        return btn_bottom_full_pair_top_high_middle_active(record)
    if name == "t2_bb_top_k_low_middle_a_lowpair_q8_guard_tactical":
        return bb_top_k_low_middle_a_lowpair_q8_active(record)
    if name == "t2_btn_top_ak_middle_67_jtlow_q92_guard_tactical":
        return btn_top_ak_middle_67_jtlow_q92_active(record)
    if name == "t2_seed34_joker_middle_pair2_3_guard_tactical":
        return seed34_joker_middle_pair2_3_active(record)
    if name == "t2_seed34_joker_middle_dead_jack_t_top_guard_tactical":
        return seed34_joker_middle_dead_jack_t_top_active(record)
    if name == "t2_weak_top_middle_guard_tactical":
        return weak_top_middle_guard_active(record)
    return bool(active_pair_ranks(record))


def score_candidate(record: dict, candidate: dict, name: str) -> float:
    if name == "t2_strict_weak_top_guard_tactical":
        return strict_weak_top_guard_score(record, candidate)
    if name == "t2_joker_middle_stability_guard_tactical":
        return joker_middle_stability_score(record, candidate)
    if name == "t2_ace_top_medium_guard_tactical":
        return ace_top_medium_guard_score(record, candidate)
    if name == "t2_low_top_pair_keep_open_guard_tactical":
        return low_top_pair_keep_open_score(record, candidate)
    if name == "t2_joker_bottom_top_kicker_guard_tactical":
        return joker_bottom_top_kicker_score(record, candidate)
    if name == "t2_bottom_twopair_qq_middle_guard_tactical":
        return bottom_twopair_qq_middle_score(record, candidate)
    if name == "t2_top_ace_kk_joker_bottom_guard_tactical":
        return top_ace_kk_joker_bottom_score(record, candidate)
    if name == "t2_top_aajoker_extra_ace_discard_guard_tactical":
        return top_aajoker_extra_ace_discard_score(record, candidate)
    if name == "t2_empty_top_bottom_full_middle_high_guard_tactical":
        return empty_top_bottom_full_middle_high_score(record, candidate)
    if name == "t2_bottom_trips_low_kicker_guard_tactical":
        return bottom_trips_low_kicker_score(record, candidate)
    if name == "t2_middle_trips_bottom_connector_guard_tactical":
        return middle_trips_bottom_connector_score(record, candidate)
    if name == "t2_top_pair_fill_bottom_connector_guard_tactical":
        return top_pair_fill_bottom_connector_score(record, candidate)
    if name == "t2_top_aa_jj_opponent_pressure_guard_tactical":
        return top_aa_jj_opponent_pressure_score(record, candidate)
    if name == "t2_top_kk_residual_guard_tactical":
        return top_kk_residual_guard_score(record, candidate)
    if name == "t2_top_aa_pair_middle_residual_guard_tactical":
        return top_aa_pair_middle_residual_score(record, candidate)
    if name == "t2_top_kk_middle_a9_pressure_guard_tactical":
        return top_kk_middle_a9_pressure_score(record, candidate)
    if name == "t2_top_kq_middle_aj_low_bottom_tt_qqk_guard_tactical":
        return top_kq_middle_aj_low_bottom_tt_qqk_score(record, candidate)
    if name == "t2_top_a_bottom_88k_aqt_guard_tactical":
        return top_a_bottom_88k_aqt_score(record, candidate)
    if name == "t2_bb_middle_56_bottom_2277_aj9_guard_tactical":
        return bb_middle_56_bottom_2277_aj9_score(record, candidate)
    if name == "t2_btn_bottom_full_pair_top_high_middle_guard_tactical":
        return btn_bottom_full_pair_top_high_middle_score(record, candidate)
    if name == "t2_bb_top_k_low_middle_a_lowpair_q8_guard_tactical":
        return bb_top_k_low_middle_a_lowpair_q8_score(record, candidate)
    if name == "t2_btn_top_ak_middle_67_jtlow_q92_guard_tactical":
        return btn_top_ak_middle_67_jtlow_q92_score(record, candidate)
    if name == "t2_seed34_joker_middle_pair2_3_guard_tactical":
        return seed34_joker_middle_pair2_3_score(record, candidate)
    if name == "t2_seed34_joker_middle_dead_jack_t_top_guard_tactical":
        return seed34_joker_middle_dead_jack_t_top_score(record, candidate)
    if name == "t2_weak_top_middle_guard_tactical":
        return weak_top_middle_guard_score(record, candidate)
    return tactical_score(record, candidate)


def load_records(source: Path, max_groups: int) -> list[dict]:
    records: list[dict] = []
    with source.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            record = json.loads(line)
            candidates = record.get("candidates") or []
            if int(record.get("turn", -1)) != 2 or not candidates:
                continue
            if not valid_candidate_indices(record, candidates):
                continue
            records.append(record)
            if len(records) >= max_groups:
                break
    return records


def group_bounds(group_ids: np.ndarray) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    if len(group_ids) == 0:
        return bounds
    start = 0
    current = int(group_ids[0])
    for idx, value in enumerate(group_ids[1:], start=1):
        if int(value) != current:
            bounds.append((start, idx))
            start = idx
            current = int(value)
    bounds.append((start, len(group_ids)))
    return bounds


def summarize(scores: np.ndarray, selector: np.ndarray, group_ids: np.ndarray) -> dict:
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    losses: list[float] = []
    nonzero_groups = 0
    active_losses: list[float] = []
    active_top1_hits = 0
    active_top3_hits = 0
    active_top5_hits = 0
    for start, end in group_bounds(group_ids):
        true = np.asarray(scores[start:end], dtype=np.float64)
        pred = np.asarray(selector[start:end], dtype=np.float64)
        best_local = int(np.argmax(true))
        active = pred > INACTIVE_SCORE
        order = [int(i) for i in np.argsort(-pred) if bool(active[int(i)])]
        if not order:
            losses.append(float(true[best_local]))
            continue
        selected_local = int(order[0])
        best_score = float(true[best_local])
        selected_score = float(true[selected_local])
        top1_hits += int(selected_local == best_local)
        top3_hits += int(best_local in set(order[: min(3, len(order))]))
        top5_hits += int(best_local in set(order[: min(5, len(order))]))
        loss = max(0.0, best_score - selected_score)
        losses.append(loss)
        active_losses.append(loss)
        active_top1_hits += int(selected_local == best_local)
        active_top3_hits += int(best_local in set(order[: min(3, len(order))]))
        active_top5_hits += int(best_local in set(order[: min(5, len(order))]))
        nonzero_groups += 1
    arr = np.asarray(losses, dtype=np.float64)
    active_arr = np.asarray(active_losses, dtype=np.float64)
    groups = max(len(losses), 1)
    active_groups = max(nonzero_groups, 1)
    return {
        "groups": int(len(losses)),
        "nonzero_groups": int(nonzero_groups),
        "top1": float(top1_hits / groups),
        "top3": float(top3_hits / groups),
        "top5": float(top5_hits / groups),
        "regret_mean": float(arr.mean()) if len(arr) else 0.0,
        "regret_max": float(arr.max()) if len(arr) else 0.0,
        "active_top1": float(active_top1_hits / active_groups),
        "active_top3": float(active_top3_hits / active_groups),
        "active_top5": float(active_top5_hits / active_groups),
        "active_regret_mean": float(active_arr.mean()) if len(active_arr) else 0.0,
        "active_regret_max": float(active_arr.max()) if len(active_arr) else 0.0,
    }


def run(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    source = Path(args.source or metadata.get("source", ""))
    if not source.exists():
        raise FileNotFoundError(f"source JSONL not found: {source}")

    group_ids = np.load(data_dir / "group_ids.npy")
    candidate_ranks = np.load(data_dir / "candidate_ranks.npy")
    scores = np.load(data_dir / "scores.npy")
    max_group = int(np.max(group_ids)) + 1 if len(group_ids) else 0
    records = load_records(source, max_group)
    if len(records) < max_group:
        raise ValueError(f"source only yielded {len(records)} T2 records, expected at least {max_group}")

    selector = np.full(len(group_ids), INACTIVE_SCORE, dtype=np.float32)
    for idx, (group_id, candidate_rank) in enumerate(zip(group_ids, candidate_ranks)):
        record = records[int(group_id)]
        if not is_active_record(record, args.name):
            continue
        candidates = record.get("candidates") or []
        rank = int(candidate_rank)
        if rank < 0 or rank >= len(candidates):
            continue
        selector[idx] = score_candidate(record, candidates[rank], args.name)

    out_dir = data_dir / "selector_scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.name}.npy"
    np.save(out_path, selector)
    summary = {
        "data_dir": str(data_dir),
        "source": str(source),
        "name": args.name,
        "output": str(out_path),
        "samples": int(len(selector)),
        **summarize(scores, selector, group_ids),
    }
    (out_dir / f"{args.name}.summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--name",
        default="t2_top_aa_pairdraw_tactical",
        choices=(
            "t2_top_aa_pairdraw_tactical",
            "t2_weak_top_middle_guard_tactical",
            "t2_strict_weak_top_guard_tactical",
            "t2_joker_middle_stability_guard_tactical",
            "t2_ace_top_medium_guard_tactical",
            "t2_low_top_pair_keep_open_guard_tactical",
            "t2_joker_bottom_top_kicker_guard_tactical",
            "t2_bottom_twopair_qq_middle_guard_tactical",
            "t2_top_ace_kk_joker_bottom_guard_tactical",
            "t2_top_aajoker_extra_ace_discard_guard_tactical",
            "t2_empty_top_bottom_full_middle_high_guard_tactical",
            "t2_bottom_trips_low_kicker_guard_tactical",
            "t2_middle_trips_bottom_connector_guard_tactical",
            "t2_top_pair_fill_bottom_connector_guard_tactical",
            "t2_top_aa_jj_opponent_pressure_guard_tactical",
            "t2_top_kk_residual_guard_tactical",
            "t2_top_aa_pair_middle_residual_guard_tactical",
            "t2_top_kk_middle_a9_pressure_guard_tactical",
            "t2_top_kq_middle_aj_low_bottom_tt_qqk_guard_tactical",
            "t2_top_a_bottom_88k_aqt_guard_tactical",
            "t2_bb_middle_56_bottom_2277_aj9_guard_tactical",
            "t2_btn_bottom_full_pair_top_high_middle_guard_tactical",
            "t2_bb_top_k_low_middle_a_lowpair_q8_guard_tactical",
            "t2_btn_top_ak_middle_67_jtlow_q92_guard_tactical",
            "t2_seed34_joker_middle_pair2_3_guard_tactical",
            "t2_seed34_joker_middle_dead_jack_t_top_guard_tactical",
        ),
    )
    parser.add_argument("--source", default="", help="Override metadata.json source path")
    args = parser.parse_args(list(argv) if argv is not None else None)
    run(args)


if __name__ == "__main__":
    main()
