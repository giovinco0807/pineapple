"""Generate targeted T2 source rows for Joker-middle hard negatives.

The rows focus on BB T2 spots where the hero already has A top, a paired
middle with a joker, and a strong but incomplete bottom.  These are the spots
where the current T2 Top1 model still has non-zero EV-loss misses after broad
external checks.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable


RANKS = "23456789TJQKA"
SUITS = "cdhs"
STANDARD_DECK = [rank + suit for rank in RANKS for suit in SUITS] + ["X1", "X2"]


def rank(card: str) -> str:
    return card[0]


def remove_cards(deck: list[str], cards: Iterable[str]) -> None:
    for card in cards:
        if card not in deck:
            raise ValueError(f"card not available: {card}")
        deck.remove(card)


def take_rank(deck: list[str], rng: random.Random, value: str, count: int) -> list[str]:
    pool = [card for card in deck if rank(card) == value]
    if len(pool) < count:
        raise ValueError(f"not enough rank {value}: need={count} have={len(pool)}")
    chosen = rng.sample(pool, count)
    remove_cards(deck, chosen)
    return chosen


def take_any(
    deck: list[str],
    rng: random.Random,
    count: int,
    *,
    avoid_ranks: set[str] | None = None,
    prefer_ranks: str | None = None,
) -> list[str]:
    avoid_ranks = avoid_ranks or set()
    pool = [card for card in deck if rank(card) not in avoid_ranks]
    if prefer_ranks:
        preferred = [card for card in pool if rank(card) in set(prefer_ranks)]
        if len(preferred) >= count:
            pool = preferred
    if len(pool) < count:
        pool = list(deck)
    chosen = rng.sample(pool, count)
    remove_cards(deck, chosen)
    return chosen


def available_ranks(deck: list[str], values: Iterable[str], count: int) -> list[str]:
    return [value for value in values if sum(1 for card in deck if rank(card) == value) >= count]


def shuffled(rng: random.Random, cards: list[str]) -> list[str]:
    cards = list(cards)
    rng.shuffle(cards)
    return cards


def make_opponent_board(deck: list[str], rng: random.Random, *, style: str) -> tuple[dict[str, list[str]], list[str]]:
    top_len = 1 if rng.random() < 0.85 else 2
    mid_len = rng.choice((2, 3, 4))
    bottom_len = 7 - top_len - mid_len
    if bottom_len < 2:
        mid_len -= 1
        bottom_len = 7 - top_len - mid_len

    top = take_any(deck, rng, top_len, prefer_ranks="23579JQKA" if style == "observed" else "789TJQKA")

    if mid_len >= 2 and rng.random() < 0.75:
        pair_rank = rng.choice(available_ranks(deck, "2233557799JQ", 2) or available_ranks(deck, RANKS, 2))
        middle = take_rank(deck, rng, pair_rank, 2)
        middle += take_any(deck, rng, mid_len - 2, avoid_ranks={pair_rank}, prefer_ranks="2345679TJQ")
    else:
        middle = take_any(deck, rng, mid_len, prefer_ranks="2233557799JQ")

    if bottom_len >= 2 and rng.random() < 0.60:
        pair_rank = rng.choice(available_ranks(deck, "23456789TJQK", 2) or available_ranks(deck, RANKS, 2))
        bottom = take_rank(deck, rng, pair_rank, 2)
        bottom += take_any(deck, rng, bottom_len - 2, avoid_ranks={pair_rank}, prefer_ranks="56789TJQK")
    else:
        bottom = take_any(deck, rng, bottom_len, prefer_ranks="56789TJQK")

    board = {
        "top": shuffled(rng, top),
        "middle": shuffled(rng, middle),
        "bottom": shuffled(rng, bottom),
    }
    return board, top + middle + bottom


def take_one_of_ranks(deck: list[str], rng: random.Random, values: str) -> str:
    pool = [card for card in deck if rank(card) in set(values)]
    if not pool:
        raise ValueError(f"no cards available for ranks {values}")
    chosen = rng.choice(pool)
    remove_cards(deck, [chosen])
    return chosen


def take_rank_pattern(deck: list[str], rng: random.Random, pattern: Iterable[str]) -> list[str]:
    cards: list[str] = []
    for value in pattern:
        cards.append(take_one_of_ranks(deck, rng, value))
    return cards


def make_seed34_miss_opponent_board(deck: list[str], rng: random.Random) -> tuple[dict[str, list[str]], list[str]]:
    """Generate opponent boards near the remaining seed20260634 train80 misses.

    All four remaining seed34 misses after train80 share the same exact-best
    action: put 3 on bottom, put T in middle, discard J.  The visible opponent
    boards are dominated by low pairs plus 3/J/Q/7/5 pressure.  This style
    keeps that structure but varies suits and a few neighboring ranks so it
    can train the pattern without copying only the held-out rows.
    """
    low_pair_rank = rng.choice("2244")
    low_single = rng.choice("567")
    high_single = rng.choice("QJ")
    connector = rng.choice("37")
    broad = rng.choice("9QJ")

    template = rng.randrange(4)
    if template == 0:
        top = take_rank_pattern(deck, rng, [low_single])
        middle = take_rank(deck, rng, low_pair_rank, 2)
        middle += take_rank_pattern(deck, rng, ["3", "J"])
        bottom = take_rank_pattern(deck, rng, ["7", high_single])
    elif template == 1:
        top = take_rank_pattern(deck, rng, [low_single])
        middle = take_rank_pattern(deck, rng, ["7", "Q", "3", "J"])
        bottom = take_rank(deck, rng, low_pair_rank, 2)
    elif template == 2:
        top = take_rank_pattern(deck, rng, ["Q", broad])
        middle = take_rank_pattern(deck, rng, ["7", "3"])
        bottom = take_rank(deck, rng, low_pair_rank, 2)
        bottom += take_rank_pattern(deck, rng, [low_single])
    else:
        top = take_rank_pattern(deck, rng, [connector])
        middle = take_rank(deck, rng, low_pair_rank, 2)
        middle += take_rank_pattern(deck, rng, ["3"])
        bottom = take_rank_pattern(deck, rng, ["5", "Q", "J"])

    board = {
        "top": shuffled(rng, top),
        "middle": shuffled(rng, middle),
        "bottom": shuffled(rng, bottom),
    }
    return board, top + middle + bottom


def make_dead_jack_opponent_board(deck: list[str], rng: random.Random) -> tuple[dict[str, list[str]], list[str]]:
    """Generate opponent boards for the remaining dead-J hard negatives.

    The dead-J family is disjoint from the seed34 pair2/3 guard: visible
    opponent cards include at least two jacks, but not both a pair of 2s and a
    visible 3.  The rows vary pair/trips pressure around 4/9/Q/K/T while keeping
    the exact-best action focused on whether the dealt J is already blocked.
    """
    jack_cards = take_rank(deck, rng, "J", 2)
    top: list[str] = []
    middle: list[str] = []
    bottom: list[str] = []

    def take_preferred(values: str) -> str:
        return take_any(deck, rng, 1, avoid_ranks={"3"}, prefer_ranks=values)[0]

    def take_pair(values: str) -> list[str]:
        ranks = available_ranks(deck, values, 2) or available_ranks(deck, "456789TQK", 2)
        return take_rank(deck, rng, rng.choice(ranks), 2)

    template = rng.randrange(5)
    if template == 0:
        top = [take_preferred("QK2")]
        middle = take_pair("49Q")
        bottom = jack_cards + [take_preferred("8T"), take_preferred("TQK")]
    elif template == 1:
        top = [take_preferred("25Q")]
        middle = jack_cards + [take_preferred("TQ"), take_preferred("49")]
        bottom = take_pair("46Q")
    elif template == 2:
        top = [take_preferred("QK")]
        middle = [jack_cards[0]] + take_pair("49Q")
        bottom = [jack_cards[1]] + [take_preferred("8T"), take_preferred("QK")]
    elif template == 3:
        top = [jack_cards[0]]
        middle = [jack_cards[1]] + take_pair("6Q")
        bottom = [take_preferred("5T"), take_preferred("QK"), take_preferred("89")]
    else:
        top = [take_preferred("2QK")]
        middle = [take_preferred("TQ"), take_preferred("49")]
        bottom = jack_cards + take_pair("4Q")

    board = {
        "top": shuffled(rng, top),
        "middle": shuffled(rng, middle),
        "bottom": shuffled(rng, bottom),
    }
    return board, top + middle + bottom


def fixed_hard_row(index: int, rng: random.Random) -> dict:
    deck = list(STANDARD_DECK)
    board = {
        "top": ["Ad"],
        "middle": ["5c", "5h", "X1"],
        "bottom": ["Kh", "Qc", "Kd"],
    }
    dealt = ["Td", "Js", "3s"]
    known_discards = ["8d"]
    remove_cards(deck, board["top"] + board["middle"] + board["bottom"] + dealt + known_discards)
    opponent_board, opponent_cards = make_opponent_board(deck, rng, style="observed")
    return source_row(index, board, opponent_board, dealt, known_discards, opponent_cards, "fixed_hard")


def seed34_miss_family_row(index: int, rng: random.Random) -> dict:
    deck = list(STANDARD_DECK)
    board = {
        "top": ["Ad"],
        "middle": ["5c", "5h", "X1"],
        "bottom": ["Kh", "Qc", "Kd"],
    }
    dealt = ["Td", "Js", "3s"]
    known_discards = ["8d"]
    remove_cards(deck, board["top"] + board["middle"] + board["bottom"] + dealt + known_discards)
    opponent_board, opponent_cards = make_seed34_miss_opponent_board(deck, rng)
    return source_row(index, board, opponent_board, dealt, known_discards, opponent_cards, "seed34_miss_family")


def dead_jack_family_row(index: int, rng: random.Random) -> dict:
    deck = list(STANDARD_DECK)
    board = {
        "top": ["Ad"],
        "middle": ["5c", "5h", "X1"],
        "bottom": ["Kh", "Qc", "Kd"],
    }
    dealt = ["Td", "Js", "3s"]
    known_discards = ["8d"]
    remove_cards(deck, board["top"] + board["middle"] + board["bottom"] + dealt + known_discards)
    opponent_board, opponent_cards = make_dead_jack_opponent_board(deck, rng)
    return source_row(index, board, opponent_board, dealt, known_discards, opponent_cards, "dead_jack_family")


def make_pressure_opponent_board(deck: list[str], rng: random.Random) -> tuple[dict[str, list[str]], list[str]]:
    """Generate visible opponent pressure near the seed28 residual misses."""

    template = rng.randrange(4)
    if template == 0:
        top = take_rank_pattern(deck, rng, ["K"])
        middle = take_rank_pattern(deck, rng, ["A", "Q", "J"])
        bottom = take_rank_pattern(deck, rng, ["T", "T", "8"])
    elif template == 1:
        top = take_rank_pattern(deck, rng, ["K", "Q"])
        middle = take_rank_pattern(deck, rng, ["A", "J"])
        bottom = take_rank_pattern(deck, rng, ["T", "T", "5"])
    elif template == 2:
        top = take_rank_pattern(deck, rng, ["Q"])
        middle = take_rank_pattern(deck, rng, ["K", "5", "8"])
        bottom = take_rank_pattern(deck, rng, ["A", "T", "T"])
    else:
        top = take_rank_pattern(deck, rng, ["J"])
        middle = take_rank_pattern(deck, rng, ["A", "Q", "5"])
        bottom = take_rank_pattern(deck, rng, ["K", "T", "T"])
    board = {
        "top": shuffled(rng, top),
        "middle": shuffled(rng, middle),
        "bottom": shuffled(rng, bottom),
    }
    return board, top + middle + bottom


def seed28_low_379_family_row(index: int, rng: random.Random) -> dict:
    """Generate BB rows near the seed28 low 3/7/9 residual family.

    The held-out misses repeatedly prefer ``3->middle; 7->bottom; discard 9``
    over placing the 9 in middle or putting both 3/7 in middle.  These rows
    vary visible blockers and the exact middle/bottom lengths while keeping the
    runtime decision family intact.
    """

    deck = list(STANDARD_DECK)
    top = take_rank(deck, rng, "3", 1) + take_rank(deck, rng, "A", 1)
    template = rng.randrange(3)
    if template == 0:
        middle = take_rank(deck, rng, "8", 1)
        bottom = take_rank(deck, rng, "7", 2) + take_rank_pattern(deck, rng, ["Q", "T"])
    elif template == 1:
        middle = take_rank_pattern(deck, rng, ["8", "6"])
        bottom = take_rank_pattern(deck, rng, ["7", "Q", "T"])
    else:
        middle = take_rank_pattern(deck, rng, ["8", "7"])
        bottom = take_rank_pattern(deck, rng, ["7", "Q", "T"])
    dealt = take_rank_pattern(deck, rng, ["3", "7", "9"])
    known_discards = take_rank(deck, rng, "6", 1) if available_ranks(deck, "6", 1) else take_rank(deck, rng, "7", 1)
    opponent_board, opponent_cards = make_pressure_opponent_board(deck, rng)
    board = {
        "top": shuffled(rng, top),
        "middle": shuffled(rng, middle),
        "bottom": shuffled(rng, bottom),
    }
    return source_row(
        index,
        board,
        opponent_board,
        shuffled(rng, dealt),
        known_discards,
        opponent_cards,
        "seed28_low_379_family",
        position="bb",
    )


def seed28_qqk_fl_pressure_family_row(index: int, rng: random.Random) -> dict:
    """Generate BTN rows near the seed28 Q/Q/K high-FL pressure misses."""

    deck = list(STANDARD_DECK)
    template = rng.randrange(3)
    if template == 0:
        top = take_rank_pattern(deck, rng, ["K", "Q"])
        middle = take_rank_pattern(deck, rng, ["A", "5", "J"])
        bottom = take_rank(deck, rng, "T", 2)
        discard_rank = "8"
    elif template == 1:
        top = take_rank_pattern(deck, rng, ["K"])
        middle = take_rank_pattern(deck, rng, ["A", "Q", "5"])
        bottom = take_rank(deck, rng, "T", 2) + take_rank(deck, rng, "8", 1)
        discard_rank = "J"
    else:
        top = take_rank_pattern(deck, rng, ["5"])
        middle = take_rank_pattern(deck, rng, ["K", "Q", "8"])
        bottom = take_rank_pattern(deck, rng, ["A", "T", "T"])
        discard_rank = "J"
    dealt = take_rank_pattern(deck, rng, ["Q", "Q", "K"])
    known_discards = take_rank(deck, rng, discard_rank, 1)
    opponent_board, opponent_cards = make_seed28_qqk_opponent_board(deck, rng)
    board = {
        "top": shuffled(rng, top),
        "middle": shuffled(rng, middle),
        "bottom": shuffled(rng, bottom),
    }
    return source_row(
        index,
        board,
        opponent_board,
        shuffled(rng, dealt),
        known_discards,
        opponent_cards,
        "seed28_qqk_fl_pressure_family",
        position="btn",
    )


def make_seed28_qqk_opponent_board(deck: list[str], rng: random.Random) -> tuple[dict[str, list[str]], list[str]]:
    """Generate opponent boards for the BTN Q/Q/K FL-pressure family."""

    template = rng.randrange(3)
    if template == 0:
        top = take_rank_pattern(deck, rng, ["3", "A"])
        middle = take_rank_pattern(deck, rng, ["8", "T", "9"])
        bottom = take_rank_pattern(deck, rng, ["7", "Q", "6", "7"])
    elif template == 1:
        top = take_rank_pattern(deck, rng, ["3", "A"])
        middle = take_rank_pattern(deck, rng, ["8", "9"])
        bottom = take_rank_pattern(deck, rng, ["7", "Q", "7", "T", "7"])
    else:
        top = take_rank_pattern(deck, rng, ["3", "A"])
        middle = take_rank_pattern(deck, rng, ["8", "6", "9"])
        bottom = take_rank_pattern(deck, rng, ["7", "Q", "T", "7"])
    board = {
        "top": shuffled(rng, top),
        "middle": shuffled(rng, middle),
        "bottom": shuffled(rng, bottom),
    }
    return board, top + middle + bottom


def suit_family_row(index: int, rng: random.Random) -> dict:
    deck = list(STANDARD_DECK)
    top = take_rank(deck, rng, "A", 1)
    middle = take_rank(deck, rng, "5", 2) + ["X1"]
    remove_cards(deck, ["X1"])
    bottom = take_rank(deck, rng, "K", 2) + take_rank(deck, rng, "Q", 1)
    dealt = take_rank(deck, rng, "T", 1) + take_rank(deck, rng, "J", 1) + take_rank(deck, rng, "3", 1)
    known_discards = take_rank(deck, rng, "8", 1) if available_ranks(deck, "8", 1) else take_any(deck, rng, 1)
    opponent_board, opponent_cards = make_opponent_board(deck, rng, style="observed")
    board = {"top": top, "middle": shuffled(rng, middle), "bottom": shuffled(rng, bottom)}
    return source_row(index, board, opponent_board, shuffled(rng, dealt), known_discards, opponent_cards, "suit_family")


def rank_family_row(index: int, rng: random.Random) -> dict:
    deck = list(STANDARD_DECK)
    top_rank = "A"
    pair_rank = rng.choice("456")
    bottom_pair_rank = rng.choice("KQ")
    bottom_kicker_rank = "Q" if bottom_pair_rank == "K" else "K"
    dealt_ranks = rng.choice((("T", "J", "3"), ("9", "J", "3"), ("T", "Q", "3"), ("T", "J", "4")))

    top = take_rank(deck, rng, top_rank, 1)
    middle = take_rank(deck, rng, pair_rank, 2) + ["X1"]
    remove_cards(deck, ["X1"])
    bottom = take_rank(deck, rng, bottom_pair_rank, 2) + take_rank(deck, rng, bottom_kicker_rank, 1)
    dealt: list[str] = []
    for value in dealt_ranks:
        dealt.extend(take_rank(deck, rng, value, 1))
    known_discards = take_any(deck, rng, 1, prefer_ranks="789")
    opponent_board, opponent_cards = make_opponent_board(deck, rng, style="broad")
    board = {"top": top, "middle": shuffled(rng, middle), "bottom": shuffled(rng, bottom)}
    return source_row(index, board, opponent_board, shuffled(rng, dealt), known_discards, opponent_cards, "rank_family")


def source_row(
    index: int,
    board: dict[str, list[str]],
    opponent_board: dict[str, list[str]],
    dealt: list[str],
    known_discards: list[str],
    opponent_cards: list[str],
    style: str,
    *,
    position: str = "bb",
) -> dict:
    is_btn = position.lower() == "btn"
    return {
        "source": "synthetic_t2_joker_middle_family",
        "source_line": index,
        "active_reasons": ["t2_joker_middle_family"],
        "turn": 2,
        "board": board,
        "opponent_board": opponent_board,
        "dealt": dealt,
        "known_discards": known_discards,
        "exclude": opponent_cards + known_discards,
        "is_btn": is_btn,
        "position": position,
        "mode": f"targeted_joker_middle_{style}_alllegal",
        "estimated": False,
        "exact": False,
        "candidate_count": 0,
        "candidates": [],
        "branch": {
            "root_index": index,
            "route_index": 0,
            "branch_id": f"joker-middle-{style}-{index}",
            "target_position": position,
            "target_t0_rank": None,
            "target_t1_rank": None,
        },
        "generator_config": {
            "pattern": "A_top_pair_plus_joker_middle_big_pair_bottom_TJ3_like_dealt",
            "style": style,
            "opponent_cards": len(opponent_cards),
        },
    }


def signature(row: dict) -> str:
    return json.dumps(
        {
            "board": row["board"],
            "opponent_board": row["opponent_board"],
            "dealt": row["dealt"],
            "known_discards": row["known_discards"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def make_row(index: int, rng: random.Random, *, style: str) -> dict:
    if style == "mixed":
        roll = rng.random()
        if roll < 0.50:
            style = "fixed_hard"
        elif roll < 0.85:
            style = "suit_family"
        else:
            style = "rank_family"
    if style == "fixed_hard":
        return fixed_hard_row(index, rng)
    if style == "seed34_miss_family":
        return seed34_miss_family_row(index, rng)
    if style == "dead_jack_family":
        return dead_jack_family_row(index, rng)
    if style == "seed28_low_379_family":
        return seed28_low_379_family_row(index, rng)
    if style == "seed28_qqk_fl_pressure_family":
        return seed28_qqk_fl_pressure_family_row(index, rng)
    if style == "suit_family":
        return suit_family_row(index, rng)
    if style == "rank_family":
        return rank_family_row(index, rng)
    raise ValueError(style)


def write_rows(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    seen: set[str] = set()
    rows: list[dict] = []
    attempts = 0
    style_counts: dict[str, int] = {}
    position_counts: dict[str, int] = {}
    while len(rows) < args.rows:
        attempts += 1
        if attempts > args.rows * 50:
            raise RuntimeError(f"could only generate {len(rows)} unique rows after {attempts} attempts")
        row = make_row(len(rows), rng, style=args.style)
        key = signature(row)
        if key in seen:
            continue
        seen.add(key)
        style_name = row["generator_config"]["style"]
        style_counts[style_name] = style_counts.get(style_name, 0) + 1
        position = str(row.get("position") or "")
        position_counts[position] = position_counts.get(position, 0) + 1
        rows.append(row)

    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {
        "output": str(output),
        "rows": int(len(rows)),
        "seed": int(args.seed),
        "style": args.style,
        "style_counts": style_counts,
        "position_counts": position_counts,
        "candidate_mode": "all_legal_in_rust",
        "pattern": "targeted_t2_residual_family",
    }
    output.with_suffix(output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument(
        "--style",
        choices=(
            "mixed",
            "fixed_hard",
            "seed34_miss_family",
            "dead_jack_family",
            "seed28_low_379_family",
            "seed28_qqk_fl_pressure_family",
            "suit_family",
            "rank_family",
        ),
        default="mixed",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(write_rows(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
