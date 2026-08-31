"""Generate targeted T2 source rows for weak-top middle-overbuild misses.

These rows are synthetic T2 decisions designed to teach the reranker not to
over-strengthen middle when top is still weak.  They intentionally vary cards,
opponent board, and position instead of replaying holdout rows.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable


RANKS = "23456789TJQKA"
SUITS = "cdhs"
STANDARD_DECK = [rank + suit for rank in RANKS for suit in SUITS]
LOW_TOP_RANKS = "234567"
MID_RANKS = "456789T"
HIGH_DEALT_RANKS = "QKA"


def rank(card: str) -> str:
    return card[0]


def take_rank(deck: list[str], rng: random.Random, value: str, count: int) -> list[str]:
    pool = [card for card in deck if rank(card) == value]
    if len(pool) < count:
        raise ValueError(f"not enough {value} cards")
    chosen = rng.sample(pool, count)
    for card in chosen:
        deck.remove(card)
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
    for card in chosen:
        deck.remove(card)
    return chosen


def available_ranks(deck: list[str], values: Iterable[str], count: int) -> list[str]:
    return [value for value in values if sum(1 for card in deck if rank(card) == value) >= count]


def shuffled(rng: random.Random, cards: list[str]) -> list[str]:
    cards = list(cards)
    rng.shuffle(cards)
    return cards


def opponent_board(deck: list[str], rng: random.Random) -> tuple[dict[str, list[str]], list[str]]:
    top_len = rng.choice((1, 2))
    mid_len = rng.choice((2, 3))
    bottom_len = 7 - top_len - mid_len

    top = take_any(deck, rng, top_len, prefer_ranks="789TJQKA")
    pair_ranks = available_ranks(deck, "23456789TQK", 2)
    if pair_ranks and mid_len >= 2 and rng.random() < 0.65:
        pair_rank = rng.choice(pair_ranks)
        middle = take_rank(deck, rng, pair_rank, 2)
        middle += take_any(deck, rng, mid_len - 2, avoid_ranks={pair_rank})
    else:
        middle = take_any(deck, rng, mid_len)

    if bottom_len >= 2 and rng.random() < 0.55:
        pair_ranks = available_ranks(deck, "23456789TJQ", 2)
        if pair_ranks:
            pair_rank = rng.choice(pair_ranks)
            bottom = take_rank(deck, rng, pair_rank, 2)
            bottom += take_any(deck, rng, bottom_len - 2, avoid_ranks={pair_rank})
        else:
            bottom = take_any(deck, rng, bottom_len)
    else:
        bottom = take_any(deck, rng, bottom_len)

    board = {"top": shuffled(rng, top), "middle": shuffled(rng, middle), "bottom": shuffled(rng, bottom)}
    return board, top + middle + bottom


def make_top_discard_pair(index: int, rng: random.Random, *, position: str) -> dict:
    deck = list(STANDARD_DECK)
    rng.shuffle(deck)

    top_rank = rng.choice(available_ranks(deck, LOW_TOP_RANKS, 1))
    top = take_rank(deck, rng, top_rank, 1)
    pair_rank = rng.choice(available_ranks(deck, "234567", 2))
    middle_high = rng.choice(available_ranks(deck, "TJQ", 1))
    middle = take_rank(deck, rng, pair_rank, 2) + take_rank(deck, rng, middle_high, 1)
    bottom = take_any(deck, rng, 3, avoid_ranks={pair_rank, middle_high}, prefer_ranks="3456789T")

    dealt = take_any(deck, rng, 1, prefer_ranks="AK")
    dealt += take_rank(deck, rng, middle_high, 1)
    dealt += take_any(deck, rng, 1, avoid_ranks={middle_high}, prefer_ranks="3456789")
    dealt = shuffled(rng, dealt)

    opp, opponent_cards = opponent_board(deck, rng)
    known_discards = take_any(deck, rng, rng.choice((0, 1)))
    is_btn = position == "btn"
    return {
        "source": "synthetic_t2_weak_top_guard_top_discard_pair",
        "source_line": index,
        "turn": 2,
        "board": {"top": top, "middle": shuffled(rng, middle), "bottom": shuffled(rng, bottom)},
        "opponent_board": opp,
        "dealt": dealt,
        "known_discards": known_discards,
        "exclude": opponent_cards + known_discards,
        "is_btn": is_btn,
        "position": position,
        "mode": "targeted_weak_top_guard_alllegal_top_discard_pair",
        "estimated": False,
        "exact": False,
        "candidate_count": 0,
        "candidates": [],
        "branch": {
            "root_index": index,
            "route_index": 0,
            "branch_id": f"weak-top-guard-a-{index}",
            "target_position": position,
            "target_t0_rank": None,
            "target_t1_rank": None,
        },
        "generator_config": {
            "pattern": "weak_top_pair_plus_high_middle_dealt_A_or_K_high_match_side",
            "style": "top_discard_pair",
            "opponent_cards": len(opponent_cards),
        },
    }


def make_bottom_fill_over_middle(index: int, rng: random.Random, *, position: str) -> dict:
    deck = list(STANDARD_DECK)
    rng.shuffle(deck)

    top_rank = rng.choice(available_ranks(deck, LOW_TOP_RANKS, 1))
    top = take_rank(deck, rng, top_rank, 1)

    anchor = rng.choice(available_ranks(deck, "A", 1))
    match_ranks = available_ranks(deck, "56789T", 2)
    rng.shuffle(match_ranks)
    first_match = match_ranks[0]
    second_match = match_ranks[1] if len(match_ranks) > 1 else rng.choice(available_ranks(deck, MID_RANKS, 1))
    filler_rank = rng.choice([r for r in available_ranks(deck, "23456789T", 1) if r not in {first_match, second_match}] or available_ranks(deck, "23456789T", 1))
    middle = (
        take_rank(deck, rng, anchor, 1)
        + take_rank(deck, rng, first_match, 1)
        + take_rank(deck, rng, second_match, 1)
        + take_rank(deck, rng, filler_rank, 1)
    )

    bottom = take_any(deck, rng, 2, avoid_ranks={first_match, second_match, filler_rank}, prefer_ranks="789TJ")
    high = rng.choice(available_ranks(deck, HIGH_DEALT_RANKS, 1))
    dealt = take_rank(deck, rng, high, 1)
    dealt += take_rank(deck, rng, first_match, 1)
    if rng.random() < 0.55 and sum(1 for card in deck if rank(card) == second_match) >= 1:
        dealt += take_rank(deck, rng, second_match, 1)
    else:
        dealt += take_any(deck, rng, 1, avoid_ranks={high}, prefer_ranks="3456789")
    dealt = shuffled(rng, dealt)

    opp, opponent_cards = opponent_board(deck, rng)
    known_discards = take_any(deck, rng, rng.choice((0, 1)))
    is_btn = position == "btn"
    return {
        "source": "synthetic_t2_weak_top_guard_bottom_fill_over_middle",
        "source_line": index,
        "turn": 2,
        "board": {"top": top, "middle": shuffled(rng, middle), "bottom": shuffled(rng, bottom)},
        "opponent_board": opp,
        "dealt": dealt,
        "known_discards": known_discards,
        "exclude": opponent_cards + known_discards,
        "is_btn": is_btn,
        "position": position,
        "mode": "targeted_weak_top_guard_alllegal_bottom_fill_over_middle",
        "estimated": False,
        "exact": False,
        "candidate_count": 0,
        "candidates": [],
        "branch": {
            "root_index": index,
            "route_index": 0,
            "branch_id": f"weak-top-guard-b-{index}",
            "target_position": position,
            "target_t0_rank": None,
            "target_t1_rank": None,
        },
        "generator_config": {
            "pattern": "weak_top_A_middle_four_cards_dealt_high_and_middle_matches",
            "style": "bottom_fill_over_middle",
            "opponent_cards": len(opponent_cards),
        },
    }


def make_row(index: int, rng: random.Random, *, style: str, position_mode: str) -> dict:
    position = rng.choice(("btn", "bb")) if position_mode == "mixed" else position_mode
    if style == "mixed":
        style = rng.choice(("top_discard_pair", "bottom_fill_over_middle"))
    if style == "top_discard_pair":
        return make_top_discard_pair(index, rng, position=position)
    if style == "bottom_fill_over_middle":
        return make_bottom_fill_over_middle(index, rng, position=position)
    raise ValueError(style)


def write_rows(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    style_counts: dict[str, int] = {}
    position_counts: dict[str, int] = {}
    with output.open("w", encoding="utf-8") as handle:
        for index in range(args.rows):
            row = make_row(index, rng, style=args.style, position_mode=args.position)
            style_name = row["generator_config"]["style"]
            style_counts[style_name] = style_counts.get(style_name, 0) + 1
            position_counts[row["position"]] = position_counts.get(row["position"], 0) + 1
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {
        "output": str(output),
        "rows": int(args.rows),
        "seed": int(args.seed),
        "style": args.style,
        "position": args.position,
        "style_counts": style_counts,
        "position_counts": position_counts,
        "patterns": [
            "weak_top_pair_plus_high_middle_dealt_A_or_K_high_match_side",
            "weak_top_A_middle_four_cards_dealt_high_and_middle_matches",
        ],
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
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--style", choices=("mixed", "top_discard_pair", "bottom_fill_over_middle"), default="mixed")
    parser.add_argument("--position", choices=("mixed", "btn", "bb"), default="mixed")
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(write_rows(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
