"""Generate targeted T2 source rows for top-AA plus A/J draw spots."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable


RANKS = "23456789TJQKA"
SUITS = "cdhs"
STANDARD_DECK = [rank + suit for rank in RANKS for suit in SUITS]


def rank(card: str) -> str:
    return card[0]


def take_rank(deck: list[str], rng: random.Random, value: str, count: int) -> list[str]:
    cards = [card for card in deck if rank(card) == value]
    if len(cards) < count:
        raise ValueError(f"not enough {value} cards")
    chosen = rng.sample(cards, count)
    for card in chosen:
        deck.remove(card)
    return chosen


def take_any(deck: list[str], rng: random.Random, count: int, *, avoid_ranks: set[str] | None = None) -> list[str]:
    avoid_ranks = avoid_ranks or set()
    pool = [card for card in deck if rank(card) not in avoid_ranks]
    if len(pool) < count:
        pool = list(deck)
    chosen = rng.sample(pool, count)
    for card in chosen:
        deck.remove(card)
    return chosen


def available_ranks(deck: list[str], values: Iterable[str], count: int) -> list[str]:
    return [value for value in values if sum(1 for card in deck if rank(card) == value) >= count]


def make_row(index: int, rng: random.Random, *, style: str) -> dict:
    deck = list(STANDARD_DECK)
    rng.shuffle(deck)

    top = take_rank(deck, rng, "A", 2)
    top_full = style == "ae_like" and rng.random() < 0.45
    if top_full:
        top += take_rank(deck, rng, rng.choice(available_ranks(deck, "QK", 1) or ["Q"]), 1)

    middle_fillers = "569Q" if style == "ae_like" else RANKS
    middle = take_rank(deck, rng, "J", 2) + take_rank(
        deck,
        rng,
        rng.choice(available_ranks(deck, middle_fillers, 1) or available_ranks(deck, RANKS, 1)),
        1,
    )
    bottom = take_any(deck, rng, 1 if top_full else 2, avoid_ranks={"A", "J"})

    dealt = take_rank(deck, rng, "A", 1) + take_rank(deck, rng, "J", 1) + take_any(
        deck,
        rng,
        1,
        avoid_ranks={"A", "J"},
    )
    rng.shuffle(dealt)

    opp_top_pattern = "KKA" if style == "ae_like" else rng.choice(("KKA", "AKx", "KKx", "AQx"))
    if opp_top_pattern == "KKA":
        opp_top = take_rank(deck, rng, "K", 2) + take_rank(deck, rng, "A", 1)
    elif opp_top_pattern == "AKx":
        opp_top = take_rank(deck, rng, "A", 1) + take_rank(deck, rng, "K", 1) + take_any(deck, rng, 1)
    elif opp_top_pattern == "KKx":
        opp_top = take_rank(deck, rng, "K", 2) + take_any(deck, rng, 1)
    else:
        opp_top = take_rank(deck, rng, "A", 1) + take_rank(deck, rng, "Q", 1) + take_any(deck, rng, 1)

    pair_pool = "5678" if style == "ae_like" else "23456789TQ"
    pair_rank = rng.choice(available_ranks(deck, pair_pool, 2) or available_ranks(deck, "23456789TQ", 2))
    opp_middle = take_rank(deck, rng, pair_rank, 2) + take_any(deck, rng, rng.choice((0, 1)))
    if style == "ae_like":
        bottom_pair_rank = rng.choice(available_ranks(deck, "6789", 2) or available_ranks(deck, "23456789TQ", 2))
        opp_bottom = take_rank(deck, rng, bottom_pair_rank, 2) + take_any(
            deck,
            rng,
            9 - len(opp_top) - len(opp_middle) - 2,
        )
    else:
        opp_bottom = take_any(deck, rng, 9 - len(opp_top) - len(opp_middle))
    known_discards = take_any(deck, rng, 1)

    opponent_cards = opp_top + opp_middle + opp_bottom
    return {
        "source": f"synthetic_t2_top_aa_aj_{style}",
        "source_line": index,
        "turn": 2,
        "board": {"top": top, "middle": middle, "bottom": bottom},
        "opponent_board": {"top": opp_top, "middle": opp_middle, "bottom": opp_bottom},
        "dealt": dealt,
        "known_discards": known_discards,
        "exclude": opponent_cards + known_discards,
        "is_btn": True,
        "position": "btn",
        "mode": f"targeted_top_aa_aj_alllegal_{style}",
        "estimated": False,
        "exact": False,
        "candidate_count": 0,
        "candidates": [],
        "branch": {
            "root_index": index,
            "route_index": 0,
            "branch_id": f"synthetic-{index}",
            "target_position": "btn",
            "target_t0_rank": None,
            "target_t1_rank": None,
        },
        "generator_config": {
            "pattern": "top_AA_middle_JJ_dealt_AJx",
            "style": style,
            "opponent_cards": len(opponent_cards),
        },
    }


def write_rows(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    with output.open("w", encoding="utf-8") as handle:
        for index in range(args.rows):
            handle.write(json.dumps(make_row(index, rng, style=args.style), ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {
        "output": str(output),
        "rows": int(args.rows),
        "seed": int(args.seed),
        "pattern": "top_AA_middle_JJ_dealt_AJx",
        "style": args.style,
        "position": "btn",
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
    parser.add_argument("--style", choices=("broad", "ae_like"), default="broad")
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(write_rows(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
