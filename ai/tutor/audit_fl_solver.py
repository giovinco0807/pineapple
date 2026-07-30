"""Audit the production Fantasyland path against the canonical evaluator.

Checks structural integrity of fl_solver's output (cards conserved, row sizes,
no duplicates, determinism) and the correctness of what it reports (foul
verdict, royalty, Fantasyland stay) against
`ai/engine/game_engine.evaluate_board_with_joker_constraint`, which is the
canonical semantics every scoring path is supposed to consume.

Audit only: nothing here modifies fl_solver or the production bridge.

Usage:
    python -m ai.tutor.audit_fl_solver --hands 400 --version 2
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from ai.engine.game_engine import (
    check_fl_entry,
    evaluate_board_with_joker_constraint,
    evaluate_hand,
    get_bottom_royalty,
    get_middle_royalty,
    get_top_royalty,
)

RANKS = {"T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
SUITS = {"s": 0, "h": 1, "d": 2, "c": 3}
ALL_CARDS = [rank + suit for suit in "shdc" for rank in "23456789TJQKA"] + ["X1", "X2"]


def solver_path() -> Path:
    suffix = ".exe" if sys.platform.startswith("win") else ""
    return Path("ai/rust_solver/target/release") / f"fl_solver{suffix}"


def to_solver(card: str) -> dict:
    if card in ("X1", "X2"):
        return {"rank": 0, "suit": 4}
    rank = RANKS.get(card[0], int(card[0]) if card[0].isdigit() else 0)
    return {"rank": rank, "suit": SUITS[card[1]]}


def from_solver(card: dict, jokers_left: list[str]) -> str:
    if card.get("rank", 0) == 0:
        return jokers_left.pop(0) if jokers_left else "X?"
    return "23456789TJQKA"[card["rank"] - 2] + "shdc"[card["suit"]]


def run_solver(hands: list[list[str]], version: int) -> list[dict]:
    payload = "\n".join(
        json.dumps({"cards": [to_solver(card) for card in hand], "version": version})
        for hand in hands
    )
    result = subprocess.run(
        [str(solver_path())], input=payload + "\n", capture_output=True, text=True
    )
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


def canonical(rows) -> dict:
    evaluation = evaluate_board_with_joker_constraint(
        list(rows[0]), list(rows[1]), list(rows[2])
    )
    busted = bool(evaluation["busted"])
    royalty = (
        0
        if busted
        else get_top_royalty(evaluation["top"])
        + get_middle_royalty(evaluation["middle"])
        + get_bottom_royalty(evaluation["bottom"])
    )
    qualified, count = (False, 0) if busted else check_fl_entry(evaluation["top"])
    top_value = evaluate_hand(evaluation["top"], 3)
    bottom_value = evaluate_hand(evaluation["bottom"], 5)
    return {
        "busted": busted,
        "royalty": royalty,
        "fl_entry": qualified,
        "fl_cards": count,
        # Stay: trips on top or quads+ on bottom, read off constrained rows.
        "stay": (top_value // 15**5) == 3 or (bottom_value // 15**5) >= 7,
    }


def raw_foul(rows) -> bool:
    """Does the board foul if every joker takes its strongest value?

    This is what the solver's own claimed royalty implies, so a board that
    fouls here while the solver reports no bust is claiming a royalty it
    cannot legally collect.
    """
    values = [
        evaluate_hand(list(rows[0]), 3),
        evaluate_hand(list(rows[1]), 5),
        evaluate_hand(list(rows[2]), 5),
    ]
    return values[0] > values[1] or values[1] > values[2]


def audit(hands: list[list[str]], version: int) -> dict:
    started = time.time()
    responses = run_solver(hands, version)
    elapsed = time.time() - started
    findings = Counter()
    examples: dict[str, list] = {}

    def note(key: str, detail):
        findings[key] += 1
        examples.setdefault(key, [])
        if len(examples[key]) < 3:
            examples[key].append(detail)

    royalty_claimed = royalty_canonical = 0
    scored = 0
    for hand, response in zip(hands, responses):
        if not response.get("success"):
            note("solver_failed", hand)
            continue
        placement = response["placement"]
        jokers = [card for card in hand if card in ("X1", "X2")]
        rows = [
            [from_solver(card, jokers) for card in placement[key]]
            for key in ("top", "middle", "bottom")
        ]
        flat = [card for row in rows for card in row]

        if [len(row) for row in rows] != [3, 5, 5]:
            note("bad_row_sizes", (hand, [len(row) for row in rows]))
            continue
        if len(set(flat)) != 13:
            note("duplicate_cards_placed", (hand, rows))
        if not set(flat).issubset(set(hand)):
            note("card_not_in_input", (hand, sorted(set(flat) - set(hand))))
        expected_discards = sorted(Counter(hand) - Counter(flat))
        reported = placement.get("discards") or placement.get("discarded") or []
        if reported:
            got = sorted(from_solver(c, list(jokers)) for c in reported) if reported and isinstance(reported[0], dict) else sorted(reported)
            if len(got) != len(hand) - 13:
                note("discard_count_wrong", (hand, got))

        canon = canonical(rows)
        scored += 1
        royalty_claimed += placement["total_royalty"]
        royalty_canonical += canon["royalty"]
        if canon["busted"] != placement["is_bust"]:
            note("foul_verdict_differs", (rows, canon["busted"], placement["is_bust"]))
        if canon["royalty"] != placement["total_royalty"]:
            note("royalty_differs", (rows, canon["royalty"], placement["total_royalty"]))
        if raw_foul(rows) and not placement["is_bust"]:
            note("claims_no_foul_but_raw_board_fouls", (rows, placement["total_royalty"]))
        if canon["stay"] != placement.get("can_stay"):
            note("stay_differs", (rows, canon["stay"], placement.get("can_stay")))

    return {
        "version": version,
        "hands": len(hands),
        "scored": scored,
        "ms_per_hand": 1000 * elapsed / max(len(hands), 1),
        "mean_royalty_claimed": royalty_claimed / max(scored, 1),
        "mean_royalty_canonical": royalty_canonical / max(scored, 1),
        "findings": dict(findings),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=int, default=400)
    parser.add_argument("--cards", type=int, default=14)
    parser.add_argument("--version", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    hands = []
    for _ in range(args.hands):
        deck = ALL_CARDS[:]
        rng.shuffle(deck)
        hands.append(deck[: args.cards])

    report = audit(hands, args.version)
    report["cards_per_hand"] = args.cards
    report["seed"] = args.seed

    # Determinism: the same hands twice must give byte-identical placements.
    first = run_solver(hands[:50], args.version)
    second = run_solver(hands[:50], args.version)
    report["deterministic"] = json.dumps(first, sort_keys=True) == json.dumps(
        second, sort_keys=True
    )

    print(json.dumps({k: v for k, v in report.items() if k != "examples"},
                     ensure_ascii=False, indent=2, sort_keys=True))
    for key, rows in report["examples"].items():
        print(f"\n### {key}")
        for row in rows:
            print("   ", row)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
