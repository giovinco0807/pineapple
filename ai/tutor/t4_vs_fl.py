"""T4 decision for the normal player facing a Fantasyland opponent.

The FL opponent set its board at deal time, face down, and makes no further
decisions -- so the hero's state is its own hand plus deck accounting, and the
value of a placement is an expectation over the hidden FL board:

    value(action) = E_h [ score(hero_final, FL_board(h)) ],  h ~ uniform over
    n-card hands disjoint from the hero's seen cards.

Boards come from the global FL library (build_fl_board_library): entries whose
dealt-hand mask is disjoint from the hero's seen set are exactly that uniform
conditional, so scoring is a filter plus a vectorized comparison, with no
per-root solving.

Scoring, per the frozen rules contract:
- lines/scoop/royalty as usual, fouls at -6 minus the opponent's royalty;
- the hero's own FL entry adds FL_EV(entry count);
- an FL stay costs the hero FL_EV(opponent's CURRENT count): in the joker
  ruleset a stay redeals the original entry count (commit 373ee02), so the
  17-card chain compounds at 17.
The FL EV table is the v0 config; the M-C fixed point replaces it.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

import ai.tutor.exact_late as exact_late
from ai.engine.action_space import get_turn_actions
from ai.engine.encoding import ALL_CARDS, Board
from ai.engine.game_engine import (
    check_fl_entry,
    evaluate_board_with_joker_constraint,
    evaluate_hand,
    get_bottom_royalty,
    get_middle_royalty,
    get_top_royalty,
)
from ai.tutor.fl_ev_table import FL_EV
from ai.tutor.t4_first_features import hero_block

CARD_INDEX = {card: index for index, card in enumerate(ALL_CARDS)}
TWO_OPEN_SHAPES = ((3, 5, 3), (3, 4, 4), (3, 3, 5), (2, 5, 4), (2, 4, 5), (1, 5, 5))
FEATURE_SCHEMA = "ofc_t4_vs_fl_features/v1"
HERO_SIZE = 42
FL_CONTEXT_SIZE = 12
FEATURE_SIZE = HERO_SIZE + FL_CONTEXT_SIZE  # 54


class FlLibrary:
    """Solved FL boards with dealt-hand masks for the disjointness filter."""

    def __init__(self, shard_dir: Path) -> None:
        masks, values, royalty, stay, busted = [], [], [], [], []
        for shard in sorted(shard_dir.glob("shard_*.jsonl")):
            with shard.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    masks.append(row["mask"])
                    values.append(row["values"])
                    royalty.append(row["royalty"])
                    stay.append(row["stay"])
                    busted.append(row["busted"])
        self.masks = np.asarray(masks, dtype=np.uint64)
        self.values = np.asarray(values, dtype=np.int64)
        self.royalty = np.asarray(royalty, dtype=np.float64)
        self.stay = np.asarray(stay, dtype=bool)
        self.busted = np.asarray(busted, dtype=bool)

    def __len__(self) -> int:
        return len(self.masks)

    def matching(self, seen_mask: int) -> np.ndarray:
        """Indices of boards whose dealt hand avoids every seen card."""
        return np.nonzero((self.masks & np.uint64(seen_mask)) == 0)[0]


def seen_mask(cards) -> int:
    mask = 0
    for card in cards:
        mask |= 1 << CARD_INDEX[card]
    return mask


def sample_root(seed: int, opp_count: int = 14) -> dict:
    """One physical T4 root for the normal seat, opponent in FL."""
    rng = random.Random(seed)
    deck = ALL_CARDS[:]
    rng.shuffle(deck)

    def take(count: int) -> list[str]:
        cards = deck[:count]
        del deck[:count]
        return cards

    shape = rng.choice(TWO_OPEN_SHAPES)
    board = [take(shape[0]), take(shape[1]), take(shape[2])]
    dead = take(3)
    draw = take(3)
    return {
        "seed": seed,
        "board": board,
        "dead": dead,
        "draw": draw,
        "opp_count": opp_count,
    }


def hero_terminal(rows) -> dict:
    evaluation = evaluate_board_with_joker_constraint(
        list(rows[0]), list(rows[1]), list(rows[2])
    )
    busted = bool(evaluation["busted"])
    final = (evaluation["top"], evaluation["middle"], evaluation["bottom"])
    values = np.asarray(
        [
            evaluate_hand(list(final[0]), 3),
            evaluate_hand(list(final[1]), 5),
            evaluate_hand(list(final[2]), 5),
        ],
        dtype=np.int64,
    )
    royalty = (
        0.0
        if busted
        else float(
            get_top_royalty(final[0])
            + get_middle_royalty(final[1])
            + get_bottom_royalty(final[2])
        )
    )
    entry_count = 0
    if not busted:
        qualified, entry_count = check_fl_entry(list(final[0]))
        entry_count = entry_count if qualified else 0
    return {"busted": busted, "values": values, "royalty": royalty, "entry": entry_count}


def score_against_library(
    hero: dict,
    library: FlLibrary,
    indices: np.ndarray,
    opp_count: int,
) -> float:
    """Mean canonical score of one completed hero board over the FL boards."""
    fl_values = library.values[indices]
    fl_royalty = library.royalty[indices]
    fl_stay = library.stay[indices]
    fl_busted = library.busted[indices]
    fl_ev = FL_EV

    if hero["busted"]:
        base = np.where(fl_busted, 0.0, -6.0 - fl_royalty)
    else:
        lines = np.sign(hero["values"][None, :] - fl_values).sum(axis=1)
        scoop = np.where(lines == 3, 3.0, np.where(lines == -3, -3.0, 0.0))
        alive = lines + scoop + hero["royalty"] - fl_royalty
        base = np.where(fl_busted, 6.0 + hero["royalty"], alive)

    hero_entry_ev = float(fl_ev.get(hero["entry"], 0)) if not hero["busted"] else 0.0
    # A stay redeals the opponent's CURRENT count (joker-rule carryover).
    stay_cost = np.where(fl_stay & ~fl_busted, float(fl_ev.get(opp_count, 0)), 0.0)
    return float(np.mean(base + hero_entry_ev - stay_cost))


def action_values(root: dict, library: FlLibrary) -> dict[str, float]:
    board = Board(
        top=list(root["board"][0]),
        middle=list(root["board"][1]),
        bottom=list(root["board"][2]),
    )
    seen = seen_mask(
        [card for row in root["board"] for card in row]
        + list(root["dead"])
        + list(root["draw"])
    )
    indices = library.matching(seen)
    if len(indices) == 0:
        raise ValueError("no library boards compatible with this root")
    values: dict[str, float] = {}
    for action in get_turn_actions(list(root["draw"]), board):
        final = exact_late.apply_action(board, action)
        terminal = hero_terminal((final.top, final.middle, final.bottom))
        values[exact_late.action_key(action)] = score_against_library(
            terminal, library, indices, root["opp_count"]
        )
    return values


def encode_action(final_rows, root: dict, pool_cards: list[str]) -> list[float]:
    """Own-hand features plus deck context; no opponent block exists."""
    vector = hero_block(final_rows)
    counts = {"A": 0, "K": 0, "Q": 0}
    jokers = 0
    for card in pool_cards:
        if card in ("X1", "X2"):
            jokers += 1
        elif card[0] in counts:
            counts[card[0]] += 1
    one_hot = [0.0] * 4
    one_hot[root["opp_count"] - 14] = 1.0
    vector.extend(one_hot)
    vector.extend(
        [
            jokers / 2.0,
            counts["A"] / 4.0,
            counts["K"] / 4.0,
            counts["Q"] / 4.0,
            len(pool_cards) / 54.0,
            float(FL_EV.get(root["opp_count"], 0)) / 63.5,
            sum(counts.values()) / max(len(pool_cards), 1),
            (2 - jokers) / 2.0,
        ]
    )
    if len(vector) != FEATURE_SIZE:
        raise AssertionError(f"feature size drifted: {len(vector)}")
    return vector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--roots", type=int, default=5)
    parser.add_argument("--seed-base", type=int, default=41_000_000)
    args = parser.parse_args()

    library = FlLibrary(args.library)
    print(f"library: {len(library)} boards")
    for index in range(args.roots):
        root = sample_root(args.seed_base + index)
        try:
            values = action_values(root, library)
        except ValueError as error:
            print(f"seed {root['seed']}: {error}")
            continue
        best = max(values.values())
        matches = len(library.matching(seen_mask(
            [c for row in root["board"] for c in row] + root["dead"] + root["draw"]
        )))
        print(
            f"seed {root['seed']}: {len(values)} actions, {matches} FL boards, "
            f"best {best:+.3f}"
        )


if __name__ == "__main__":
    main()
