"""Look at the hands the T4 first-seat evaluator gets most wrong.

This is the workflow the regular track's `t4_features.rs` header describes:
reading the errors of a trained version showed the remaining failures were not
hard positions but positions whose deciding facts were absent from the input.
Everything it found that way was exactly computable, so it was computed rather
than learned.  This script is how that reading is done here.

For each worst-error root it prints the full board layout, the exact and
predicted EV of every legal action, and a set of diagnostic facts that the
current encoder either does or does not carry, so a missing deciding fact is
visible rather than inferred.

Usage:
    python -m ai.tutor.inspect_t4_first_errors --roots 400 --show 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import ai.tutor.exact_late as exact_late
import ai.tutor.t4_bb_exact_vs_myopic_probe as probe
import ai.tutor.t4_first_features as features
from ai.engine.action_space import get_turn_actions
from ai.engine.encoding import Board
from ai.engine.game_engine import (
    check_fl_entry,
    evaluate_board_with_joker_constraint,
    evaluate_hand,
    hand_category,
    is_joker,
)
from ai.tutor.train_t4_first_evaluator import T4FirstEvaluator

CATEGORY_NAMES = (
    "high",
    "pair",
    "two-pair",
    "trips",
    "straight",
    "flush",
    "full-house",
    "quads",
    "str-flush",
)


def load_model(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = T4FirstEvaluator(checkpoint["input_dim"], tuple(checkpoint["hidden"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["input_mean"], checkpoint["input_std"]


def board_of(rows) -> Board:
    return Board(top=list(rows[0]), middle=list(rows[1]), bottom=list(rows[2]))


def row_summary(cards, capacity: int) -> str:
    if not cards:
        return "(empty)"
    if len(cards) == capacity:
        category = hand_category(evaluate_hand(list(cards), capacity))
        return f"{' '.join(cards)}  [{CATEGORY_NAMES[min(category, 8)]}]"
    return f"{' '.join(cards)}  [{len(cards)}/{capacity}]"


def opponent_diagnostics(opponent_rows, pool) -> dict:
    """Facts about the opponent that may or may not be in the encoder."""
    capacities = (3, 5, 5)
    rooms = [capacities[i] - len(opponent_rows[i]) for i in range(3)]
    suits_by_row = []
    for index in range(3):
        counts: dict[str, int] = {}
        for card in opponent_rows[index]:
            if not is_joker(card):
                counts[card[1]] = counts.get(card[1], 0) + 1
        suits_by_row.append(max(counts.values()) if counts else 0)
    # Live flush draw: a row one card short of a flush with that suit still live.
    flush_live = []
    for index in range(3):
        if rooms[index] == 0 or capacities[index] != 5:
            flush_live.append(False)
            continue
        need = capacities[index] - len(opponent_rows[index])
        counts = {}
        for card in opponent_rows[index]:
            if not is_joker(card):
                counts[card[1]] = counts.get(card[1], 0) + 1
        best_suit = max(counts, key=counts.get) if counts else None
        if best_suit is None:
            flush_live.append(False)
            continue
        live = sum(1 for card in pool if not is_joker(card) and card[1] == best_suit)
        live += sum(1 for card in pool if is_joker(card))
        flush_live.append(counts[best_suit] + need == 5 and live >= need)
    return {
        "rooms": rooms,
        "max_suit_per_row": suits_by_row,
        "flush_completable": flush_live,
        "jokers_in_pool": sum(1 for card in pool if is_joker(card)),
    }


def exact_tables(roots: list[dict], workspace_root: Path) -> list[dict[str, float]]:
    """Exact EVs from the Rust crate: ~38 roots/s against ~0.7 in Python."""
    from ai.tutor.generate_t4_first_teacher import label_roots

    labelled = label_roots(
        roots,
        workspace_root=workspace_root,
        scratch=workspace_root / "ai" / "reports" / "_inspect_scratch",
        chunk_size=256,
    )
    return [
        {row["action_key"]: float(row["ev"]) for row in payload["actions"]}
        for payload in labelled
    ]


def inspect(root: dict, model, mean, std, exact: dict[str, float]) -> dict:
    board = board_of(root["bb_board"])
    cache = features.NodeCache.for_root(
        root["bb_board"], root["btn_board"], root["draw"], root["bb_discards"]
    )
    keys, vectors = [], []
    for action in get_turn_actions(list(root["draw"]), board):
        final = exact_late.apply_action(board, action)
        keys.append(exact_late.action_key(action))
        vectors.append(features.encode_action((final.top, final.middle, final.bottom), cache))
    scaled = (torch.tensor(np.array(vectors, dtype=np.float32)) - mean) / std
    with torch.no_grad():
        predicted = model(scaled).numpy()
    errors = np.array([abs(predicted[i] - exact[keys[i]]) for i in range(len(keys))])
    best_exact = max(exact.values())
    chosen = keys[int(np.argmax(predicted))]
    return {
        "root": root,
        "keys": keys,
        "exact": exact,
        "predicted": predicted,
        "max_error": float(errors.max()),
        "mean_error": float(errors.mean()),
        "regret": float(best_exact - exact[chosen]),
        "chosen": chosen,
        "best_exact": best_exact,
        "pool": list(cache.pool),
    }


def report(case: dict) -> None:
    root = case["root"]
    print("\n" + "=" * 90)
    print(
        f"seed {root['seed']}   max|err| {case['max_error']:.2f}   "
        f"mean|err| {case['mean_error']:.2f}   regret {case['regret']:.3f}"
    )
    print("=" * 90)
    for label, rows in (("BB (hero)", root["bb_board"]), ("BTN (opp)", root["btn_board"])):
        capacities = (3, 5, 5)
        print(f"  {label}")
        for index, name in enumerate(("top", "mid", "bot")):
            print(f"    {name}: {row_summary(rows[index], capacities[index])}")
    print(f"  draw {list(root['draw'])}   hero dead {list(root['bb_discards'])}")

    diagnostics = opponent_diagnostics(root["btn_board"], case["pool"])
    print(f"  opponent rooms {diagnostics['rooms']}   "
          f"max suit/row {diagnostics['max_suit_per_row']}   "
          f"flush completable {diagnostics['flush_completable']}   "
          f"jokers in pool {diagnostics['jokers_in_pool']}")

    print(f"\n  {'exact':>9}{'pred':>9}{'err':>8}   action")
    order = sorted(range(len(case["keys"])), key=lambda i: -case["exact"][case["keys"][i]])
    for i in order:
        key = case["keys"][i]
        payload = json.loads(key)
        placed = ", ".join(f"{card}->{row}" for card, row in payload["placements"])
        marker = ""
        if key == case["chosen"]:
            marker += "  <- model picks"
        if case["exact"][key] == case["best_exact"]:
            marker += "  (exact best)"
        print(
            f"  {case['exact'][key]:>9.3f}{case['predicted'][i]:>9.3f}"
            f"{abs(case['predicted'][i] - case['exact'][key]):>8.2f}   "
            f"{placed} (discard {payload['discard']}){marker}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("D:/ofc_data/t4_first_model_pass1/evaluator_best.pt"))
    parser.add_argument("--roots", type=int, default=400)
    parser.add_argument("--seed-base", type=int, default=3_000_000)
    parser.add_argument("--show", type=int, default=8)
    args = parser.parse_args()

    model, mean, std = load_model(args.model)
    roots = [
        probe.sample_random_root(args.seed_base + index) for index in range(args.roots)
    ]
    tables = exact_tables(roots, Path.cwd())
    cases = [
        inspect(root, model, mean, std, table)
        for root, table in zip(roots, tables)
    ]

    regrets = np.array([case["regret"] for case in cases])
    hits = float((regrets == 0).mean())
    print(f"\nroots {len(cases)}   picks exact best {100 * hits:.1f}%   "
          f"mean regret {regrets.mean():.4f}   max regret {regrets.max():.2f}")

    print(f"\n### worst {args.show} by max absolute error")
    for case in sorted(cases, key=lambda item: -item["max_error"])[: args.show]:
        report(case)

    print(f"\n### worst {args.show} by regret (wrong action chosen)")
    for case in sorted(cases, key=lambda item: -item["regret"])[: args.show]:
        if case["regret"] > 0:
            report(case)


if __name__ == "__main__":
    main()
