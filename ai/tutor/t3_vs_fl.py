"""T3 decision for the normal player facing a Fantasyland opponent.

With no opponent decisions anywhere, the continuation from a T3 action is just
the hero's own T4 draw:

    value(T3 action) = E_draw [ max_a4  T4_vs_FL(final(a4)) ]

and the learned T4-vs-FL evaluator already marginalizes the hidden FL hand
given everything the hero has seen -- the draw lands in the seen set, so
feeding sampled draws through the evaluator integrates over (draw, FL hand)
with exactly the right joint.  No reply model, no shared-block subtlety.

Usage:
    python -m ai.tutor.t3_vs_fl --model D:/ofc_data/t4_vs_fl_model_pilot/evaluator_best.pt --roots 5
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

import ai.tutor.exact_late as exact_late
from ai.engine.action_space import get_turn_actions
from ai.engine.encoding import ALL_CARDS, Board
from ai.tutor.t4_vs_fl import CARD_INDEX, encode_action, seen_mask

FOUR_OPEN_SHAPES = (
    (3, 5, 1), (3, 4, 2), (3, 3, 3), (3, 2, 4), (3, 1, 5),
    (2, 5, 2), (2, 4, 3), (2, 3, 4), (2, 2, 5),
    (1, 5, 3), (1, 4, 4), (1, 3, 5),
)


def sample_root(seed: int, opp_count: int = 14) -> dict:
    """One physical T3 root for the normal seat, opponent in FL."""
    rng = random.Random(seed)
    deck = ALL_CARDS[:]
    rng.shuffle(deck)

    def take(count: int) -> list[str]:
        cards = deck[:count]
        del deck[:count]
        return cards

    shape = rng.choice(FOUR_OPEN_SHAPES)
    board = [take(shape[0]), take(shape[1]), take(shape[2])]
    dead = take(2)          # T1/T2 discards
    draw = take(3)
    return {
        "seed": seed,
        "board": board,
        "dead": dead,
        "draw": draw,
        "opp_count": opp_count,
    }


class LearnedT4VsFl:
    # torch is imported here rather than at module scope: label
    # generators import this module only for sample_root, and fleet
    # workers would otherwise need an 800 MB dependency to draw a root.
    def __init__(self, path: Path, device: str = "cpu") -> None:
        import torch

        from ai.tutor.train_t4_first_evaluator import T4FirstEvaluator

        checkpoint = torch.load(path, map_location=device, weights_only=False)
        self.model = T4FirstEvaluator(
            checkpoint["input_dim"], tuple(checkpoint["hidden"])
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.mean = checkpoint["input_mean"]
        self.std = checkpoint["input_std"]

    def node_values(self, vectors, groups) -> np.ndarray:
        if not vectors:
            return np.zeros(0)
        batch = (
            torch.tensor(np.asarray(vectors, dtype=np.float32)) - self.mean
        ) / self.std
        with torch.no_grad():
            predicted = self.model(batch).numpy()
        out = np.empty(len(groups))
        cursor = 0
        for index, size in enumerate(groups):
            out[index] = predicted[cursor : cursor + size].max()
            cursor += size
        return out


def action_values(
    root: dict,
    evaluator: LearnedT4VsFl,
    *,
    draw_sample: int = 300,
) -> dict[str, float]:
    board = Board(
        top=list(root["board"][0]),
        middle=list(root["board"][1]),
        bottom=list(root["board"][2]),
    )
    values: dict[str, float] = {}
    for action in get_turn_actions(list(root["draw"]), board):
        after = exact_late.apply_action(board, action)
        rows_11 = (after.top, after.middle, after.bottom)
        dead_3 = list(root["dead"]) + [action.discard]
        seen_now = (
            [card for row in rows_11 for card in row]
            + dead_3
            + []
        )
        unseen = [
            card
            for card in ALL_CARDS
            if not ((1 << CARD_INDEX[card]) & seen_mask(seen_now))
        ]
        rng = random.Random(root["seed"] * 977 + hash(action.discard) % 9973)
        draws = [tuple(rng.sample(unseen, 3)) for _ in range(draw_sample)]

        t4_board = Board(
            top=list(rows_11[0]), middle=list(rows_11[1]), bottom=list(rows_11[2])
        )
        vectors, groups = [], []
        for draw in draws:
            pool_cards = [card for card in unseen if card not in draw]
            t4_root = {"opp_count": root["opp_count"]}
            count = 0
            for t4_action in get_turn_actions(list(draw), t4_board):
                final = exact_late.apply_action(t4_board, t4_action)
                vectors.append(
                    encode_action(
                        (final.top, final.middle, final.bottom), t4_root, pool_cards
                    )
                )
                count += 1
            groups.append(count)
        node_values = evaluator.node_values(vectors, groups)
        values[exact_late.action_key(action)] = float(node_values.mean())
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--roots", type=int, default=5)
    parser.add_argument("--seed-base", type=int, default=51_000_000)
    parser.add_argument("--draw-sample", type=int, default=300)
    args = parser.parse_args()
    evaluator = LearnedT4VsFl(args.model)
    import time

    for index in range(args.roots):
        root = sample_root(args.seed_base + index)
        started = time.time()
        values = action_values(root, evaluator, draw_sample=args.draw_sample)
        best = max(values.values())
        print(
            f"seed {root['seed']}: {len(values)} actions, best {best:+.3f}, "
            f"{time.time() - started:.2f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
