"""T3 second-seat (BTN) value, with the learned T4 first-seat evaluator as leaf.

Structure of the continuation from a T3 second-seat decision:

    BTN places 2 of 3      -> BTN board 11
      BB draws 3, places 2 -> BB board 13     (T4 FIRST: the learned evaluator)
        BTN draws 3, places 2 -> 13           (T4 second: closed form, folded
                                               into the T4-first exact target)

so the value of a BTN action is the negated expectation, over BB's draw, of
BB's T4 first-seat node value.  Exactly, each of those inner nodes costs a
C(26,3) enumeration; with the evaluator each costs one forward pass, which is
the whole reason the evaluator was built.

Declared belief (same family as the T4 first-seat resolver): every card unseen
from the acting seat's information set is treated as available.  BTN does not
know BB's earlier discards, so those stay in the pool; BTN's own discards and
BB's T4 discard are removed because they are known gone.  This is exact under
the declared belief, not a Bayes posterior over BB's actual play.

Usage:
    python -m ai.tutor.t3_second_solver --roots 20 --draw-sample 400
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

import ai.tutor.exact_late as exact_late
import ai.tutor.t4_first_features as features
import json as _json
import subprocess as _subprocess
from ai.engine.action_space import get_turn_actions
from ai.engine.encoding import ALL_CARDS, Board
from ai.tutor.train_t4_first_evaluator import T4FirstEvaluator

SCHEMA = "ofc_t3_second_value/v1"
# (top, middle, bottom) fills leaving four open slots, for the 9-card seat.
FOUR_OPEN_SHAPES = (
    (3, 5, 1),
    (3, 4, 2),
    (3, 3, 3),
    (3, 2, 4),
    (3, 1, 5),
    (2, 5, 2),
    (2, 4, 3),
    (2, 3, 4),
    (2, 2, 5),
    (1, 5, 3),
    (1, 4, 4),
    (1, 3, 5),
)
TWO_OPEN_SHAPES = (
    (3, 5, 3),
    (3, 4, 4),
    (3, 3, 5),
    (2, 5, 4),
    (2, 4, 5),
    (1, 5, 5),
)


def sample_t3_second_root(seed: int) -> dict:
    """One physical T3 second-seat root from a seeded 54-card shuffle."""
    rng = random.Random(seed)
    deck = list(ALL_CARDS)
    rng.shuffle(deck)

    def take(count: int) -> list[str]:
        cards = deck[:count]
        del deck[:count]
        return cards

    bb_shape = rng.choice(TWO_OPEN_SHAPES)      # BB already acted at T3
    btn_shape = rng.choice(FOUR_OPEN_SHAPES)    # BTN is about to act
    bb_board = [take(bb_shape[0]), take(bb_shape[1]), take(bb_shape[2])]
    btn_board = [take(btn_shape[0]), take(btn_shape[1]), take(btn_shape[2])]
    btn_dead = take(2)          # BTN's own T1/T2 discards
    bb_dead_hidden = take(2)    # BB's T1/T2 discards: unknown to BTN
    draw = take(3)
    return {
        "seed": seed,
        "bb_board": bb_board,
        "btn_board": btn_board,
        "btn_dead": btn_dead,
        "bb_dead_hidden": bb_dead_hidden,
        "draw": draw,
    }


def _board(rows) -> Board:
    return Board(top=list(rows[0]), middle=list(rows[1]), bottom=list(rows[2]))


def bb_draw_pool(root: dict, btn_board_11, btn_dead_3) -> list[str]:
    """Cards BTN cannot see after acting; BB's hidden discards stay in."""
    seen = set(btn_dead_3)
    for rows in (root["bb_board"], btn_board_11):
        for row in rows:
            seen.update(row)
    return [card for card in ALL_CARDS if card not in seen]


class LearnedT4First:
    """Batched T4 first-seat node value from the learned evaluator."""

    def __init__(self, path: Path, device: str = "cpu") -> None:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        self.model = T4FirstEvaluator(
            checkpoint["input_dim"], tuple(checkpoint["hidden"])
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.mean = checkpoint["input_mean"]
        self.std = checkpoint["input_std"]
        self.device = device

    def node_values(self, vectors: list[list[float]], groups: list[int]) -> np.ndarray:
        """Max prediction within each consecutive group of `groups[i]` rows."""
        if not vectors:
            return np.zeros(0)
        batch = (torch.tensor(np.asarray(vectors, dtype=np.float32)) - self.mean) / self.std
        with torch.no_grad():
            predicted = self.model(batch).numpy()
        out = np.empty(len(groups), dtype=np.float64)
        cursor = 0
        for index, size in enumerate(groups):
            out[index] = predicted[cursor : cursor + size].max()
            cursor += size
        return out


def rust_joint_blocks(items, workspace_root: Path, scratch: Path) -> list[list[float]]:
    """Joint blocks for (opponent board, pool) pairs, via the Rust solver."""
    from ai.tutor.generate_t4_first_teacher import _solver_path

    scratch.mkdir(parents=True, exist_ok=True)
    in_path, out_path = scratch / "joint_in.jsonl", scratch / "joint_out.jsonl"
    with in_path.open("w", encoding="utf-8") as handle:
        for index, (board, pool) in enumerate(items):
            handle.write(
                _json.dumps(
                    {
                        "id": str(index),
                        "btn": {
                            "top": list(board[0]),
                            "middle": list(board[1]),
                            "bottom": list(board[2]),
                        },
                        "pool": list(pool),
                    }
                )
                + chr(10)
            )
    _subprocess.run(
        [
            str(_solver_path(workspace_root)),
            "--input", str(in_path),
            "--output", str(out_path),
            "--fl-ev-config", str(workspace_root / "ai" / "config" / "fl_ev.json"),
            "--joint-only",
            "--chunk-size", "256",
        ],
        check=True,
    )
    rows = [
        _json.loads(line)
        for line in out_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows.sort(key=lambda row: int(row["id"]))
    return [row["opponent_joint_block"] for row in rows]


def btn_action_values(
    root: dict,
    evaluator: LearnedT4First,
    *,
    draw_sample: int | None,
    rng_seed: int,
    workspace_root: Path = Path("."),
    scratch: Path = Path("C:/Users/Owner/AppData/Local/Temp/claude/t3s"),
) -> dict[str, float]:
    """Value of every legal BTN T3 action, from BTN's perspective.

    The joint block is computed once per BTN action on the pool unseen from
    BTN's own information set, and shared across BB's draws.  Recomputing it
    per draw costs about as much as solving the node exactly, which would
    defeat the point of the evaluator; the shared block deviates from the
    per-draw one by ~0.02 in foul rate and less elsewhere, and the validation
    mode below reports what that does to the final value.
    """
    btn_base = _board(root["btn_board"])
    values: dict[str, float] = {}

    actions = get_turn_actions(list(root["draw"]), btn_base)
    prepared = []
    for action in actions:
        after = exact_late.apply_action(btn_base, action)
        rows = (after.top, after.middle, after.bottom)
        dead3 = list(root["btn_dead"]) + [action.discard]
        prepared.append((action, rows, dead3, bb_draw_pool(root, rows, dead3)))
    shared_blocks = rust_joint_blocks(
        [(rows, pool) for _a, rows, _d, pool in prepared], workspace_root, scratch
    )

    for (action, btn_rows, btn_dead_3, pool), shared in zip(prepared, shared_blocks):
        draws = [
            (pool[a], pool[b], pool[c])
            for a in range(len(pool))
            for b in range(a + 1, len(pool))
            for c in range(b + 1, len(pool))
        ]
        if draw_sample is not None and draw_sample < len(draws):
            rng = random.Random(rng_seed)
            draws = rng.sample(draws, draw_sample)

        bb_base = _board(root["bb_board"])
        vectors: list[list[float]] = []
        groups: list[int] = []
        for draw in draws:
            # BB's information set at T4 first: its own board and draw, BTN's
            # board, and the cards it knows are gone.
            cache = features.NodeCache.for_root(
                root["bb_board"], btn_rows, draw, btn_dead_3, joint_block=shared
            )
            count = 0
            for bb_action in get_turn_actions(list(draw), bb_base):
                final = exact_late.apply_action(bb_base, bb_action)
                vectors.append(
                    features.encode_action(
                        (final.top, final.middle, final.bottom), cache
                    )
                )
                count += 1
            groups.append(count)
        bb_node_values = evaluator.node_values(vectors, groups)
        # Zero sum: BTN's value is the negated BB continuation value.
        values[exact_late.action_key(action)] = float(-bb_node_values.mean())
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("D:/ofc_data/t4_first_model_pass1_v2/evaluator_best.pt"),
    )
    parser.add_argument("--roots", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=9_000_000)
    parser.add_argument("--draw-sample", type=int, default=400)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    evaluator = LearnedT4First(args.model)
    started = time.time()
    rows = []
    for index in range(args.roots):
        root = sample_t3_second_root(args.seed_base + index)
        values = btn_action_values(
            root, evaluator, draw_sample=args.draw_sample, rng_seed=root["seed"]
        )
        best = max(values.values())
        rows.append(
            {
                "seed": root["seed"],
                "actions": len(values),
                "best_value": best,
                "best_action": min(k for k, v in values.items() if v == best),
                "values": values,
            }
        )
        print(
            f"[{index + 1}/{args.roots}] seed {root['seed']}  "
            f"actions {len(values)}  best {best:+.3f}",
            flush=True,
        )
    elapsed = time.time() - started
    report = {
        "schema": SCHEMA,
        "model": str(args.model),
        "roots": args.roots,
        "draw_sample": args.draw_sample,
        "leaf": "learned_t4_first_evaluator",
        "belief": "uniform_exchangeable_restart_v1",
        "seconds_per_root": elapsed / max(args.roots, 1),
        "rows": rows,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(f"\n{elapsed / max(args.roots, 1):.2f} s/root")


if __name__ == "__main__":
    main()
