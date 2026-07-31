"""Generate T3-vs-FL teacher data: Rust labels, Python features.

Labels come from the Rust T3-vs-FL mode (parity vs the Python path: worst
difference 3e-6 over 45 action values), which runs the learned T4-vs-FL v2
evaluator as its leaf.  These are approximate teachers -- the leaf's own
noise-adjusted RMSE is ~0.7 -- and the manifest says so.

Features for the T3 evaluator are the actor's own 11-card partial board
(the t3_second actor block: made values, rooms, ordering slack, FL facts)
plus the same 12-dim FL context the T4 encoder uses.  60 dims, no opponent
block, because the opponent still has nothing observable.

Usage:
    python -m ai.tutor.generate_t3_vs_fl_teacher --roots 30000 --out-dir <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

import ai.tutor.exact_late as exact_late
from ai.engine.action_space import get_turn_actions
from ai.engine.encoding import ALL_CARDS, Board
from ai.tutor.generate_t4_first_teacher import _solver_path
from ai.tutor.t3_second_features import actor_block
from ai.tutor.t3_vs_fl import sample_root
from ai.tutor.t4_vs_fl import CARD_INDEX, seen_mask
from ai.mcts.rollout_evaluator import RolloutEvaluator

DATASET_SCHEMA = "ofc_t3_vs_fl_teacher/v1"
SPLITS = ("fit", "dev", "test")
ACTOR_SIZE = 48
FL_CONTEXT_SIZE = 12
FEATURE_SIZE = ACTOR_SIZE + FL_CONTEXT_SIZE  # 60


def split_of(seed: int) -> str:
    digest = hashlib.sha256(f"t3-vs-fl-teacher-v1/{seed}".encode()).digest()
    bucket = int.from_bytes(digest[:4], "big") % 100
    return "fit" if bucket < 80 else ("dev" if bucket < 90 else "test")


def fl_context(pool_cards: list[str], opp_count: int) -> list[float]:
    """Pinned to t4_vs_fl.encode_action's context block."""
    counts = {"A": 0, "K": 0, "Q": 0}
    jokers = 0
    for card in pool_cards:
        if card in ("X1", "X2"):
            jokers += 1
        elif card[0] in counts:
            counts[card[0]] += 1
    one_hot = [0.0] * 4
    one_hot[opp_count - 14] = 1.0
    return one_hot + [
        jokers / 2.0,
        counts["A"] / 4.0,
        counts["K"] / 4.0,
        counts["Q"] / 4.0,
        len(pool_cards) / 54.0,
        float(RolloutEvaluator.FL_EV.get(opp_count, 0)) / 63.5,
        sum(counts.values()) / max(len(pool_cards), 1),
        (2 - jokers) / 2.0,
    ]


def encode_t3_action(rows_11, dead_3, opp_count: int) -> list[float]:
    seen = seen_mask([card for row in rows_11 for card in row] + list(dead_3))
    pool = [card for card in ALL_CARDS if not ((1 << CARD_INDEX[card]) & seen)]
    actor, _categories = actor_block(rows_11, pool)
    vector = actor + fl_context(pool, opp_count)
    if len(vector) != FEATURE_SIZE:
        raise AssertionError(f"feature size drifted: {len(vector)}")
    return vector


def run(
    *,
    roots: int,
    seed: int,
    out_dir: Path,
    model: Path,
    draw_sample: int,
    batch: int,
    workspace_root: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / "scratch"
    scratch.mkdir(exist_ok=True)
    buffers = {name: {"x": [], "y": [], "j": []} for name in SPLITS}
    started = time.time()

    for offset in range(0, roots, batch):
        size = min(batch, roots - offset)
        generated = [sample_root(seed + offset + index) for index in range(size)]
        with (scratch / "in.jsonl").open("w", encoding="utf-8") as handle:
            for root in generated:
                handle.write(
                    json.dumps(
                        {
                            "id": str(root["seed"]),
                            "board": {
                                "top": root["board"][0],
                                "middle": root["board"][1],
                                "bottom": root["board"][2],
                            },
                            "dead": root["dead"],
                            "draw": root["draw"],
                            "opp_count": root["opp_count"],
                            "draw_sample": draw_sample,
                        }
                    )
                    + "\n"
                )
        subprocess.run(
            [
                str(_solver_path(workspace_root)),
                "--input", str(scratch / "in.jsonl"),
                "--output", str(scratch / "out.jsonl"),
                "--fl-ev-config", str(workspace_root / "ai" / "config" / "fl_ev.json"),
                "--t3-vs-fl-model", str(model),
                "--chunk-size", "64",
            ],
            check=True,
        )
        labels = {}
        with (scratch / "out.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    payload = json.loads(line)
                    labels[payload["id"]] = {
                        row["action_key"]: row["value"] for row in payload["actions"]
                    }
        for root in generated:
            table = labels.get(str(root["seed"]))
            if table is None:
                raise RuntimeError(f"solver skipped root {root['seed']}")
            board = Board(
                top=list(root["board"][0]),
                middle=list(root["board"][1]),
                bottom=list(root["board"][2]),
            )
            jokers_visible = sum(
                1
                for card in (
                    [c for row in root["board"] for c in row]
                    + root["dead"]
                    + root["draw"]
                )
                if card in ("X1", "X2")
            )
            target = buffers[split_of(root["seed"])]
            for action in get_turn_actions(list(root["draw"]), board):
                key = exact_late.action_key(action)
                if key not in table:
                    continue
                after = exact_late.apply_action(board, action)
                rows_11 = (after.top, after.middle, after.bottom)
                dead_3 = list(root["dead"]) + [action.discard]
                target["x"].append(
                    encode_t3_action(rows_11, dead_3, root["opp_count"])
                )
                target["y"].append(table[key])
                target["j"].append(jokers_visible)
        done = offset + size
        rate = done / (time.time() - started)
        print(
            f"[{done}/{roots}] {rate:.1f} roots/s eta {(roots-done)/rate/60:.0f} min",
            flush=True,
        )

    manifest = {
        "schema": DATASET_SCHEMA,
        "feature_size": FEATURE_SIZE,
        "label": "rust_t3_vs_fl_value_over_learned_t4_leaf",
        "target_is_exact": False,
        "leaf_model": str(model),
        "draw_sample": draw_sample,
        "roots": roots,
        "seed": seed,
        "split_rule": "sha256('t3-vs-fl-teacher-v1/<seed>') % 100 -> 80/10/10",
        "elapsed_seconds": time.time() - started,
        "splits": {},
    }
    for name in SPLITS:
        x = np.asarray(buffers[name]["x"], dtype=np.float32)
        y = np.asarray(buffers[name]["y"], dtype=np.float32)
        j = np.asarray(buffers[name]["j"], dtype=np.int8)
        np.savez_compressed(out_dir / f"{name}.npz", x=x, y=y, jokers=j)
        manifest["splits"][name] = {
            "rows": int(x.shape[0]),
            "ev_mean": float(y.mean()) if y.size else None,
            "ev_std": float(y.std()) if y.size else None,
        }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=75_000_000)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--model", type=Path,
        default=Path("D:/ofc_data/t4_vs_fl_model_v2/evaluator.bin"),
    )
    parser.add_argument("--draw-sample", type=int, default=300)
    parser.add_argument("--batch", type=int, default=1_000)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    manifest = run(
        roots=args.roots,
        seed=args.seed,
        out_dir=args.out_dir,
        model=args.model,
        draw_sample=args.draw_sample,
        batch=args.batch,
        workspace_root=args.workspace_root.resolve(strict=True),
    )
    print(json.dumps(manifest["splits"], indent=2))


if __name__ == "__main__":
    main()
