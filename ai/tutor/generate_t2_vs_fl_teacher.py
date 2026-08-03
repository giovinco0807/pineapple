"""Generate T2-vs-FL teacher data: playout labels, 101-dim features.

Labels come from the Rust T2-vs-FL playout labeler: the light 60-dim T3
policy chooses moves (validated 2026-08-03: its label effect sits inside the
draw-noise floor on every metric, argmax regret 0.026 vs floor 0.078) and
every scored number is exact library scoring at the T4 terminal.

Features are the light-lap T2 encoder: the actor's own 9-card after-board
(actor block 48) + its per-row completion outlook over the unseen pool
(rowwise 41, emitted by the labeler) + FL context (12) = 101 dims.  The
four-open-slot joint block is deliberately deferred; the regret gate decides
whether it is missed.

Usage:
    python -m ai.tutor.generate_t2_vs_fl_teacher --roots 12000 --out-dir <dir>
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
from ai.tutor.generate_t3_vs_fl_teacher import fl_context
from ai.tutor.generate_t4_first_teacher import _solver_path
from ai.tutor.t2_policy_label_experiment import sample_t2_root
from ai.tutor.t3_second_features import actor_block
from ai.tutor.t4_vs_fl import CARD_INDEX, seen_mask

DATASET_SCHEMA = "ofc_t2_vs_fl_teacher/v1_playout_labels"
SPLITS = ("fit", "dev", "test")
ACTOR_SIZE = 48
ROWWISE_SIZE = 41
FL_CONTEXT_SIZE = 12
FEATURE_SIZE = ACTOR_SIZE + ROWWISE_SIZE + FL_CONTEXT_SIZE  # 101


def split_of(seed: int) -> str:
    digest = hashlib.sha256(f"t2-vs-fl-teacher-v1/{seed}".encode()).digest()
    bucket = int.from_bytes(digest[:4], "big") % 100
    return "fit" if bucket < 80 else ("dev" if bucket < 90 else "test")


def encode_t2_action(rows_9, dead_2, opp_count: int, rowwise) -> list[float]:
    """101 dims: actor + own rowwise outlook (from Rust) + FL context."""
    seen = seen_mask([card for row in rows_9 for card in row] + list(dead_2))
    pool = [card for card in ALL_CARDS if not ((1 << CARD_INDEX[card]) & seen)]
    actor, _categories = actor_block(rows_9, pool)
    vector = (
        actor
        + [float(value) for value in rowwise]
        + fl_context(pool, opp_count)
    )
    if len(vector) != FEATURE_SIZE:
        raise AssertionError(f"feature size drifted: {len(vector)}")
    return vector


def run(
    *,
    roots: int,
    seed: int,
    out_dir: Path,
    t3_model: Path,
    t3_samples: int,
    t4_draw_sample: int,
    batch: int,
    workspace_root: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / "scratch"
    scratch.mkdir(exist_ok=True)
    chunk_dir = out_dir / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    started = time.time()

    for offset in range(0, roots, batch):
        size = min(batch, roots - offset)
        chunk_path = chunk_dir / f"chunk_{offset:07d}.npz"
        if chunk_path.exists():
            print(f"[{offset + size}/{roots}] chunk exists, skipping", flush=True)
            continue
        buffers = {name: {"x": [], "y": [], "j": []} for name in SPLITS}
        generated = [sample_t2_root(seed + offset + index) for index in range(size)]
        with (scratch / "in.jsonl").open("w", encoding="utf-8") as handle:
            for root in generated:
                payload = dict(root)
                payload["t3_samples"] = t3_samples
                payload["t4_draw_sample"] = t4_draw_sample
                handle.write(json.dumps(payload) + "\n")
        subprocess.run(
            [
                str(_solver_path(workspace_root)),
                "--input", str(scratch / "in.jsonl"),
                "--output", str(scratch / "out.jsonl"),
                "--fl-ev-config", str(workspace_root / "ai" / "config" / "fl_ev.json"),
                "--t2-vs-fl-library", "D:/ofc_data/fl_library_14_v3",
                "--t2-t3-model", str(t3_model),
                "--chunk-size", "16",
            ],
            check=True,
        )
        labels = {}
        with (scratch / "out.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    payload = json.loads(line)
                    labels[payload["id"]] = {
                        row["action_key"]: row for row in payload["actions"]
                    }
        for root in generated:
            table = labels.get(root["id"])
            if table is None:
                raise RuntimeError(f"labeler skipped root {root['id']}")
            board = Board(
                top=list(root["board"]["top"]),
                middle=list(root["board"]["middle"]),
                bottom=list(root["board"]["bottom"]),
            )
            jokers_visible = sum(
                1
                for card in (
                    root["board"]["top"] + root["board"]["middle"]
                    + root["board"]["bottom"] + root["dead"] + root["draw"]
                )
                if card in ("X1", "X2")
            )
            target = buffers[split_of(int(root["id"]))]
            for action in get_turn_actions(list(root["draw"]), board):
                key = exact_late.action_key(action)
                if key not in table:
                    continue
                row = table[key]
                after = exact_late.apply_action(board, action)
                target["x"].append(
                    encode_t2_action(
                        (after.top, after.middle, after.bottom),
                        list(root["dead"]) + [action.discard],
                        root["opp_count"],
                        row["own_rowwise_block"],
                    )
                )
                target["y"].append(row["value"])
                target["j"].append(jokers_visible)
        arrays = {}
        for name in SPLITS:
            arrays[f"{name}_x"] = np.asarray(buffers[name]["x"], dtype=np.float32)
            arrays[f"{name}_y"] = np.asarray(buffers[name]["y"], dtype=np.float32)
            arrays[f"{name}_j"] = np.asarray(buffers[name]["j"], dtype=np.int8)
        temp = chunk_path.with_suffix(".tmp.npz")
        np.savez_compressed(temp, **arrays)
        temp.replace(chunk_path)
        done = offset + size
        rate = done / (time.time() - started)
        print(
            f"[{done}/{roots}] {rate:.2f} roots/s eta {(roots-done)/rate/60:.0f} min",
            flush=True,
        )

    chunks = [np.load(path) for path in sorted(chunk_dir.glob("chunk_*.npz"))]

    def merged(key: str, empty_shape: tuple, dtype) -> np.ndarray:
        parts = [c[key] for c in chunks if c[key].size]
        if not parts:
            return np.zeros(empty_shape, dtype=dtype)
        return np.concatenate(parts)

    manifest = {
        "schema": DATASET_SCHEMA,
        "feature_size": FEATURE_SIZE,
        "label": "playout_light_t3_policy_library_scoring",
        "policy_validation": "argmax_regret 0.026 vs draw-noise floor 0.078 (60 roots)",
        "t3_samples": t3_samples,
        "t4_draw_sample": t4_draw_sample,
        "library": "D:/ofc_data/fl_library_14_v3",
        "roots": roots,
        "seed": seed,
        "split_rule": "sha256('t2-vs-fl-teacher-v1/<seed>') % 100 -> 80/10/10",
        "elapsed_seconds": time.time() - started,
        "splits": {},
    }
    for name in SPLITS:
        x = merged(f"{name}_x", (0, FEATURE_SIZE), np.float32)
        y = merged(f"{name}_y", (0,), np.float32)
        j = merged(f"{name}_j", (0,), np.int8)
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
    parser.add_argument("--roots", type=int, default=12_000)
    parser.add_argument("--seed", type=int, default=98_000_000)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--t3-model", type=Path,
        default=Path("D:/ofc_data/t3_vs_fl_model_v1/evaluator.bin"),
    )
    parser.add_argument("--t3-samples", type=int, default=50)
    parser.add_argument("--t4-draw-sample", type=int, default=100)
    parser.add_argument("--batch", type=int, default=500)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    manifest = run(
        roots=args.roots,
        seed=args.seed,
        out_dir=args.out_dir,
        t3_model=args.t3_model,
        t3_samples=args.t3_samples,
        t4_draw_sample=args.t4_draw_sample,
        batch=args.batch,
        workspace_root=args.workspace_root.resolve(strict=True),
    )
    print(json.dumps(manifest["splits"], indent=2))


if __name__ == "__main__":
    main()
