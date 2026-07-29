"""Generate exact T4 first-seat teacher data for the learned evaluator.

Roots come from the probe's seeded 54-card generator, so board shapes are
unfiltered and Joker 0/1/2 strata all appear naturally.  Labels are the exact
uniform-deal EVs from the `t4_first_exact` Rust crate, which is bitwise
identical to `ai/tutor/t4_bb_exact_resolver.py` on every checked root.

Roots are split by seed *before* labelling, and the split is recorded in the
manifest, so a later evaluation cannot quietly reselect its holdout.

Usage:
    python -m ai.tutor.generate_t4_first_teacher --roots 50000 --out-dir <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import ai.tutor.exact_late as exact_late
import ai.tutor.t4_first_features as features
import ai.tutor.t4_bb_exact_vs_myopic_probe as probe
from ai.engine.action_space import get_turn_actions
from ai.engine.encoding import Board

DATASET_SCHEMA = "ofc_t4_first_teacher/v1"
SPLITS = ("fit", "dev", "test")


def _solver_path(workspace_root: Path) -> Path:
    suffix = ".exe" if sys.platform.startswith("win") else ""
    return (
        workspace_root
        / "ai"
        / "rust_solver"
        / "target"
        / "release"
        / f"t4_first_exact{suffix}"
    )


def split_of(seed: int) -> str:
    """Deterministic seed-keyed split, fixed before any label is computed."""
    digest = hashlib.sha256(f"t4-first-teacher-v1/{seed}".encode()).digest()
    bucket = int.from_bytes(digest[:4], "big") % 100
    if bucket < 80:
        return "fit"
    if bucket < 90:
        return "dev"
    return "test"


def label_roots(
    roots: list[dict],
    *,
    workspace_root: Path,
    scratch: Path,
    chunk_size: int,
) -> list[dict]:
    solver = _solver_path(workspace_root)
    if not solver.is_file():
        raise FileNotFoundError(
            "build first: cargo build --release -p t4_first_exact"
        )
    scratch.mkdir(parents=True, exist_ok=True)
    in_path = scratch / "roots.jsonl"
    out_path = scratch / "labels.jsonl"
    with in_path.open("w", encoding="utf-8") as handle:
        for index, root in enumerate(roots):
            handle.write(
                json.dumps(
                    {
                        "id": str(index),
                        "bb": {
                            "top": list(root["bb_board"][0]),
                            "middle": list(root["bb_board"][1]),
                            "bottom": list(root["bb_board"][2]),
                        },
                        "btn": {
                            "top": list(root["btn_board"][0]),
                            "middle": list(root["btn_board"][1]),
                            "bottom": list(root["btn_board"][2]),
                        },
                        "draw": list(root["draw"]),
                        "dead": list(root["bb_discards"]),
                    }
                )
                + "\n"
            )
    subprocess.run(
        [
            str(solver),
            "--input",
            str(in_path),
            "--output",
            str(out_path),
            "--fl-ev-config",
            str(workspace_root / "ai" / "config" / "fl_ev.json"),
            "--chunk-size",
            str(chunk_size),
        ],
        check=True,
    )
    labels = []
    with out_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                labels.append(json.loads(line))
    labels.sort(key=lambda payload: int(payload["id"]))
    if len(labels) != len(roots):
        raise RuntimeError(
            f"solver returned {len(labels)} results for {len(roots)} roots"
        )
    return labels


def build_rows(root: dict, labelled: dict) -> list[tuple[list[float], float, int]]:
    """One feature row per legal action, sharing the node cache."""
    board = Board(
        top=list(root["bb_board"][0]),
        middle=list(root["bb_board"][1]),
        bottom=list(root["bb_board"][2]),
    )
    cache = features.NodeCache.for_root(
        root["bb_board"],
        root["btn_board"],
        root["draw"],
        root["bb_discards"],
        joint_block=labelled.get("opponent_joint_block"),
    )
    ev_by_key = {row["action_key"]: float(row["ev"]) for row in labelled["actions"]}
    all_cards = (
        [card for rows in (root["bb_board"], root["btn_board"]) for row in rows for card in row]
        + list(root["draw"])
    )
    jokers = sum(1 for card in all_cards if card in ("X1", "X2"))

    rows = []
    for action in get_turn_actions(list(root["draw"]), board):
        key = exact_late.action_key(action)
        if key not in ev_by_key:
            raise RuntimeError(f"solver did not label action {key}")
        final = exact_late.apply_action(board, action)
        vector = features.encode_action((final.top, final.middle, final.bottom), cache)
        rows.append((vector, ev_by_key[key], jokers))
    return rows


def run(
    *,
    roots: int,
    seed: int,
    workspace_root: Path,
    out_dir: Path,
    batch: int,
    chunk_size: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / "scratch"
    buffers: dict[str, dict[str, list]] = {
        name: {"x": [], "y": [], "j": []} for name in SPLITS
    }
    started = time.time()
    labelled_roots = 0

    for offset in range(0, roots, batch):
        size = min(batch, roots - offset)
        generated = [
            probe.sample_random_root(seed + offset + index) for index in range(size)
        ]
        labels = label_roots(
            generated,
            workspace_root=workspace_root,
            scratch=scratch,
            chunk_size=chunk_size,
        )
        for root, labelled in zip(generated, labels):
            target = buffers[split_of(root["seed"])]
            for vector, ev, jokers in build_rows(root, labelled):
                target["x"].append(vector)
                target["y"].append(ev)
                target["j"].append(jokers)
        labelled_roots += size
        elapsed = time.time() - started
        rate = labelled_roots / elapsed if elapsed > 0 else 0.0
        remaining = (roots - labelled_roots) / rate if rate > 0 else float("nan")
        print(
            f"[{labelled_roots}/{roots}] {rate:.1f} roots/s "
            f"eta {remaining/60:.1f} min",
            flush=True,
        )

    manifest = {
        "schema": DATASET_SCHEMA,
        "feature_schema": features.FEATURE_SCHEMA,
        "feature_size": features.FEATURE_SIZE,
        "label": "exact_uniform_deal_ev_t4_first",
        "roots": roots,
        "seed": seed,
        "split_rule": "sha256('t4-first-teacher-v1/<seed>') % 100 -> 80/10/10",
        "split_fixed_before_labelling": True,
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
            "ev_min": float(y.min()) if y.size else None,
            "ev_max": float(y.max()) if y.size else None,
            "joker_strata": {
                str(value): int((j == value).sum()) for value in sorted(set(j.tolist()))
            },
        }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=770_000)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--batch", type=int, default=2_000)
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()
    manifest = run(
        roots=args.roots,
        seed=args.seed,
        workspace_root=args.workspace_root.resolve(strict=True),
        out_dir=args.out_dir,
        batch=args.batch,
        chunk_size=args.chunk_size,
    )
    print(json.dumps(manifest["splits"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
