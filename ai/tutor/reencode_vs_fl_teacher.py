"""Re-encode a vs-FL teacher with the sampled joint-outlook block.

Labels are the expensive half of a teacher and they are reused verbatim:
chunks are seed-deterministic, so this reconstructs each chunk's roots and
actions in generation order, verifies alignment by comparing the stored
actor block against a freshly computed one (any mismatch is fatal), asks the
Rust solver for the sampled joint block per action, and writes new chunks
with x = actor(48) + rowwise(41) + joint(8) + context(12) = 109 dims.

Usage:
    python -m ai.tutor.reencode_vs_fl_teacher --street t2 \
        --in-dir D:/ofc_data/t2_vs_fl_teacher_v1 \
        --out-dir D:/ofc_data/t2_vs_fl_teacher_v2
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

import ai.tutor.exact_late as exact_late
from ai.engine.action_space import get_turn_actions
from ai.engine.encoding import Board
from ai.tutor.generate_t0_vs_fl_teacher import rows_of_action_key, sample_t0_root
from ai.tutor.generate_t1_vs_fl_teacher import sample_t1_root
from ai.tutor.solver_paths import _solver_path
from ai.tutor.t2_policy_label_experiment import sample_t2_root
from ai.tutor.t3_second_features import actor_block
from ai.tutor.t4_vs_fl import CARD_INDEX, seen_mask

from ai.engine.encoding import ALL_CARDS

SPLITS = ("fit", "dev", "test")
NEW_FEATURE_SIZE = 48 + 41 + 8 + 12  # 109

SAMPLERS = {"t0": sample_t0_root, "t1": sample_t1_root, "t2": sample_t2_root}


def chunk_rows(street: str, root: dict) -> list[tuple[list, list]]:
    """(rows_after, dead_after) per action, in the generator's order."""
    out = []
    if street == "t0":
        # T0 actions came from the labeler's enumeration; reproduce it via the
        # same Rust order is not possible offline, so T0 re-encoding relies on
        # the stored action order being the labeler's sorted action_key order.
        raise NotImplementedError("t0 uses --from-labeler mode")
    board = Board(
        top=list(root["board"]["top"]),
        middle=list(root["board"]["middle"]),
        bottom=list(root["board"]["bottom"]),
    )
    for action in get_turn_actions(list(root["draw"]), board):
        after = exact_late.apply_action(board, action)
        out.append(
            (
                [list(after.top), list(after.middle), list(after.bottom)],
                list(root.get("dead", [])) + [action.discard],
            )
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--street", choices=["t1", "t2"], required=True)
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=150)
    parser.add_argument("--max-arrangements", type=int, default=32)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    workspace_root = args.workspace_root.resolve(strict=True)
    sampler = SAMPLERS[args.street]

    manifest = json.loads((args.in_dir / "manifest.json").read_text(encoding="utf-8"))
    seed, roots_total = manifest["seed"], manifest["roots"]
    out_chunk_dir = args.out_dir / "chunks"
    out_chunk_dir.mkdir(parents=True, exist_ok=True)
    scratch = args.out_dir / "scratch"
    scratch.mkdir(exist_ok=True)
    started = time.time()

    chunk_paths = sorted((args.in_dir / "chunks").glob("chunk_*.npz"))
    for chunk_path in chunk_paths:
        out_path = out_chunk_dir / chunk_path.name
        if out_path.exists():
            print(f"{chunk_path.name}: exists, skipping", flush=True)
            continue
        stem_number = int(chunk_path.stem.split("_")[1])
        # T2 v1 chunks are offset-named; T1/T0 carry the absolute first seed.
        first_root = stem_number if stem_number >= seed else seed + stem_number
        old = np.load(chunk_path)
        per_chunk = []
        index = first_root
        rows_expected = sum(old[f"{name}_x"].shape[0] for name in SPLITS)
        while sum(len(r[1]) for r in per_chunk) < rows_expected:
            root = sampler(index)
            per_chunk.append((root, chunk_rows(args.street, root)))
            index += 1
            if index - first_root > 2000:
                raise RuntimeError("row count never matched: alignment bug")

        # Order rows exactly as the generator buffered them: per root, per
        # action, appended to that root's split bucket.
        from ai.tutor.generate_t1_vs_fl_teacher import split_of as split_t1
        from ai.tutor.generate_t2_vs_fl_teacher import split_of as split_t2
        split_of = split_t1 if args.street == "t1" else split_t2
        per_split_rows: dict[str, list] = {name: [] for name in SPLITS}
        requests = []
        for root, actions in per_chunk:
            bucket = split_of(int(root["id"]))
            for position, (rows_after, dead_after) in enumerate(actions):
                request_id = f"{root['id']}/{position}"
                pool = None
                per_split_rows[bucket].append((request_id, rows_after, dead_after, root))
                requests.append(
                    {
                        "id": request_id,
                        "board": {
                            "top": rows_after[0],
                            "middle": rows_after[1],
                            "bottom": rows_after[2],
                        },
                        "pool": [
                            card for card in ALL_CARDS
                            if not (
                                (1 << CARD_INDEX[card])
                                & seen_mask(
                                    [c for r in rows_after for c in r] + dead_after
                                )
                            )
                        ],
                        "samples": args.samples,
                        "max_arrangements": args.max_arrangements,
                    }
                )
        with (scratch / "in.jsonl").open("w", encoding="utf-8") as handle:
            for request in requests:
                handle.write(json.dumps(request) + "\n")
        subprocess.run(
            [
                str(_solver_path(workspace_root)).replace("t4_first_exact.exe", "t4_first_exact_jo.exe"),
                "--input", str(scratch / "in.jsonl"),
                "--output", str(scratch / "out.jsonl"),
                "--fl-ev-config", str(workspace_root / "ai" / "config" / "fl_ev.json"),
                "--joint-outlook", "--chunk-size", "256",
            ],
            check=True,
        )
        blocks = {}
        with (scratch / "out.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    payload = json.loads(line)
                    blocks[payload["id"]] = payload["joint_block"]

        arrays = {}
        for name in SPLITS:
            old_x = old[f"{name}_x"]
            rows = per_split_rows[name]
            if len(rows) != old_x.shape[0]:
                raise RuntimeError(
                    f"{chunk_path.name} {name}: {len(rows)} reconstructed vs "
                    f"{old_x.shape[0]} stored -- alignment bug"
                )
            new_x = np.zeros((old_x.shape[0], NEW_FEATURE_SIZE), dtype=np.float32)
            for position, (request_id, rows_after, dead_after, root) in enumerate(rows):
                seen = seen_mask([c for r in rows_after for c in r] + dead_after)
                pool = [
                    card for card in ALL_CARDS
                    if not ((1 << CARD_INDEX[card]) & seen)
                ]
                actor, _categories = actor_block(rows_after, pool)
                stored_actor = old_x[position, :48]
                if not np.allclose(actor, stored_actor, atol=1e-5):
                    raise RuntimeError(
                        f"{chunk_path.name} {name} row {position}: actor block "
                        "mismatch -- alignment bug"
                    )
                new_x[position, :48] = old_x[position, :48]
                new_x[position, 48:89] = old_x[position, 48:89]
                new_x[position, 89:97] = np.asarray(blocks[request_id], dtype=np.float32)
                new_x[position, 97:109] = old_x[position, 89:101]
            arrays[f"{name}_x"] = new_x
            arrays[f"{name}_y"] = old[f"{name}_y"]
            arrays[f"{name}_j"] = old[f"{name}_j"]
        temp = out_path.with_suffix(".tmp.npz")
        np.savez_compressed(temp, **arrays)
        temp.replace(out_path)
        done = chunk_paths.index(chunk_path) + 1
        rate = done / (time.time() - started)
        print(
            f"[{done}/{len(chunk_paths)}] chunks, eta {(len(chunk_paths)-done)/rate/60:.0f} min",
            flush=True,
        )

    # Merge into splits + manifest.
    chunks = [np.load(path) for path in sorted(out_chunk_dir.glob("chunk_*.npz"))]

    def merged(key: str, empty_shape: tuple, dtype) -> np.ndarray:
        parts = [c[key] for c in chunks if c[key].size]
        if not parts:
            return np.zeros(empty_shape, dtype=dtype)
        return np.concatenate(parts)

    new_manifest = dict(manifest)
    new_manifest["feature_size"] = NEW_FEATURE_SIZE
    new_manifest["schema"] = manifest["schema"] + "+sampled_joint_v1"
    new_manifest["joint_samples"] = args.samples
    new_manifest["joint_max_arrangements"] = args.max_arrangements
    new_manifest["labels_reused_from"] = str(args.in_dir)
    new_manifest["splits"] = {}
    for name in SPLITS:
        x = merged(f"{name}_x", (0, NEW_FEATURE_SIZE), np.float32)
        y = merged(f"{name}_y", (0,), np.float32)
        j = merged(f"{name}_j", (0,), np.int8)
        np.savez_compressed(args.out_dir / f"{name}.npz", x=x, y=y, jokers=j)
        new_manifest["splits"][name] = {
            "rows": int(x.shape[0]),
            "ev_mean": float(y.mean()) if y.size else None,
        }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(new_manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(new_manifest["splits"], indent=2))


if __name__ == "__main__":
    main()
