"""Generate T0-vs-FL teacher data: three-street playout labels, 101 dims.

T0 is the initial five-card placement -- the street the legacy serving
policy is weakest at, and the reason the light lap exists.  The Rust labeler
enumerates every distinct placement (~232), plays each forward with the
T1/T2 evaluators and the light T3 policy choosing moves, and prices the
terminal exactly against the FL board library.

Features are the same light-lap encoder as T1/T2: actor block over the
5-card placement + its rowwise outlook (from the labeler) + FL context.
Action identity comes from the labeler's action_key (sorted rows joined by
'|'), which this generator parses back into rows to encode.

Usage:
    python -m ai.tutor.generate_t0_vs_fl_teacher --roots 1500 --out-dir <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np

from ai.tutor.generate_t2_vs_fl_teacher import FEATURE_SIZE, SPLITS, encode_t2_action
from ai.tutor.solver_paths import _solver_path, count_library_args
from ai.tutor.t2_policy_label_experiment import ALL_CARDS

DATASET_SCHEMA = "ofc_t0_vs_fl_teacher/v1_playout_labels"


def sample_t0_root(seed: int) -> dict:
    rng = random.Random(seed)
    deck = ALL_CARDS[:]
    rng.shuffle(deck)
    return {
        "id": str(seed),
        "cards": deck[:5],
        "opp_count": rng.choices([14, 15, 16, 17], weights=[6, 2, 1, 1])[0],
    }


def split_of(seed: int) -> str:
    digest = hashlib.sha256(f"t0-vs-fl-teacher-v1/{seed}".encode()).digest()
    bucket = int.from_bytes(digest[:4], "big") % 100
    return "fit" if bucket < 80 else ("dev" if bucket < 90 else "test")


def rows_of_action_key(key: str) -> tuple[list[str], list[str], list[str]]:
    top, middle, bottom = (part.split(",") if part else [] for part in key.split("|"))
    return list(top), list(middle), list(bottom)


def run(
    *,
    roots: int,
    seed: int,
    out_dir: Path,
    library: str,
    t1_model: Path,
    t2_model: Path,
    t3_model: Path,
    t1_samples: int,
    t2_samples: int,
    t3_samples: int,
    t4_draw_sample: int,
    batch: int,
    workspace_root: Path,
    force_opp_count: int = 0,
    merge: bool = True,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / "scratch"
    scratch.mkdir(exist_ok=True)
    chunk_dir = out_dir / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    started = time.time()

    for offset in range(0, roots, batch):
        size = min(batch, roots - offset)
        chunk_path = chunk_dir / f"chunk_{seed + offset:012d}.npz"
        if chunk_path.exists():
            print(f"[{offset + size}/{roots}] chunk exists, skipping", flush=True)
            continue
        buffers = {name: {"x": [], "y": [], "j": []} for name in SPLITS}
        generated = [sample_t0_root(seed + offset + index) for index in range(size)]
        if force_opp_count:
            for root in generated:
                root["opp_count"] = force_opp_count
        with (scratch / "in.jsonl").open("w", encoding="utf-8") as handle:
            for root in generated:
                payload = dict(root)
                payload["t1_samples"] = t1_samples
                payload["t2_samples"] = t2_samples
                payload["t3_samples"] = t3_samples
                payload["t4_draw_sample"] = t4_draw_sample
                handle.write(json.dumps(payload) + "\n")
        subprocess.run(
            [
                str(_solver_path(workspace_root)),
                "--input", str(scratch / "in.jsonl"),
                "--output", str(scratch / "out.jsonl"),
                "--fl-ev-config", str(workspace_root / "ai" / "config" / "fl_ev.json"),
                "--t0-vs-fl-library", library,                *count_library_args(),

                "--t0-t1-model", str(t1_model),
                "--t1-t2-model", str(t2_model),
                "--t2-t3-model", str(t3_model),
                "--chunk-size", "8",
            ],
            check=True,
        )
        labels = {}
        with (scratch / "out.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    payload = json.loads(line)
                    labels[payload["id"]] = payload["actions"]
        for root in generated:
            actions = labels.get(root["id"])
            if actions is None:
                raise RuntimeError(f"labeler skipped root {root['id']}")
            jokers_visible = sum(
                1 for card in root["cards"] if card in ("X1", "X2")
            )
            target = buffers[split_of(int(root["id"]))]
            for row in actions:
                rows_5 = rows_of_action_key(row["action_key"])
                target["x"].append(
                    encode_t2_action(
                        rows_5, [], root["opp_count"], row["own_rowwise_block"],
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
            f"[{done}/{roots}] {rate:.3f} roots/s eta {(roots-done)/rate/60:.0f} min",
            flush=True,
        )

    if not merge:
        return {"chunks": len(list(chunk_dir.glob("chunk_*.npz")))}

    chunks = [np.load(path) for path in sorted(chunk_dir.glob("chunk_*.npz"))]

    def merged(key: str, empty_shape: tuple, dtype) -> np.ndarray:
        parts = [c[key] for c in chunks if c[key].size]
        if not parts:
            return np.zeros(empty_shape, dtype=dtype)
        return np.concatenate(parts)

    manifest = {
        "schema": DATASET_SCHEMA,
        "feature_size": FEATURE_SIZE,
        "label": "three_street_playout_t1_t2_t3_movers_library_scoring",
        "t1_samples": t1_samples,
        "t2_samples": t2_samples,
        "t3_samples": t3_samples,
        "t4_draw_sample": t4_draw_sample,
        "library": library,
        "roots": roots,
        "seed": seed,
        "split_rule": "sha256('t0-vs-fl-teacher-v1/<seed>') % 100 -> 80/10/10",
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
    parser.add_argument("--roots", type=int, default=1_500)
    parser.add_argument("--seed", type=int, default=130_000_000)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--library", default="D:/ofc_data/fl_library_14_v3")
    parser.add_argument(
        "--t1-model", type=Path,
        default=Path("D:/ofc_data/t1_vs_fl_model_v1/evaluator.bin"),
    )
    parser.add_argument(
        "--t2-model", type=Path,
        default=Path("D:/ofc_data/t2_vs_fl_model_v1/evaluator.bin"),
    )
    parser.add_argument(
        "--t3-model", type=Path,
        default=Path("D:/ofc_data/t3_vs_fl_model_v1/evaluator.bin"),
    )
    parser.add_argument("--t1-samples", type=int, default=6)
    parser.add_argument("--t2-samples", type=int, default=4)
    parser.add_argument("--t3-samples", type=int, default=3)
    parser.add_argument("--t4-draw-sample", type=int, default=40)
    parser.add_argument(
        "--force-opp-count", type=int, default=0,
        help="Fix every root to this FL count (per-count teachers).",
    )
    parser.add_argument("--batch", type=int, default=25)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--no-merge", action="store_true")
    args = parser.parse_args()
    manifest = run(
        roots=args.roots,
        seed=args.seed,
        out_dir=args.out_dir,
        library=args.library,
        t1_model=args.t1_model,
        t2_model=args.t2_model,
        t3_model=args.t3_model,
        t1_samples=args.t1_samples,
        t2_samples=args.t2_samples,
        t3_samples=args.t3_samples,
        t4_draw_sample=args.t4_draw_sample,
        batch=args.batch,
        workspace_root=args.workspace_root.resolve(strict=True),
        force_opp_count=args.force_opp_count,
        merge=not args.no_merge,
    )
    print(json.dumps(manifest.get("splits", manifest), indent=2))


if __name__ == "__main__":
    main()
