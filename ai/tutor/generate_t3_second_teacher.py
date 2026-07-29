"""Generate T3 second-seat teacher data via the Rust solver.

Targets are BTN action values computed with the learned T4 first-seat
evaluator as the leaf.  Validated against the exact reference at MAE 0.303 /
correlation 0.991 over 24 action values, so this is an approximate teacher and
is labelled as such; it is not an exact solve.

The fit/dev/test split is keyed on the root seed and fixed before any value is
computed, so a later evaluation cannot reselect its holdout.

Usage:
    python -m ai.tutor.generate_t3_second_teacher --roots 50000 --out-dir <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from ai.tutor.generate_t4_first_teacher import _solver_path
from ai.tutor.t3_second_solver import sample_t3_second_root

SCHEMA = "ofc_t3_second_teacher/v1"
SPLITS = ("fit", "dev", "test")


def split_of(seed: int) -> str:
    digest = hashlib.sha256(f"t3-second-teacher-v1/{seed}".encode()).digest()
    bucket = int.from_bytes(digest[:4], "big") % 100
    return "fit" if bucket < 80 else ("dev" if bucket < 90 else "test")


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
    scratch.mkdir(parents=True, exist_ok=True)
    solver = _solver_path(workspace_root)
    counts = {name: 0 for name in SPLITS}
    handles = {
        name: (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") for name in SPLITS
    }
    started = time.time()
    try:
        for offset in range(0, roots, batch):
            size = min(batch, roots - offset)
            generated = [
                sample_t3_second_root(seed + offset + index) for index in range(size)
            ]
            in_path, out_path = scratch / "in.jsonl", scratch / "out.jsonl"
            with in_path.open("w", encoding="utf-8") as handle:
                for root in generated:
                    handle.write(
                        json.dumps(
                            {
                                "id": str(root["seed"]),
                                "bb": {
                                    "top": root["bb_board"][0],
                                    "middle": root["bb_board"][1],
                                    "bottom": root["bb_board"][2],
                                },
                                "btn": {
                                    "top": root["btn_board"][0],
                                    "middle": root["btn_board"][1],
                                    "bottom": root["btn_board"][2],
                                },
                                "btn_dead": root["btn_dead"],
                                "draw": root["draw"],
                                "draw_sample": draw_sample,
                            }
                        )
                        + "\n"
                    )
            subprocess.run(
                [
                    str(solver),
                    "--input", str(in_path),
                    "--output", str(out_path),
                    "--fl-ev-config", str(workspace_root / "ai" / "config" / "fl_ev.json"),
                    "--t3-second-model", str(model),
                    "--chunk-size", "256",
                ],
                check=True,
            )
            by_id = {}
            with out_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        payload = json.loads(line)
                        by_id[payload["id"]] = payload
            for root in generated:
                payload = by_id.get(str(root["seed"]))
                if payload is None:
                    raise RuntimeError(f"solver skipped root {root['seed']}")
                name = split_of(root["seed"])
                handles[name].write(
                    json.dumps(
                        {
                            "seed": root["seed"],
                            "bb_board": root["bb_board"],
                            "btn_board": root["btn_board"],
                            "btn_dead": root["btn_dead"],
                            "draw": root["draw"],
                            "actions": payload["actions"],
                        }
                    )
                    + "\n"
                )
                counts[name] += 1
            done = offset + size
            rate = done / (time.time() - started)
            print(
                f"[{done}/{roots}] {rate:.1f} roots/s "
                f"eta {(roots - done) / rate / 60:.1f} min",
                flush=True,
            )
    finally:
        for handle in handles.values():
            handle.close()

    manifest = {
        "schema": SCHEMA,
        "roots": roots,
        "seed": seed,
        "draw_sample": draw_sample,
        "model": str(model),
        "leaf": "learned_t4_first_evaluator",
        "target_is_exact": False,
        "validated_against_exact": {"mae": 0.3029, "correlation": 0.9913, "samples": 24},
        "split_rule": "sha256('t3-second-teacher-v1/<seed>') % 100 -> 80/10/10",
        "split_fixed_before_labelling": True,
        "elapsed_seconds": time.time() - started,
        "counts": counts,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=12_000_000)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("D:/ofc_data/t4_first_model_pass1_v2/evaluator.bin"),
    )
    parser.add_argument("--draw-sample", type=int, default=300)
    parser.add_argument("--batch", type=int, default=2_000)
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
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
