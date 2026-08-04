"""Does a light T3 playout policy change T2-vs-FL labels?

Three labeling runs over the same T2 roots:
  A: full 109-dim T3-vs-FL v2 evaluator chooses playout moves,
  B: light 60-dim (actor+context) v1 model chooses -- ~30x cheaper features,
  C: run A's policy with independent draw seeds -- the label noise floor.

Common random numbers make A-vs-B exact: both see identical T3 draws, and a
playout contributes a difference only when the two policies choose different
T3 actions.  Acceptance per the distillation plan: the A-B label difference
(mean shift and argmax flips) must sit within the A-C noise band.

Usage:
    python -m ai.tutor.t2_policy_label_experiment --roots 60 --t3-samples 50
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np

from ai.tutor.solver_paths import _solver_path

ALL_CARDS = [rank + suit for suit in "shdc" for rank in "23456789TJQKA"] + ["X1", "X2"]
ROW_CAPACITY = (3, 5, 5)


def sample_t2_root(seed: int) -> dict:
    rng = random.Random(seed)
    deck = ALL_CARDS[:]
    rng.shuffle(deck)
    # A legal 7-card split: rows within capacity, six slots open.
    while True:
        top = rng.randint(0, 3)
        mid = rng.randint(0, 5)
        bot = 7 - top - mid
        if 0 <= bot <= 5:
            break
    cards = deck[:7]
    return {
        "id": str(seed),
        "board": {
            "top": cards[:top],
            "middle": cards[top : top + mid],
            "bottom": cards[top + mid :],
        },
        "dead": [deck[7]],
        "draw": deck[8:11],
        "opp_count": rng.choices([14, 15, 16, 17], weights=[6, 2, 1, 1])[0],
    }


def label(
    roots: list[dict],
    out_path: Path,
    model: str,
    id_prefix: str,
    t3_samples: int,
    t4_draw_sample: int,
    workspace_root: Path,
) -> dict:
    in_path = out_path.with_suffix(".in.jsonl")
    with in_path.open("w", encoding="utf-8") as handle:
        for root in roots:
            payload = dict(root)
            payload["id"] = id_prefix + root["id"]
            payload["t3_samples"] = t3_samples
            payload["t4_draw_sample"] = t4_draw_sample
            handle.write(json.dumps(payload) + "\n")
    started = time.time()
    subprocess.run(
        [
            str(_solver_path(workspace_root)),
            "--input", str(in_path),
            "--output", str(out_path),
            "--fl-ev-config", str(workspace_root / "ai" / "config" / "fl_ev.json"),
            "--t2-vs-fl-library", "D:/ofc_data/fl_library_14_v3",
            "--t2-t3-model", model,
            "--chunk-size", "8",
        ],
        check=True,
    )
    elapsed = time.time() - started
    out = {}
    with out_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                out[payload["id"].removeprefix(id_prefix)] = {
                    row["action_key"]: row["value"] for row in payload["actions"]
                }
    print(f"  run done in {elapsed/60:.1f} min", flush=True)
    return out


def compare(name: str, base: dict, other: dict) -> dict:
    deltas, flips, best_gaps = [], 0, []
    for root_id, base_actions in base.items():
        other_actions = other[root_id]
        keys = sorted(base_actions)
        base_values = np.array([base_actions[k] for k in keys])
        other_values = np.array([other_actions[k] for k in keys])
        deltas.extend(np.abs(base_values - other_values).tolist())
        base_pick = int(base_values.argmax())
        other_pick = int(other_values.argmax())
        flips += int(base_pick != other_pick)
        # What the base labeler thinks is lost by trusting the other's argmax.
        best_gaps.append(float(base_values.max() - base_values[other_pick]))
    result = {
        "mean_abs_delta": float(np.mean(deltas)),
        "p95_abs_delta": float(np.percentile(deltas, 95)),
        "argmax_flip_rate": flips / len(base),
        "mean_argmax_regret": float(np.mean(best_gaps)),
    }
    print(f"{name}: {json.dumps(result)}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=int, default=60)
    parser.add_argument("--seed", type=int, default=97_000_000)
    parser.add_argument("--t3-samples", type=int, default=50)
    parser.add_argument("--t4-draw-sample", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=Path("D:/ofc_data/t2_policy_experiment"))
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    workspace_root = args.workspace_root.resolve(strict=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    roots = [sample_t2_root(args.seed + index) for index in range(args.roots)]

    full = "D:/ofc_data/t3_vs_fl_model_v2/evaluator.bin"
    light = "D:/ofc_data/t3_vs_fl_model_v1/evaluator.bin"
    print("run A: full policy", flush=True)
    run_a = label(roots, args.out_dir / "a_full.jsonl", full, "",
                  args.t3_samples, args.t4_draw_sample, workspace_root)
    print("run B: light policy (same draws)", flush=True)
    run_b = label(roots, args.out_dir / "b_light.jsonl", light, "",
                  args.t3_samples, args.t4_draw_sample, workspace_root)
    print("run C: full policy, independent draws (noise floor)", flush=True)
    run_c = label(roots, args.out_dir / "c_floor.jsonl", full, "b:",
                  args.t3_samples, args.t4_draw_sample, workspace_root)

    report = {
        "roots": args.roots,
        "t3_samples": args.t3_samples,
        "t4_draw_sample": args.t4_draw_sample,
        "light_vs_full": compare("A-vs-B (policy effect)", run_a, run_b),
        "noise_floor": compare("A-vs-C (draw noise)", run_a, run_c),
    }
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
