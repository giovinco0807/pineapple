"""Held-out regret against stored teacher labels, grouped per root.

The full gates relabel fresh roots with two independent referees; that costs
hours per street.  This is the cheap companion: the teacher's own test split
(roots the model never trained on), regrouped per root, model argmax charged
against the stored label-best.  No referee floor -- the number reads as an
upper bound on charged regret under the teacher's own label noise -- but it
is free, so every retrain can be sanity-checked in seconds.

Usage:
    python -m ai.tutor.eval_vs_fl_heldout_regret --street t0 \
        --teacher-dir D:/ofc_data/t0_vs_fl_teacher_v1 \
        --model-dir D:/ofc_data/t0_vs_fl_model_v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ai.tutor.reencode_vs_fl_teacher import SAMPLERS, chunk_rows
from ai.tutor.train_t4_first_evaluator import T4FirstEvaluator

SPLIT_FNS = {}


def _splits():
    from ai.tutor.generate_t0_vs_fl_teacher import split_of as t0
    from ai.tutor.generate_t1_vs_fl_teacher import split_of as t1
    from ai.tutor.generate_t2_vs_fl_teacher import split_of as t2
    return {"t0": t0, "t1": t1, "t2": t2}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--street", choices=["t0", "t1", "t2"], required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    split_of = _splits()[args.street]
    sampler = SAMPLERS[args.street]

    manifest = json.loads(
        (args.teacher_dir / "manifest.json").read_text(encoding="utf-8")
    )
    test = np.load(args.teacher_dir / "test.npz")
    x, y = test["x"], test["y"]

    checkpoint = torch.load(
        args.model_dir / "evaluator_best.pt", map_location="cpu", weights_only=False
    )
    model = T4FirstEvaluator(checkpoint["input_dim"], tuple(checkpoint["hidden"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    if checkpoint["input_dim"] != x.shape[1]:
        raise SystemExit(
            f"model expects {checkpoint['input_dim']} dims, teacher has {x.shape[1]}"
        )
    mean = checkpoint["input_mean"].numpy().astype(np.float32)
    scale = checkpoint["input_std"].numpy().astype(np.float32)
    with torch.no_grad():
        predicted = (
            model(torch.from_numpy((x - mean) / scale)).squeeze(-1).numpy()
        )

    # Regroup rows per root by replaying the deterministic generation order.
    seed, roots_total = manifest["seed"], manifest["roots"]
    regrets, best_picked, cursor = [], 0, 0
    for index in range(seed, seed + roots_total):
        if split_of(index) != "test":
            continue
        root = sampler(index)
        count = len(chunk_rows(args.street, root))
        if cursor + count > len(y):
            raise SystemExit("row grouping overran the test split: alignment bug")
        values = y[cursor:cursor + count]
        scores = predicted[cursor:cursor + count]
        regret = float(values.max() - values[int(scores.argmax())])
        regrets.append(regret)
        best_picked += int(regret == 0.0)
        cursor += count
    if cursor != len(y):
        raise SystemExit(
            f"grouping consumed {cursor} of {len(y)} test rows: alignment bug"
        )

    regrets_arr = np.asarray(regrets)
    result = {
        "street": args.street,
        "roots": len(regrets),
        "mean_regret_vs_stored_labels": float(regrets_arr.mean()),
        "best_pick_rate": best_picked / len(regrets),
        "regret_p95": float(np.percentile(regrets_arr, 95)),
        "note": "no referee floor; label noise included in the charge",
    }
    (args.model_dir / "heldout_regret.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
