"""Judge the T3-BTN model against a high-sample referee.

    python scripts/gate_three_max_t3.py --corpus corpus.jsonl --model m.pt \
        --roots 300 --referee-samples 384 --workers 14

Both the model and the corpus labels are scored on the SAME referee, which is a
fresh labelling of the same roots at a much higher sample count from a disjoint
seed stream.  That gives the two numbers that mean something together:

``label_regret``  what the corpus's own answer gives up on the referee -- the
                  noise floor, which no model trained on those labels can beat
``model_regret``  what the model's answer gives up on the referee
``gap``           model_regret - label_regret, the only part that is the
                  model's fault

A large gap says fix the model (features, capacity, data volume).  A large
floor with a small gap says fix the labels (more samples per root) -- and
heads-up spent real money learning that these two failures look identical if
you only report one number.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from ofc_regular.three_max.exact import evaluate_t3  # noqa: E402
from ofc_regular.three_max.features import encode_record_action  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_three_max_t3_floor import SEED_OFFSET, observation_of  # noqa: E402
from train_three_max_t3 import Ranker, stable_holdout  # noqa: E402


def _referee(args: tuple) -> dict[str, float]:
    line, samples, fl_ev_14 = args
    record = json.loads(line)
    ranked = evaluate_t3(
        observation_of(record),
        samples=samples,
        seed=SEED_OFFSET + 31 * record["seed"],
        fl_ev_per_pair={14: fl_ev_14},
    )
    return {
        json.dumps([list(pair) for pair in candidate.action.placements]): candidate.ev
        for candidate in ranked
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--roots", type=int, default=300)
    parser.add_argument("--referee-samples", type=int, default=384)
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fl-ev-14", type=float, default=9.6)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    lines = []
    with args.corpus.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if stable_holdout(record["seed"], fraction=args.holdout_fraction):
                lines.append(line)
            if len(lines) >= args.roots:
                break
    print(f"judging {len(lines)} held-out roots at {args.referee_samples} samples", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        referees = list(
            pool.map(
                _referee,
                [(line, args.referee_samples, args.fl_ev_14) for line in lines],
            )
        )

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    model = Ranker(input_dim=checkpoint["input_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    mean = checkpoint["mean"]
    std = checkpoint["std"]

    label_regrets: list[float] = []
    model_regrets: list[float] = []
    spreads: list[float] = []
    widths: list[int] = []
    model_top1 = 0
    label_top1 = 0
    for index, (line, referee) in enumerate(zip(lines, referees)):
        record = json.loads(line)
        # The corpus stores actions sorted by label EV, so index 0 is the
        # teacher's answer.  Ties in a model's scores resolve to index 0, which
        # would hand a constant-output model the floor for free.  Permute.
        order = np.random.default_rng(0x5EED + index).permutation(
            len(record["actions"])
        )
        record["actions"] = [record["actions"][i] for i in order]
        keys = [
            json.dumps([list(pair) for pair in action["placements"]])
            for action in record["actions"]
        ]
        truth = [referee[key] for key in keys]
        best = max(truth)
        best_index = truth.index(best)
        # Regret is in raw points, and the raw scale is a property of the ROOT
        # DISTRIBUTION, not of the model: heads-up-played roots have 2.4x the
        # within-root spread that referee-played ones do.  Two corpora can only
        # be compared after dividing it out.
        spreads.append(best - min(truth))
        widths.append(len(truth))

        label_pick = int(np.argmax([action["ev"] for action in record["actions"]]))
        label_regrets.append(best - truth[label_pick])
        label_top1 += label_pick == best_index

        features = np.asarray(
            [
                encode_record_action(record, index).features
                for index in range(len(record["actions"]))
            ],
            dtype=np.float32,
        )
        with torch.no_grad():
            scores = model(torch.from_numpy((features - mean) / std)).numpy()
        model_pick = int(np.argmax(scores))
        model_regrets.append(best - truth[model_pick])
        model_top1 += model_pick == best_index

    mean_spread = statistics.fmean(spreads) or 1.0
    report = {
        "roots": len(lines),
        "referee_samples": args.referee_samples,
        "mean_fan_width": statistics.fmean(widths),
        "mean_within_root_spread": mean_spread,
        "label_regret_normalised": statistics.fmean(label_regrets) / mean_spread,
        "model_regret_normalised": statistics.fmean(model_regrets) / mean_spread,
        "gap_normalised": (
            statistics.fmean(model_regrets) - statistics.fmean(label_regrets)
        ) / mean_spread,
        "label_regret": statistics.fmean(label_regrets),
        "model_regret": statistics.fmean(model_regrets),
        "gap": statistics.fmean(model_regrets) - statistics.fmean(label_regrets),
        "label_top1": label_top1 / len(lines),
        "model_top1": model_top1 / len(lines),
        "model_regret_p95": sorted(model_regrets)[int(0.95 * (len(model_regrets) - 1))],
        "label_regret_p95": sorted(label_regrets)[int(0.95 * (len(label_regrets) - 1))],
        "model": str(args.model),
        "corpus": str(args.corpus),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
