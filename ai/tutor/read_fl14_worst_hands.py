"""Read the roots an FL14 evaluator loses most on, and what it did there.

The gate reports a number; this reports the positions behind it.  The method
this project uses is train, gate, then read -- the one time a feature block was
added on a hunch instead (pool suits) the gate refuted it, and the one time the
worst hands were read first (the joint block) it took T2 from 0.265 to 0.056.

Row k of root R in the encoded split is action k of record R in the label file,
in file order, so the pick can be named without re-deriving any deal.

Usage:
    python -m ai.tutor.read_fl14_worst_hands --street t3 \
        --model D:/ofc_data/fl14_t3_model_v1/evaluator_best.pt \
        --data-dir D:/ofc_data/fl14_t3_teacher_v1 \
        --labels D:/ofc_data/fl14_teacher_v1/t3_labels.jsonl --worst 15
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ai.tutor.train_t4_first_evaluator import T4FirstEvaluator


def action_keys(labels: Path, street: str, wanted: set[int]) -> dict[int, list[tuple]]:
    """`root -> [(key, value), ...]` in the order the encoder consumed them."""
    out: dict[int, list[tuple]] = defaultdict(list)
    with labels.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            root = int(record["root"])
            if root not in wanted:
                continue
            if street == "t3":
                out[root] += [
                    (a["action_key"], float(a["value"]), record) for a in record["actions"]
                ]
            else:
                # The encoder drops placements the draw cannot make, so the
                # same filter runs here or the rows stop lining up.
                draw = record["draw"].split(",")
                for action in record["actions"]:
                    left = list(draw)
                    legal = True
                    for name in action["a"].split("@")[0].split("+"):
                        if name in left:
                            left.remove(name)
                        else:
                            legal = False
                            break
                    if legal:
                        out[root].append((action["a"], float(action["v"]), record))
    return out


def describe(key: str, record: dict, street: str) -> str:
    if street == "t3":
        rows, discard = key.rsplit("|", 1)
        return f"{rows}  (discard {discard})"
    return f"{record['board']}  +draw {record['draw']}  ->  {key}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--street", choices=["t3", "t4"], required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--worst", type=int, default=15)
    args = parser.parse_args()

    payload = np.load(args.data_dir / f"{args.split}.npz")
    x, y, roots = payload["x"], payload["y"], payload["roots"]

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    model = T4FirstEvaluator(checkpoint["input_dim"], tuple(checkpoint["hidden"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        prediction = model(
            (torch.tensor(x) - checkpoint["input_mean"]) / checkpoint["input_std"]
        ).numpy()

    groups: dict[int, list[int]] = defaultdict(list)
    for index, root in enumerate(roots):
        groups[int(root)].append(index)

    scored = []
    for root, indices in groups.items():
        truth = y[indices]
        pick = int(np.argmax(prediction[indices]))
        best = int(np.argmax(truth))
        scored.append((float(truth[best] - truth[pick]), root, pick, best, indices))
    scored.sort(reverse=True)

    worst = scored[: args.worst]
    keys = action_keys(args.labels, args.street, {row[1] for row in worst})

    print(f"# {args.street} worst {len(worst)} of {len(scored)} roots "
          f"(mean regret {np.mean([r[0] for r in scored]):.4f})\n")
    for regret, root, pick, best, indices in worst:
        actions = keys.get(root)
        if not actions or len(actions) != len(indices):
            print(f"root {root}: regret {regret:.3f} "
                  f"[{len(actions or [])} keys vs {len(indices)} rows -- skipped]")
            continue
        record = actions[0][2]
        print(f"root {root}  regret {regret:.3f}   actions {len(indices)}")
        print(f"  board {record['board']}  dead {record['dead']}  draw {record['draw']}")
        print(f"  teacher {describe(actions[best][0], record, args.street)}")
        print(f"          value {y[indices[best]]:+.3f}   model says {prediction[indices[best]]:+.3f}")
        print(f"  model   {describe(actions[pick][0], record, args.street)}")
        print(f"          value {y[indices[pick]]:+.3f}   model says {prediction[indices[pick]]:+.3f}")
        spread = float(y[indices].max() - y[indices].min())
        print(f"  root spread {spread:.3f}   model spread "
              f"{float(prediction[indices].max() - prediction[indices].min()):.3f}\n")


if __name__ == "__main__":
    main()
