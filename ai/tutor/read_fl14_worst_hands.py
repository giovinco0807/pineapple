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
from ai.tutor.encode_fl14_teacher import stable_root_id


def action_keys(labels: Path, street: str, wanted: set[int]) -> dict[int, list[tuple]]:
    """`root -> [(key, value), ...]` in the order the encoder consumed them."""
    out: dict[int, list[tuple]] = defaultdict(list)
    root_ids: dict[int, str] = {}
    with labels.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            root = stable_root_id(record)
            if root not in wanted:
                continue
            record_id = str(record.get("id", root))
            if root in root_ids and root_ids[root] != record_id:
                raise RuntimeError(
                    f"root key {root} is reused by ids {root_ids[root]} and {record_id}; "
                    "the encoded split cannot identify their actions safely"
                )
            root_ids[root] = record_id
            if street in ("t2", "t3"):
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
    if street in ("t2", "t3"):
        rows, discard = key.rsplit("|", 1)
        return f"{rows}  (discard {discard})"
    return f"{record['board']}  +draw {record['draw']}  ->  {key}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--street", choices=["t2", "t3", "t4"], required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--worst", type=int, default=15)
    parser.add_argument(
        "--json-out", type=Path,
        help="Also write the scored worst roots as structured JSON.",
    )
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

    regrets = np.asarray([row[0] for row in scored], dtype=np.float64)

    worst = scored[: args.worst]
    keys = action_keys(args.labels, args.street, {row[1] for row in worst})

    print(f"# {args.street} worst {len(worst)} of {len(scored)} roots "
          f"(mean regret {regrets.mean():.4f})\n")
    structured = []
    for regret, root, pick, best, indices in worst:
        actions = keys.get(root)
        if not actions or len(actions) != len(indices):
            print(f"root {root}: regret {regret:.3f} "
                  f"[{len(actions or [])} keys vs {len(indices)} rows -- skipped]")
            continue
        record = actions[0][2]
        rows = record["board"].split("|")[:3]
        visible = ",".join(rows + [record.get("dead", ""), record.get("draw", "")])
        structured.append(
            {
                "root": root,
                "regret": regret,
                "actions": len(indices),
                "shape": [0 if not row else len(row.split(",")) for row in rows],
                "visible_jokers": visible.count("X"),
                "board": record["board"],
                "dead": record["dead"],
                "draw": record["draw"],
                "teacher_action": actions[best][0],
                "teacher_value": float(y[indices[best]]),
                "teacher_prediction": float(prediction[indices[best]]),
                "model_action": actions[pick][0],
                "model_value": float(y[indices[pick]]),
                "model_prediction": float(prediction[indices[pick]]),
                "root_spread": float(y[indices].max() - y[indices].min()),
                "model_spread": float(prediction[indices].max() - prediction[indices].min()),
            }
        )
        print(f"root {root}  regret {regret:.3f}   actions {len(indices)}")
        print(f"  board {record['board']}  dead {record['dead']}  draw {record['draw']}")
        print(f"  teacher {describe(actions[best][0], record, args.street)}")
        print(f"          value {y[indices[best]]:+.3f}   model says {prediction[indices[best]]:+.3f}")
        print(f"  model   {describe(actions[pick][0], record, args.street)}")
        print(f"          value {y[indices[pick]]:+.3f}   model says {prediction[indices[pick]]:+.3f}")
        spread = float(y[indices].max() - y[indices].min())
        print(f"  root spread {spread:.3f}   model spread "
              f"{float(prediction[indices].max() - prediction[indices].min()):.3f}\n")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                {
                    "street": args.street,
                    "split": args.split,
                    "roots": len(scored),
                    "mean_regret": float(regrets.mean()),
                    "regret_quantiles": {
                        "p50": float(np.quantile(regrets, 0.50)),
                        "p90": float(np.quantile(regrets, 0.90)),
                        "p95": float(np.quantile(regrets, 0.95)),
                        "p99": float(np.quantile(regrets, 0.99)),
                        "max": float(regrets.max()),
                    },
                    "zero_regret_rate": float(np.mean(regrets == 0.0)),
                    "worst": structured,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
