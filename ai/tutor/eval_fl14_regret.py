"""The FL14 gate: charged regret against the teacher's own noise floor.

Correlation and MAE are not the criterion here, and that is not a stylistic
preference -- it has been measured twice on this project.  A T3 evaluator with
corr 0.931 carried regret 0.279, and a T2 one with corr 0.990 carried a gap of
0.265.  A model can track the level of a value function beautifully and still
pick the wrong action, because picking depends on differences *within* a root,
which is where nearly all the useful signal is and almost none of the variance.

So the number reported is:

    regret   mean over roots of  y[teacher's best] - y[model's pick]
    floor    the same quantity when the "model" is a second labeling pass
    gap      regret - floor

The floor exists because the teacher is not exact.  Everything about it is --
the opponent's play is a best response over its whole frontier, the scoring is
exact, hero's draw is enumerated -- except *how many* opponents were drawn.
Two passes over the same positions with independent opponent streams disagree
by some amount, and no model can be asked to beat that.  Charge the floor the
way the model is charged, symmetrized so neither pass is privileged.

Usage:
    # the floor, from two label passes over the same roots
    python -m ai.tutor.eval_fl14_regret --floor \
        --labels-a D:/ofc_data/fl14_teacher_v1/t3_labels.jsonl \
        --labels-b D:/ofc_data/fl14_floor_v1/t3_labels.jsonl

    # the model, on the split it never trained on
    python -m ai.tutor.eval_fl14_regret --model D:/ofc_data/fl14_t3_model/evaluator_best.pt \
        --data-dir D:/ofc_data/fl14_t3_teacher_v1 --split test
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def charged(values: list[float], pick: int) -> float:
    """What taking `pick` costs against this root's best action."""
    return max(values) - values[pick]


def model_regret(model_path: Path, data_dir: Path, split: str) -> dict:
    import torch

    from ai.tutor.train_t4_first_evaluator import T4FirstEvaluator

    payload = np.load(data_dir / f"{split}.npz")
    if "roots" not in payload:
        raise SystemExit(
            f"{split}.npz has no `roots` array -- re-encode; regret is a "
            "within-root quantity and cannot be recovered from the rows alone"
        )
    x, y, roots = payload["x"], payload["y"], payload["roots"]

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = T4FirstEvaluator(checkpoint["input_dim"], tuple(checkpoint["hidden"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    features = (torch.tensor(x) - checkpoint["input_mean"]) / checkpoint["input_std"]
    with torch.no_grad():
        prediction = model(features).numpy()

    groups: dict[int, list[int]] = defaultdict(list)
    for index, root in enumerate(roots):
        groups[int(root)].append(index)

    regrets, top1 = [], 0
    for indices in groups.values():
        truth = [float(y[i]) for i in indices]
        pick = max(range(len(indices)), key=lambda k: prediction[indices[k]])
        regrets.append(charged(truth, pick))
        top1 += truth[pick] == max(truth)
    return summarize(regrets, {"top1": top1 / max(len(regrets), 1)})


def read_actions(path: Path) -> dict[int, dict[str, float]]:
    """`root -> action key -> value`, for either street's label file."""
    out: dict[int, dict[str, float]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            root = int(record["root"])
            if "action_key" in record["actions"][0]:
                actions = {a["action_key"]: float(a["value"]) for a in record["actions"]}
            else:
                # A T4 record's actions are keyed within its own draw, so the
                # draw joins the key -- two passes harvest different draws and
                # only the shared ones can be compared.
                actions = {
                    f"{record['draw']}/{a['a']}": float(a["v"]) for a in record["actions"]
                }
            # A root can contribute more than one T4 decision; merge them.
            out.setdefault(root, {}).update(actions)
    return out


def floor_regret(path_a: Path, path_b: Path) -> dict:
    a, b = read_actions(path_a), read_actions(path_b)
    shared_roots = sorted(set(a) & set(b))
    if not shared_roots:
        raise SystemExit("the two passes share no roots -- was the deal seed the same?")
    regrets, skipped = [], 0
    for root in shared_roots:
        keys = sorted(set(a[root]) & set(b[root]))
        if len(keys) < 2:
            skipped += 1
            continue
        va = [a[root][k] for k in keys]
        vb = [b[root][k] for k in keys]
        # Symmetric: each pass is charged for trusting the other's pick, so
        # neither one is treated as the truth the other is measured against.
        pick_b = max(range(len(keys)), key=lambda k: vb[k])
        pick_a = max(range(len(keys)), key=lambda k: va[k])
        regrets.append(0.5 * (charged(va, pick_b) + charged(vb, pick_a)))
    return summarize(
        regrets, {"roots_compared": len(regrets), "roots_skipped": skipped}
    )


def summarize(regrets: list[float], extra: dict) -> dict:
    array = np.asarray(regrets, dtype=np.float64)
    if array.size == 0:
        raise SystemExit("nothing to summarize")
    return {
        "roots": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
        "zero_fraction": float((array <= 1e-9).mean()),
        **extra,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floor", action="store_true")
    parser.add_argument("--labels-a", type=Path)
    parser.add_argument("--labels-b", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--floor-value", type=float, default=None,
                        help="a floor measured earlier, to report the gap against")
    args = parser.parse_args()

    report: dict = {}
    if args.floor:
        if not (args.labels_a and args.labels_b):
            raise SystemExit("--floor needs --labels-a and --labels-b")
        report["floor"] = floor_regret(args.labels_a, args.labels_b)
    if args.model:
        if not args.data_dir:
            raise SystemExit("--model needs --data-dir")
        report["model"] = model_regret(args.model, args.data_dir, args.split)
        report["split"] = args.split
        floor = args.floor_value
        if floor is None and "floor" in report:
            floor = report["floor"]["mean"]
        if floor is not None:
            report["gap_over_floor"] = report["model"]["mean"] - floor
    if not report:
        raise SystemExit("nothing asked for: pass --floor and/or --model")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
