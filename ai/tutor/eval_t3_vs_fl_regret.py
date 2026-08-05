"""Regret gate for a T3-vs-FL evaluator: action selection against exact labels.

Correlation and MAE passed v1 while action selection failed it (mean regret
0.279, 45x the T4-vs-FL figure), so this is the acceptance metric, computed
the way the model will actually be used: on fresh roots, the model picks the
argmax action from its features, and is charged the label-EV difference to
the label-best action.  Labels come from the Rust direct-library scorer --
the same path that labeled the training data, on disjoint seeds.

Gate (from ai/reports/t3_vs_fl_v1_20260731): mean regret < 0.05.

Usage:
    python -m ai.tutor.eval_t3_vs_fl_regret \
        --model-dir D:/ofc_data/t3_vs_fl_model_v2 --roots 500
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

import ai.tutor.exact_late as exact_late
from ai.engine.action_space import get_turn_actions
from ai.engine.encoding import Board
from ai.tutor.generate_t3_vs_fl_teacher import encode_t3_action
from ai.tutor.solver_paths import _solver_path, count_library_args
from ai.tutor.t3_vs_fl import sample_root
from ai.tutor.train_t4_first_evaluator import T4FirstEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--roots", type=int, default=500)
    parser.add_argument("--seed", type=int, default=95_000_000)
    parser.add_argument("--draw-sample", type=int, default=300)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--floor",
        action="store_true",
        help="Also label with an independent referee (different draw seeds) "
        "and report the teacher self-disagreement floor, per the regular "
        "track's gap-over-floor methodology.",
    )
    args = parser.parse_args()
    workspace_root = args.workspace_root.resolve(strict=True)

    checkpoint = torch.load(
        args.model_dir / "evaluator_best.pt", map_location="cpu", weights_only=False
    )
    model = T4FirstEvaluator(checkpoint["input_dim"], tuple(checkpoint["hidden"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    mean = checkpoint["input_mean"].numpy().astype(np.float32)
    scale = checkpoint["input_std"].numpy().astype(np.float32)

    scratch = args.model_dir / "regret_scratch"
    scratch.mkdir(exist_ok=True)
    roots = [sample_root(args.seed + index) for index in range(args.roots)]

    def label_roots(id_prefix: str, tag: str) -> dict:
        """One referee pass; the id feeds the draw-sampling seed, so a
        different prefix yields an independent referee."""
        with (scratch / f"in_{tag}.jsonl").open("w", encoding="utf-8") as handle:
            for root in roots:
                handle.write(
                    json.dumps(
                        {
                            "id": id_prefix + str(root["seed"]),
                            "board": {
                                "top": root["board"][0],
                                "middle": root["board"][1],
                                "bottom": root["board"][2],
                            },
                            "dead": root["dead"],
                            "draw": root["draw"],
                            "opp_count": root["opp_count"],
                            "draw_sample": args.draw_sample,
                        }
                    )
                    + "\n"
                )
        subprocess.run(
            [
                str(_solver_path(workspace_root)),
                "--input", str(scratch / f"in_{tag}.jsonl"),
                "--output", str(scratch / f"out_{tag}.jsonl"),
                "--fl-ev-config", str(workspace_root / "ai" / "config" / "fl_ev.json"),
                "--t3-vs-fl-library",
                "D:/ofc_data/fl_library_14_v3",                *count_library_args(),

                "--chunk-size", "64",
            ],
            check=True,
        )
        out = {}
        with (scratch / f"out_{tag}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    payload = json.loads(line)
                    out[payload["id"].removeprefix(id_prefix)] = {
                        row["action_key"]: row for row in payload["actions"]
                    }
        return out

    labels = label_roots("", "a")
    labels_b = label_roots("b:", "b") if args.floor else None

    regrets: list[float] = []
    floor_regrets: list[float] = []
    jokers_strata: dict[int, list[float]] = {0: [], 1: [], 2: []}
    best_picked = 0
    for root in roots:
        table = labels[str(root["seed"])]
        board = Board(
            top=list(root["board"][0]),
            middle=list(root["board"][1]),
            bottom=list(root["board"][2]),
        )
        jokers_visible = sum(
            1
            for card in (
                [c for row in root["board"] for c in row]
                + root["dead"]
                + root["draw"]
            )
            if card in ("X1", "X2")
        )
        features, values, values_b = [], [], []
        for action in get_turn_actions(list(root["draw"]), board):
            key = exact_late.action_key(action)
            if key not in table:
                continue
            row = table[key]
            after = exact_late.apply_action(board, action)
            features.append(
                encode_t3_action(
                    (after.top, after.middle, after.bottom),
                    list(root["dead"]) + [action.discard],
                    root["opp_count"],
                    row["own_rowwise_block"],
                    row["own_joint_block"],
                )
            )
            values.append(row["value"])
            if labels_b is not None:
                values_b.append(labels_b[str(root["seed"])][key]["value"])
        x = (np.asarray(features, dtype=np.float32) - mean) / scale
        with torch.no_grad():
            predicted = model(torch.from_numpy(x)).squeeze(-1).numpy()
        values_arr = np.asarray(values)
        regret = float(values_arr.max() - values_arr[int(predicted.argmax())])
        regrets.append(regret)
        jokers_strata[min(jokers_visible, 2)].append(regret)
        best_picked += int(regret == 0.0)
        if labels_b is not None:
            # Referee B's best pick, charged by referee A: the teacher
            # self-disagreement floor no chooser can beat.
            values_b_arr = np.asarray(values_b)
            floor_regrets.append(
                float(values_arr.max() - values_arr[int(values_b_arr.argmax())])
            )

    regrets_arr = np.asarray(regrets)
    result = {
        "roots": len(regrets),
        "mean_regret": float(regrets_arr.mean()),
        "best_pick_rate": best_picked / len(regrets),
        "regret_p95": float(np.percentile(regrets_arr, 95)),
        "regret_max": float(regrets_arr.max()),
        "per_jokers_mean": {
            str(k): (float(np.mean(v)) if v else None)
            for k, v in jokers_strata.items()
        },
        "gate": "PASS" if regrets_arr.mean() < 0.05 else "FAIL",
        "seed": args.seed,
        "draw_sample": args.draw_sample,
    }
    if floor_regrets:
        floor_arr = np.asarray(floor_regrets)
        result["floor_mean"] = float(floor_arr.mean())
        result["gap_over_floor"] = float(regrets_arr.mean() - floor_arr.mean())
    (args.model_dir / "regret_report.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
