"""Regret gate for the T2-vs-FL evaluator, charged by playout labels.

Same protocol as the T3 gate: the model picks an action from its features,
and is charged the label-EV it gave up against the label-best action, on
fresh roots disjoint from training.  Two independent labeling passes give the
teacher self-disagreement floor, because playout labels are noisier than the
T3 teacher's and a raw regret number without its floor is unreadable (the
lesson that turned the T3 v2 "FAIL" into a pass at gap 0.0036).

Usage:
    python -m ai.tutor.eval_t2_vs_fl_regret \
        --model-dir D:/ofc_data/t2_vs_fl_model_v1 --roots 300 --floor
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
from ai.tutor.generate_t2_vs_fl_teacher import encode_t2_action
from ai.tutor.generate_t4_first_teacher import _solver_path
from ai.tutor.t2_policy_label_experiment import sample_t2_root
from ai.tutor.train_t4_first_evaluator import T4FirstEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--roots", type=int, default=300)
    parser.add_argument("--seed", type=int, default=99_000_000)
    parser.add_argument("--t3-samples", type=int, default=50)
    parser.add_argument("--t4-draw-sample", type=int, default=100)
    parser.add_argument(
        "--t3-model", type=Path,
        default=Path("D:/ofc_data/t3_vs_fl_model_v1/evaluator.bin"),
    )
    parser.add_argument("--floor", action="store_true")
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
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
    roots = [sample_t2_root(args.seed + index) for index in range(args.roots)]

    def label_roots(id_prefix: str, tag: str) -> dict:
        """One labeling pass.  The request id seeds the playout draws, so a
        different prefix yields an independent referee."""
        in_path = scratch / f"in_{tag}.jsonl"
        out_path = scratch / f"out_{tag}.jsonl"
        with in_path.open("w", encoding="utf-8") as handle:
            for root in roots:
                payload = dict(root)
                payload["id"] = id_prefix + root["id"]
                payload["t3_samples"] = args.t3_samples
                payload["t4_draw_sample"] = args.t4_draw_sample
                handle.write(json.dumps(payload) + "\n")
        subprocess.run(
            [
                str(_solver_path(workspace_root)),
                "--input", str(in_path),
                "--output", str(out_path),
                "--fl-ev-config", str(workspace_root / "ai" / "config" / "fl_ev.json"),
                "--t2-vs-fl-library", "D:/ofc_data/fl_library_14_v3",
                "--t2-t3-model", str(args.t3_model),
                "--chunk-size", "16",
            ],
            check=True,
        )
        out = {}
        with out_path.open(encoding="utf-8") as handle:
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
        table = labels[root["id"]]
        board = Board(
            top=list(root["board"]["top"]),
            middle=list(root["board"]["middle"]),
            bottom=list(root["board"]["bottom"]),
        )
        jokers_visible = sum(
            1
            for card in (
                root["board"]["top"] + root["board"]["middle"]
                + root["board"]["bottom"] + root["dead"] + root["draw"]
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
                encode_t2_action(
                    (after.top, after.middle, after.bottom),
                    list(root["dead"]) + [action.discard],
                    root["opp_count"],
                    row["own_rowwise_block"],
                )
            )
            values.append(row["value"])
            if labels_b is not None:
                values_b.append(labels_b[root["id"]][key]["value"])
        x = (np.asarray(features, dtype=np.float32) - mean) / scale
        with torch.no_grad():
            predicted = model(torch.from_numpy(x)).squeeze(-1).numpy()
        values_arr = np.asarray(values)
        regret = float(values_arr.max() - values_arr[int(predicted.argmax())])
        regrets.append(regret)
        jokers_strata[min(jokers_visible, 2)].append(regret)
        best_picked += int(regret == 0.0)
        if labels_b is not None:
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
        "seed": args.seed,
        "t3_samples": args.t3_samples,
        "t4_draw_sample": args.t4_draw_sample,
    }
    if floor_regrets:
        floor_arr = np.asarray(floor_regrets)
        result["floor_mean"] = float(floor_arr.mean())
        result["gap_over_floor"] = float(regrets_arr.mean() - floor_arr.mean())
        result["gate"] = "PASS" if result["gap_over_floor"] < 0.05 else "FAIL"
    (args.model_dir / "regret_report.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
