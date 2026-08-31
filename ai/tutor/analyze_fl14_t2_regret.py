"""Group a fixed T2 model's regret and write its worst decisions.

Model selection must already be finished.  This opens one encoded split and
uses the original labels only to name the model and teacher actions.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ai.tutor.encode_fl14_teacher import stable_root_id
from ai.tutor.train_t4_first_evaluator import T4FirstEvaluator, metrics


def row_count(text: str) -> int:
    return 0 if not text else len(text.split(","))


def action_parts(key: str) -> tuple[list[str], str]:
    *rows, discard = key.split("|")
    return rows, discard


def rank_band(card: str) -> str:
    if card.startswith("X"):
        return "joker"
    rank = card[0]
    if rank in "23456":
        return "low_2_6"
    if rank in "789T":
        return "mid_7_T"
    return "high_J_A"


def card_rank(card: str) -> str:
    return "X" if card.startswith("X") else card[0]


def matching_rows(card: str, board_rows: list[str]) -> list[int]:
    """Root rows holding the same natural rank as this card."""
    rank = card_rank(card)
    if rank == "X":
        return []
    out = []
    for index, text in enumerate(board_rows):
        cards = [] if not text else text.split(",")
        if any(card_rank(held) == rank for held in cards):
            out.append(index)
    return out


def destination(card: str, rows: list[str]) -> str:
    for index, text in enumerate(rows):
        if card in ([] if not text else text.split(",")):
            return ("top", "middle", "bottom")[index]
    return "discard"


def summary(rows: list[dict], key) -> list[dict]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[str(key(row))].append(row["regret"])
    total = sum(sum(values) for values in groups.values())
    out = []
    for name, values in groups.items():
        array = np.asarray(values, dtype=np.float64)
        charged = float(array.sum())
        out.append(
            {
                "group": name,
                "roots": len(values),
                "mean_regret": float(array.mean()),
                "p90_regret": float(np.quantile(array, 0.9)),
                "max_regret": float(array.max()),
                "zero_regret_rate": float(np.mean(array == 0)),
                "total_regret": charged,
                "regret_share": charged / total if total else 0.0,
            }
        )
    return sorted(out, key=lambda row: (-row["total_regret"], row["group"]))


def load_labels(path: Path) -> dict[int, dict]:
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            root = stable_root_id(record)
            if root in records:
                raise RuntimeError(f"duplicate decision id {record.get('id')}")
            records[root] = record
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--worst", type=int, default=50)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    parser.add_argument(
        "--roots-out",
        type=Path,
        help="Optional played-root JSONL containing the reported worst roots.",
    )
    args = parser.parse_args()

    with np.load(args.data_dir / f"{args.split}.npz") as payload:
        x, y, roots = payload["x"], payload["y"], payload["roots"]
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    model = T4FirstEvaluator(checkpoint["input_dim"], tuple(checkpoint["hidden"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        truth = torch.tensor(y)
        prediction_tensor = model(
            (torch.tensor(x) - checkpoint["input_mean"]) / checkpoint["input_std"]
        )
        prediction = prediction_tensor.numpy()
        point_metrics = metrics(prediction_tensor, truth)

    indices_by_root: dict[int, list[int]] = defaultdict(list)
    for index, root in enumerate(roots):
        indices_by_root[int(root)].append(index)
    labels = load_labels(args.labels)
    rows = []
    for root, indices in indices_by_root.items():
        record = labels[root]
        if len(indices) != len(record["actions"]):
            raise RuntimeError(
                f"root {root}: {len(indices)} encoded rows vs "
                f"{len(record['actions'])} actions"
            )
        values = y[indices]
        predicted = prediction[indices]
        best = int(np.argmax(values))
        pick = int(np.argmax(predicted))
        teacher = record["actions"][best]
        chosen = record["actions"][pick]
        teacher_rows, teacher_discard = action_parts(teacher["action_key"])
        model_rows, model_discard = action_parts(chosen["action_key"])
        board_rows = record["board"].split("|")[:3]
        visible = record["board"] + "|" + record["dead"] + "|" + record["draw"]
        if pick == best:
            disagreement = "same_action"
        elif teacher_discard == model_discard:
            disagreement = "same_discard_different_rows"
        else:
            disagreement = "different_discard"
        teacher_top = row_count(teacher_rows[0])
        model_top = row_count(model_rows[0])
        top_bias = (
            "teacher_more_top" if teacher_top > model_top
            else "model_more_top" if teacher_top < model_top
            else "same_top_count"
        )
        top_composition = (
            "same_top_cards"
            if sorted(teacher_rows[0].split(",") if teacher_rows[0] else [])
            == sorted(model_rows[0].split(",") if model_rows[0] else [])
            else "different_top_cards"
        )
        model_discard_matches = matching_rows(model_discard, board_rows)
        teacher_discard_matches = matching_rows(teacher_discard, board_rows)
        rows.append(
            {
                "root": root,
                "record_id": str(record["id"]),
                "regret": float(values[best] - values[pick]),
                "shape": "-".join(str(row_count(row)) for row in board_rows),
                "root_top_cards": row_count(board_rows[0]),
                "visible_jokers": visible.count("X"),
                "actions": len(indices),
                "disagreement": disagreement,
                "top_bias": top_bias,
                "top_composition": top_composition,
                "model_discard_pairing": "pairs_root" if model_discard_matches else "no_root_pair",
                "teacher_discard_pairing": "pairs_root" if teacher_discard_matches else "no_root_pair",
                "model_discard_teacher_destination": destination(model_discard, teacher_rows),
                "teacher_discard_model_destination": destination(teacher_discard, model_rows),
                "teacher_discard_band": rank_band(teacher_discard),
                "model_discard_band": rank_band(model_discard),
                "board": record["board"],
                "dead": record["dead"],
                "draw": record["draw"],
                "teacher_action": teacher["action_key"],
                "teacher_value": float(values[best]),
                "teacher_prediction": float(predicted[best]),
                "model_action": chosen["action_key"],
                "model_value": float(values[pick]),
                "model_prediction": float(predicted[pick]),
                "root_spread": float(values.max() - values.min()),
                "model_spread": float(predicted.max() - predicted.min()),
            }
        )
    rows.sort(key=lambda row: row["regret"], reverse=True)
    regrets = np.asarray([row["regret"] for row in rows])
    grouped = {
        "shape": summary(rows, lambda row: row["shape"]),
        "root_top_cards": summary(rows, lambda row: row["root_top_cards"]),
        "visible_jokers": summary(rows, lambda row: row["visible_jokers"]),
        "disagreement": summary(rows, lambda row: row["disagreement"]),
        "top_bias": summary(rows, lambda row: row["top_bias"]),
        "top_composition": summary(rows, lambda row: row["top_composition"]),
        "model_discard_pairing": summary(rows, lambda row: row["model_discard_pairing"]),
        "model_discard_teacher_destination": summary(
            rows, lambda row: row["model_discard_teacher_destination"]
        ),
        "shape_x_top_composition": summary(
            rows, lambda row: f"{row['shape']} / {row['top_composition']}"
        ),
        "model_discard_band": summary(rows, lambda row: row["model_discard_band"]),
    }
    report = {
        "split": args.split,
        "roots": len(rows),
        "rows": len(y),
        "point_metrics": point_metrics,
        "mean_regret": float(regrets.mean()),
        "zero_regret_rate": float(np.mean(regrets == 0)),
        "quantiles": {
            "p50": float(np.quantile(regrets, 0.5)),
            "p90": float(np.quantile(regrets, 0.9)),
            "p95": float(np.quantile(regrets, 0.95)),
            "p99": float(np.quantile(regrets, 0.99)),
            "max": float(regrets.max()),
        },
        "groups": grouped,
        "worst": rows[: args.worst],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.roots_out:
        args.roots_out.parent.mkdir(parents=True, exist_ok=True)
        root_lines = []
        for row in rows[: args.worst]:
            record = labels[row["root"]]
            root_lines.append(
                json.dumps(
                    {
                        "id": str(record["id"]),
                        "rows": [
                            [] if not text else text.split(",")
                            for text in record["board"].split("|")[:3]
                        ],
                        "dead": [] if not record["dead"] else record["dead"].split(","),
                        "draw": record["draw"].split(","),
                    },
                    separators=(",", ":"),
                )
            )
        args.roots_out.write_text("\n".join(root_lines) + "\n", encoding="utf-8")

    md = [
        f"# T2 {args.split} regret analysis",
        "",
        f"- roots: {len(rows):,}",
        f"- mean regret: {regrets.mean():.6f}",
        f"- zero regret: {np.mean(regrets == 0):.2%}",
        f"- p90 / p99 / max: {np.quantile(regrets, .9):.3f} / "
        f"{np.quantile(regrets, .99):.3f} / {regrets.max():.3f}",
    ]
    for name in (
        "shape", "root_top_cards", "visible_jokers", "disagreement", "top_bias",
        "top_composition", "model_discard_pairing", "model_discard_teacher_destination",
        "shape_x_top_composition",
    ):
        md += ["", f"## By {name}", "", "| group | roots | mean | p90 | max | regret share |", "|---|---:|---:|---:|---:|---:|"]
        for row in grouped[name]:
            md.append(
                f"| {row['group']} | {row['roots']} | {row['mean_regret']:.3f} | "
                f"{row['p90_regret']:.3f} | {row['max_regret']:.3f} | "
                f"{row['regret_share']:.1%} |"
            )
    md += ["", f"## Worst {min(args.worst, len(rows))}", ""]
    for index, row in enumerate(rows[: args.worst], 1):
        md += [
            f"### {index}. root {row['record_id']} — regret {row['regret']:.3f}",
            "",
            f"- root: `{row['board']}`; dead `{row['dead']}`; draw `{row['draw']}`",
            f"- teacher: `{row['teacher_action']}` ({row['teacher_value']:+.3f}, "
            f"pred {row['teacher_prediction']:+.3f})",
            f"- model: `{row['model_action']}` ({row['model_value']:+.3f}, "
            f"pred {row['model_prediction']:+.3f})",
            f"- {row['disagreement']}; {row['top_bias']}",
            "",
        ]
    args.md_out.write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
