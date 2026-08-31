"""Evaluate a T2 reranker with a narrow record-shape model switch.

This is a diagnostic tool.  It keeps the normal model for most T2 spots and
uses a specialist model only when a runtime-available board/dealt shape gate is
active.  Teacher EV is used only for evaluation.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.training.evaluate_t2_runtime_override import apply_override, evaluate, load_eval_sets
from ai.training.train_t2_selector_feature_pool_reranker import predict_dataset


def card_rank(card: str) -> str:
    return str(card or "")[:1]


def row_cards(record: dict, row: str) -> list[str]:
    return list((record.get("board") or {}).get(row) or [])


def rank_counts(cards: Iterable[str]) -> Counter:
    return Counter(card_rank(card) for card in cards)


def has_pair(cards: Iterable[str], rank: str) -> bool:
    counts = rank_counts(cards)
    return int(counts.get(rank, 0)) >= 2


def has_top_aa(record: dict) -> bool:
    counts = rank_counts(row_cards(record, "top"))
    return int(counts.get("A", 0)) >= 2


def has_middle_jj(record: dict) -> bool:
    return has_pair(row_cards(record, "middle"), "J")


def dealt_has_aj(record: dict) -> bool:
    counts = rank_counts(record.get("dealt") or [])
    return int(counts.get("A", 0)) >= 1 and int(counts.get("J", 0)) >= 1


def target_aaj_gate(record: dict, gate: str) -> bool:
    if int(record.get("turn", -1)) != 2:
        return False
    if not (has_top_aa(record) and has_middle_jj(record) and dealt_has_aj(record)):
        return False
    if gate == "top_aa_mid_jj_dealt_aj":
        return True
    if gate == "top_aa_mid_jj_dealt_aj_bottom_le2":
        return len(row_cards(record, "bottom")) <= 2
    if gate == "top_aa_mid_jj_dealt_aj_bottom_le1":
        return len(row_cards(record, "bottom")) <= 1
    raise ValueError(f"unknown gate: {gate}")


def load_source_records(data_dir: Path) -> list[dict]:
    metadata_path = data_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source = Path(metadata["source"])
    records: list[dict] = []
    with source.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if int(record.get("turn", -1)) == 2 and record.get("candidates"):
                records.append(record)
    return records


def switch_predictions(ds, base_pred: np.ndarray, switch_pred: np.ndarray, records: list[dict], gate: str) -> tuple[np.ndarray, list[dict]]:
    if len(records) != len(ds.bounds):
        raise ValueError(
            f"{ds.spec.name}: record/bounds mismatch: {len(records)} records != {len(ds.bounds)} groups"
        )
    out = np.array(base_pred, copy=True)
    rows: list[dict] = []
    for group_id, ((start, end), record) in enumerate(zip(ds.bounds, records)):
        if not target_aaj_gate(record, gate):
            continue
        out[start:end] = switch_pred[start:end]
        base_best = start + int(np.argmax(base_pred[start:end]))
        switch_best = start + int(np.argmax(switch_pred[start:end]))
        teacher_best = start + int(np.argmax(ds.scores[start:end]))
        rows.append(
            {
                "dataset": ds.spec.name,
                "group_id": int(group_id),
                "base_index": int(base_best),
                "switch_index": int(switch_best),
                "teacher_best_index": int(teacher_best),
                "base_ev_loss": max(0.0, float(ds.scores[teacher_best] - ds.scores[base_best])),
                "switch_ev_loss": max(0.0, float(ds.scores[teacher_best] - ds.scores[switch_best])),
                "position": record.get("position"),
                "board": record.get("board"),
                "opponent_board": record.get("opponent_board"),
                "dealt": record.get("dealt"),
            }
        )
    return out, rows


def run(args: argparse.Namespace) -> None:
    config = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    datasets = load_eval_sets(args, config)
    base_model = joblib.load(args.base_model_path)
    switch_model = joblib.load(args.switch_model_path)

    base_pred = {}
    specialist_pred = {}
    switched_pred = {}
    switch_rows: list[dict] = []
    override_selector_names = []
    for value in args.override_selector or []:
        override_selector_names.extend(name.strip() for name in str(value).split(",") if name.strip())
    for ds in datasets:
        base = predict_dataset(
            ds,
            base_model,
            args.model_kind,
            int(config["pool_k"]),
            config["feature_scope"],
            bool(config["pool_local_features"]),
        )
        specialist = predict_dataset(
            ds,
            switch_model,
            args.model_kind,
            int(config["pool_k"]),
            config["feature_scope"],
            bool(config["pool_local_features"]),
        )
        for selector_name in override_selector_names:
            base, _rows = apply_override(ds, base, selector_name)
            specialist, _rows = apply_override(ds, specialist, selector_name)
        records = load_source_records(Path(ds.spec.path))
        switched, rows = switch_predictions(ds, base, specialist, records, args.gate)
        base_pred[ds.spec.name] = base
        specialist_pred[ds.spec.name] = specialist
        switched_pred[ds.spec.name] = switched
        switch_rows.extend(rows)

    base_aggregate, base_per_dataset = evaluate(datasets, base_pred, args.topks, args.ev_hit_epsilon)
    specialist_aggregate, specialist_per_dataset = evaluate(
        datasets, specialist_pred, args.topks, args.ev_hit_epsilon
    )
    switched_aggregate, switched_per_dataset = evaluate(
        datasets, switched_pred, args.topks, args.ev_hit_epsilon
    )
    deltas = [float(row["switch_ev_loss"] - row["base_ev_loss"]) for row in switch_rows]
    output = {
        "source_summary": str(Path(args.summary)),
        "base_model_path": str(Path(args.base_model_path)),
        "switch_model_path": str(Path(args.switch_model_path)),
        "model_kind": args.model_kind,
        "gate": args.gate,
        "eval_data": args.eval_data,
        "npy_selectors": args.npy_selector,
        "override_selector": override_selector_names,
        "base": {"aggregate": base_aggregate, "datasets": base_per_dataset},
        "specialist_global": {"aggregate": specialist_aggregate, "datasets": specialist_per_dataset},
        "switched": {"aggregate": switched_aggregate, "datasets": switched_per_dataset},
        "switch_rows": switch_rows,
        "switch_active_groups": int(len(switch_rows)),
        "switch_better_groups": int(sum(1 for row in switch_rows if row["switch_ev_loss"] < row["base_ev_loss"])),
        "switch_worse_groups": int(sum(1 for row in switch_rows if row["switch_ev_loss"] > row["base_ev_loss"])),
        "switch_delta_ev_loss_mean": float(np.mean(deltas)) if deltas else 0.0,
        "switch_delta_ev_loss_max": float(np.max(deltas)) if deltas else 0.0,
        "switch_delta_ev_loss_min": float(np.min(deltas)) if deltas else 0.0,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(out_path),
                "base": base_aggregate,
                "specialist_global": specialist_aggregate,
                "switched": switched_aggregate,
                "switch_active_groups": output["switch_active_groups"],
                "switch_better_groups": output["switch_better_groups"],
                "switch_worse_groups": output["switch_worse_groups"],
            },
            indent=2,
        )
    )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--eval-data", action="append", required=True, help="name=path")
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--switch-model-path", required=True)
    parser.add_argument("--model-kind", default="hgb_cls_l63")
    parser.add_argument("--selector", action="append", default=[])
    parser.add_argument("--npy-selector", action="append", default=["t2_top_aa_pairdraw_tactical"])
    parser.add_argument(
        "--override-selector",
        action="append",
        default=None,
        help="Runtime override selector name. May be repeated or comma-separated.",
    )
    parser.add_argument(
        "--gate",
        choices=[
            "top_aa_mid_jj_dealt_aj",
            "top_aa_mid_jj_dealt_aj_bottom_le2",
            "top_aa_mid_jj_dealt_aj_bottom_le1",
        ],
        default="top_aa_mid_jj_dealt_aj",
    )
    parser.add_argument("--topks", type=int, nargs="*", default=[1, 3, 5, 10, 15, 20])
    parser.add_argument("--ev-hit-epsilon", type=float, default=1.0e-9)
    parser.add_argument("--no-base", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    run(args)


if __name__ == "__main__":
    main()
