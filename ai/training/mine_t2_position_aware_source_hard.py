"""Mine T2 source rows that stress the position-aware Top1 policy.

The input eval data is a pseudo-teacher conversion of T2 source rows.  Its
scores come from the T3 model, so this script is only for choosing which rows
to spend exact/cap time on.  The selected source JSONL should be exact-labeled
before it is used as training or final evaluation data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np

from ai.training.evaluate_t2_position_aware_union import (
    dataset_predictions,
    group_positions,
    load_positions,
    load_stack_specs,
    parse_named_path_value,
    top_local_indices,
)
from ai.training.train_t2_selector_feature_ranker import (
    load_selector_scores,
    parse_named_path,
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def compact_action(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if not candidate:
        return None
    action = candidate.get("action") if isinstance(candidate.get("action"), dict) else candidate
    return {
        "placements": action.get("placements") or candidate.get("placements") or [],
        "discard": action.get("discard", candidate.get("discard")),
    }


def local_ev_loss(scores: np.ndarray, candidates: list[int]) -> float:
    if not candidates:
        return 0.0
    best = float(np.max(scores))
    pool_best = max(float(scores[idx]) for idx in candidates)
    return max(0.0, best - pool_best)


def model_for_position(position: str, args: argparse.Namespace) -> str:
    if position == "bb":
        return args.bb_model
    if position == "btn":
        return args.btn_model
    return args.fallback_model


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    training_run = Path(args.training_run)
    model_specs, selector_specs, lgbm_selectors, npy_selectors = load_stack_specs(training_run)
    loaded_base = {spec.name: joblib.load(spec.path) for spec in model_specs}
    loaded_lgbm = {name: joblib.load(path) for name, path in lgbm_selectors}
    final_models = {name: joblib.load(path) for name, path in [parse_named_path_value(v) for v in args.final_model]}

    data_spec = parse_named_path(f"source={args.eval_data}")
    data = load_selector_scores(
        data_spec,
        {spec.name: spec for spec in model_specs},
        selector_specs,
        lgbm_selectors,
        loaded_base,
        loaded_lgbm,
        include_base=True,
        npy_selectors=npy_selectors,
    )
    predictions = dataset_predictions(data, final_models)
    positions = group_positions(load_positions(data.spec.path, len(data.scores)), data.bounds)
    source_rows = list(iter_jsonl(Path(args.source)))
    if len(source_rows) < len(data.bounds):
        raise ValueError(f"source rows {len(source_rows)} < eval groups {len(data.bounds)}")

    rows: list[dict[str, Any]] = []
    for group_id, (start, end) in enumerate(data.bounds):
        position = positions[group_id]
        if args.position != "all" and position != args.position:
            continue
        model_name = model_for_position(position, args)
        pred = predictions[model_name]
        local_scores = np.asarray(data.scores[start:end], dtype=np.float64)
        if len(local_scores) == 0:
            continue
        best_local = int(np.argmax(local_scores))
        top1_local = top_local_indices(pred, start, end, 1)[0]
        top3 = top_local_indices(pred, start, end, 3)
        top5 = top_local_indices(pred, start, end, 5)
        top10 = top_local_indices(pred, start, end, 10)
        source = source_rows[group_id]
        candidates = list(source.get("candidates") or [])
        row = {
            "group_id": int(group_id),
            "source_global_index": int(group_id),
            "source_line": source.get("source_line"),
            "position": position,
            "model": model_name,
            "candidate_count": int(end - start),
            "pseudo_best_local": best_local,
            "pseudo_best_score": float(local_scores[best_local]),
            "model_top1_local": int(top1_local),
            "model_top1_score": float(local_scores[top1_local]),
            "model_top1_ev_loss": float(local_ev_loss(local_scores, [top1_local])),
            "model_top3_ev_loss": float(local_ev_loss(local_scores, top3)),
            "model_top5_ev_loss": float(local_ev_loss(local_scores, top5)),
            "model_top10_ev_loss": float(local_ev_loss(local_scores, top10)),
            "board": source.get("board"),
            "opponent_board": source.get("opponent_board"),
            "dealt": source.get("dealt"),
            "known_discards": source.get("known_discards"),
            "best_action": compact_action(candidates[best_local] if best_local < len(candidates) else None),
            "model_top1_action": compact_action(candidates[top1_local] if top1_local < len(candidates) else None),
        }
        if row["model_top1_ev_loss"] >= args.min_top1_loss:
            rows.append(row)

    rows.sort(
        key=lambda item: (
            float(item["model_top1_ev_loss"]),
            float(item["model_top3_ev_loss"]),
            float(item["model_top5_ev_loss"]),
        ),
        reverse=True,
    )
    selected = rows[: args.limit]
    details_path = out_dir / "position_aware_top1_pseudo_losses.details.jsonl"
    with details_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    selected_path = out_dir / f"selected_{args.position}_top{len(selected)}_for_exact.jsonl"
    with selected_path.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(source_rows[int(row["source_global_index"])], ensure_ascii=False, separators=(",", ":")) + "\n")

    losses = [float(row["model_top1_ev_loss"]) for row in rows]
    selected_losses = [float(row["model_top1_ev_loss"]) for row in selected]
    summary = {
        "source": str(Path(args.source)),
        "eval_data": str(Path(args.eval_data)),
        "training_run": str(training_run),
        "position": args.position,
        "bb_model": args.bb_model,
        "btn_model": args.btn_model,
        "fallback_model": args.fallback_model,
        "groups_scanned": int(sum(1 for p in positions if args.position == "all" or p == args.position)),
        "positive_rows": int(len(rows)),
        "selected_count": int(len(selected)),
        "selected_indices": [int(row["source_global_index"]) for row in selected],
        "selected_top1_losses": selected_losses,
        "positive_top1_loss_mean": float(np.mean(losses)) if losses else 0.0,
        "positive_top1_loss_max": float(np.max(losses)) if losses else 0.0,
        "details": str(details_path),
        "selected_source": str(selected_path),
        "note": "Pseudo losses use T3-model source scores; exact-label selected_source before final evaluation.",
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--eval-data", required=True)
    parser.add_argument("--training-run", required=True)
    parser.add_argument("--final-model", action="append", required=True, help="name=path")
    parser.add_argument("--bb-model", default="hgb_cls_l31_state")
    parser.add_argument("--btn-model", default="extra_trees_cls_d12_state")
    parser.add_argument("--fallback-model", default="extra_trees_cls_d12_state")
    parser.add_argument("--position", choices=["bb", "btn", "all"], default="all")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--min-top1-loss", type=float, default=0.0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
