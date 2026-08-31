"""Evaluate and mine misses for no-leak T2 sklearn meta ensembles."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.training.evaluate_action_value_reranker import parse_topks
from ai.training.evaluate_action_value_set_reranker import summarize_groups
from ai.training.train_t2_sklearn_meta_ranker import compact_metrics, load_runtime_features


def load_jsonl(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_model(value: str) -> tuple[str, Path, str]:
    parts = value.split("=", 1)
    if len(parts) != 2:
        raise ValueError("--model must be name=path,target")
    name = parts[0].strip()
    items = [part.strip() for part in parts[1].split(",")]
    if len(items) != 2:
        raise ValueError("--model must be name=path,target")
    target = items[1].lower()
    if target not in {"residual", "score"}:
        raise ValueError("model target must be residual or score")
    return name, Path(items[0]), target


def parse_selector(value: str) -> tuple[str, list[str]]:
    parts = value.split("=", 1)
    if len(parts) != 2:
        raise ValueError("--selector must be name=model_a+model_b")
    name = parts[0].strip()
    models = [part.strip() for part in parts[1].split("+") if part.strip()]
    if not name or not models:
        raise ValueError("--selector must include a name and at least one model")
    return name, models


def parse_gammas(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part.strip()]


def action_summary(candidate: dict | None) -> dict | None:
    if candidate is None:
        return None
    return {
        "placements": candidate.get("placements"),
        "discard": candidate.get("discard"),
        "ev": candidate.get("ev", candidate.get("score")),
        "bust_prob": candidate.get("bust_prob", candidate.get("bust_rate")),
        "fl_rate": candidate.get("fl_rate"),
        "fl_type_rates": candidate.get("fl_type_rates"),
        "teacher_rank": candidate.get("teacher_rank"),
        "forced_bust": candidate.get("forced_bust"),
    }


def compact_action_rows(
    start: int,
    pred_order: np.ndarray,
    true_order: np.ndarray,
    local_pred: np.ndarray,
    local_scores: np.ndarray,
    candidates: list[dict],
    topn: int,
) -> list[dict]:
    true_rank = np.empty(len(true_order), dtype=np.int32)
    true_rank[true_order] = np.arange(1, len(true_order) + 1, dtype=np.int32)
    rows: list[dict] = []
    for local_idx_raw in pred_order[: min(topn, len(pred_order))]:
        local_idx = int(local_idx_raw)
        candidate = candidates[local_idx] if local_idx < len(candidates) else None
        rows.append(
            {
                "sample_index": int(start + local_idx),
                "candidate_offset": local_idx,
                "pred_score": float(local_pred[local_idx]),
                "teacher_ev": float(local_scores[local_idx]),
                "teacher_rank": int(true_rank[local_idx]),
                "action": action_summary(candidate),
            }
        )
    return rows


def mine_misses(
    scores: np.ndarray,
    pred: np.ndarray,
    bounds: list[tuple[int, int]],
    teacher_rows: list[dict],
    topn: int,
) -> list[dict]:
    misses: list[dict] = []
    for group_pos, (start, end) in enumerate(bounds):
        local_scores = np.asarray(scores[start:end], dtype=np.float64)
        local_pred = np.asarray(pred[start:end], dtype=np.float64)
        true_order = np.argsort(-local_scores)
        pred_order = np.argsort(-local_pred)
        best_local = int(true_order[0])
        chosen_local = int(pred_order[0])
        if chosen_local == best_local:
            continue
        row = teacher_rows[group_pos] if group_pos < len(teacher_rows) else {}
        candidates = list(row.get("candidates") or [])
        best_ev = float(local_scores[best_local])
        chosen_ev = float(local_scores[chosen_local])
        best_pred_rank = int(np.where(pred_order == best_local)[0][0]) + 1
        misses.append(
            {
                "group_id": int(group_pos),
                "group_position": int(group_pos),
                "position": row.get("position"),
                "is_btn": row.get("is_btn"),
                "turn": row.get("turn"),
                "board": row.get("board"),
                "opponent_board": row.get("opponent_board"),
                "dealt": row.get("dealt"),
                "known_discards": row.get("known_discards"),
                "future_hidden_deals": row.get("future_hidden_deals"),
                "candidate_count": int(end - start),
                "best_candidate_offset": best_local,
                "chosen_candidate_offset": chosen_local,
                "best_ev": best_ev,
                "chosen_ev": chosen_ev,
                "regret": float(best_ev - chosen_ev),
                "top1_ev_loss": float(best_ev - chosen_ev),
                "best_pred_rank": best_pred_rank,
                "best_pred_score": float(local_pred[best_local]),
                "chosen_pred_score": float(local_pred[chosen_local]),
                "teacher_best": action_summary(candidates[best_local] if best_local < len(candidates) else None),
                "model_chosen": action_summary(candidates[chosen_local] if chosen_local < len(candidates) else None),
                "top_predictions": compact_action_rows(
                    start,
                    pred_order,
                    true_order,
                    local_pred,
                    local_scores,
                    candidates,
                    topn,
                ),
            }
        )
    misses.sort(key=lambda item: float(item["regret"]), reverse=True)
    return misses


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-data", required=True)
    parser.add_argument("--teacher-jsonl")
    parser.add_argument("--model", action="append", required=True, help="name=path,target")
    parser.add_argument("--selector", action="append", required=True, help="name=model_a+model_b")
    parser.add_argument("--gammas", default="0,0.25,0.5,0.75,0.9,1.0,1.15,1.3")
    parser.add_argument("--topks", default="1,3,5,10,15,20")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--miss-selector")
    parser.add_argument("--miss-gamma", type=float)
    parser.add_argument("--miss-topn", type=int, default=8)
    args = parser.parse_args(list(argv) if argv is not None else None)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = Path(args.eval_data)
    x, scores, base_scores, bounds = load_runtime_features(eval_dir)
    topks = parse_topks(args.topks)
    gammas = parse_gammas(args.gammas)
    model_specs = dict((name, {"path": path, "target": target}) for name, path, target in map(parse_model, args.model))
    selector_specs = dict(parse_selector(value) for value in args.selector)

    predict_started = time.time()
    residuals: dict[str, np.ndarray] = {}
    for name, spec in model_specs.items():
        model = joblib.load(spec["path"])
        pred = model.predict(x).astype(np.float32)
        residuals[name] = pred if spec["target"] == "residual" else pred - base_scores
    prediction_seconds = time.time() - predict_started

    rows: list[dict] = []
    base_metrics, _ = summarize_groups(scores, base_scores, bounds, topks)
    rows.append({"selector": "base_scores", "gamma": 0.0, "metrics": compact_metrics(base_metrics)})
    pred_by_key: dict[tuple[str, float], np.ndarray] = {("base_scores", 0.0): base_scores}
    for selector, names in selector_specs.items():
        missing = [name for name in names if name not in residuals]
        if missing:
            raise ValueError(f"selector {selector} references unknown model(s): {missing}")
        avg_residual = np.mean([residuals[name] for name in names], axis=0).astype(np.float32)
        for gamma in gammas:
            pred = base_scores + float(gamma) * avg_residual
            pred_by_key[(selector, float(gamma))] = pred.astype(np.float32)
            metrics, _ = summarize_groups(scores, pred.astype(np.float32), bounds, topks)
            rows.append({"selector": selector, "gamma": float(gamma), "metrics": compact_metrics(metrics)})
    rows.sort(
        key=lambda row: (
            float(row["metrics"].get("group_top1", 0.0)),
            -float(row["metrics"].get("group_top1_regret", 0.0)),
            float(row["metrics"].get("group_top3", 0.0)),
            -float(row["metrics"].get("group_top10_rerank_regret", 0.0)),
        ),
        reverse=True,
    )

    misses_path = None
    miss_count = 0
    if args.miss_selector is not None and args.miss_gamma is not None:
        key = (args.miss_selector, float(args.miss_gamma))
        if key not in pred_by_key:
            raise ValueError(f"no prediction for miss selector/gamma: {key}")
        teacher_rows = load_jsonl(Path(args.teacher_jsonl) if args.teacher_jsonl else None)
        misses = mine_misses(scores, pred_by_key[key], bounds, teacher_rows, args.miss_topn)
        misses_path = out_dir / f"{args.miss_selector}_g{str(args.miss_gamma).replace('.', 'p')}.misses.jsonl"
        with misses_path.open("w", encoding="utf-8") as f:
            for item in misses:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        miss_count = len(misses)

    summary = {
        "eval_data": str(eval_dir),
        "teacher_jsonl": args.teacher_jsonl,
        "models": {
            name: {"path": str(spec["path"]), "target": spec["target"]}
            for name, spec in model_specs.items()
        },
        "selectors": selector_specs,
        "gammas": gammas,
        "topks": topks,
        "groups": int(len(bounds)),
        "samples": int(len(scores)),
        "prediction_seconds": prediction_seconds,
        "rows": rows,
        "misses_path": str(misses_path) if misses_path else None,
        "miss_count": int(miss_count),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# T2 Sklearn Meta Ensemble Eval",
        "",
        f"- eval_data: `{eval_dir}`",
        f"- groups: `{len(bounds)}`",
        f"- samples: `{len(scores)}`",
        f"- prediction_seconds: `{prediction_seconds:.6f}`",
        "",
        "| selector | gamma | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg3 | Reg10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows[:12]:
        m = row["metrics"]
        lines.append(
            f"| {row['selector']} | {float(row['gamma']):.2f} | "
            f"{float(m.get('group_top1', 0.0)):.1%} | "
            f"{float(m.get('group_top3', 0.0)):.1%} | "
            f"{float(m.get('group_top5', 0.0)):.1%} | "
            f"{float(m.get('group_top10', 0.0)):.1%} | "
            f"{float(m.get('group_top20', 0.0)):.1%} | "
            f"{float(m.get('group_top1_regret', 0.0)):.3f} | "
            f"{float(m.get('group_top3_rerank_regret', 0.0)):.3f} | "
            f"{float(m.get('group_top10_rerank_regret', 0.0)):.3f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(out_dir / "summary.json"), "misses": str(misses_path) if misses_path else None}, indent=2))


if __name__ == "__main__":
    main()
