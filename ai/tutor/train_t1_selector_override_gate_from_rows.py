"""Train/evaluate a T1 override gate from saved selector rows.

This is a cheap replay path for rank-fixed T1 final selectors.  It does not
rerun recursive refinement; it rebuilds the final-selector pick from
``selector_rows.jsonl`` and trains a gate that decides whether that pick should
replace the model Top1.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch

from ai.tutor.hybrid_t1t2 import _linear_gate_probability, _override_gate_features
from ai.tutor.train_t1_final_selector import selector_score
from ai.tutor.train_t1_override_gate import DEFAULT_FEATURES


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def with_fl_types(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["predicted_fl_types"] = {
        "qq": float(out.get("predicted_qq", 0.0) or 0.0),
        "kk": float(out.get("predicted_kk", 0.0) or 0.0),
        "aa": float(out.get("predicted_aa", 0.0) or 0.0),
        "trips": float(out.get("predicted_trips", 0.0) or 0.0),
    }
    return out


def group_rows(paths: list[Path], *, min_teacher_margin: float = 0.0) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path_index, path in enumerate(paths):
        for row in iter_jsonl(path):
            if int(row.get("turn", -1)) != 1:
                continue
            grouped[f"{path_index}:{int(row.get('line', 0))}"].append(with_fl_types(row))

    out: dict[str, list[dict[str, Any]]] = {}
    for key, rows in grouped.items():
        if not rows:
            continue
        if float(rows[0].get("teacher_margin", 0.0) or 0.0) < float(min_teacher_margin):
            continue
        if not any(row.get("teacher_score") is not None for row in rows):
            continue
        out[key] = rows
    return out


def choose_model_top1(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    explicit = [row for row in rows if bool(row.get("is_model_top1"))]
    candidates = explicit or rows
    refined = [row for row in candidates if row.get("refined_score") is not None]
    if not refined:
        return None
    return min(refined, key=lambda row: int(row.get("model_rank", 999999) or 999999))


def choose_selector(rows: list[dict[str, Any]], selector: dict[str, Any]) -> dict[str, Any] | None:
    refined = [row for row in rows if row.get("refined_score") is not None]
    if not refined:
        return None
    return max(
        refined,
        key=lambda row: (
            selector_score(row, selector),
            float(row.get("refined_score", float("-inf")) or float("-inf")),
            float(row.get("model_score", float("-inf")) or float("-inf")),
            -int(row.get("model_rank", 999999) or 999999),
        ),
    )


def build_decision_rows(
    groups: dict[str, list[dict[str, Any]]],
    selector: dict[str, Any],
) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for key, rows in groups.items():
        model_top1 = choose_model_top1(rows)
        selector_pick = choose_selector(rows, selector)
        if model_top1 is None or selector_pick is None:
            continue
        if int(model_top1.get("action_idx", -1)) == int(selector_pick.get("action_idx", -2)):
            continue
        if model_top1.get("teacher_score") is None or selector_pick.get("teacher_score") is None:
            continue
        model_score = float(model_top1.get("teacher_score", 0.0) or 0.0)
        selector_score_value = float(selector_pick.get("teacher_score", 0.0) or 0.0)
        best_score = float(selector_pick.get("teacher_best_score", 0.0) or 0.0)
        features = _override_gate_features(model_top1, selector_pick)
        decisions.append(
            {
                "key": key,
                "line": selector_pick.get("line"),
                "features": features,
                "label": 1 if selector_score_value > model_score else 0,
                "improvement": selector_score_value - model_score,
                "teacher_best_score": best_score,
                "model_teacher_score": model_score,
                "selector_teacher_score": selector_score_value,
                "model_top1_hit": bool(model_top1.get("is_teacher_best")),
                "selector_top1_hit": bool(selector_pick.get("is_teacher_best")),
                "model_action_idx": int(model_top1.get("action_idx", -1)),
                "selector_action_idx": int(selector_pick.get("action_idx", -1)),
                "refined_delta": float(features.get("refined_delta", 0.0) or 0.0),
            }
        )
    return decisions


def vectorize(
    rows: list[dict[str, Any]],
    features: list[str],
    *,
    means: dict[str, float] | None = None,
    scales: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float], dict[str, float]]:
    raw = [[float((row.get("features") or {}).get(name, 0.0)) for name in features] for row in rows]
    labels = torch.tensor([float(row.get("label", 0)) for row in rows], dtype=torch.float32)
    improvements = torch.tensor([abs(float(row.get("improvement", 0.0) or 0.0)) for row in rows], dtype=torch.float32)
    if means is None or scales is None:
        matrix = torch.tensor(raw, dtype=torch.float32)
        mean_tensor = matrix.mean(dim=0)
        scale_tensor = matrix.std(dim=0, unbiased=False)
        scale_tensor = torch.where(scale_tensor < 1e-9, torch.ones_like(scale_tensor), scale_tensor)
        means = {name: float(mean_tensor[i].item()) for i, name in enumerate(features)}
        scales = {name: float(scale_tensor[i].item()) for i, name in enumerate(features)}
    mean = torch.tensor([means[name] for name in features], dtype=torch.float32)
    scale = torch.tensor([max(scales[name], 1e-9) for name in features], dtype=torch.float32)
    x = (torch.tensor(raw, dtype=torch.float32) - mean) / scale
    return x, labels, improvements, means, scales


def train_gate(rows: list[dict[str, Any]], features: list[str], args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    x, labels, improvements, means, scales = vectorize(rows, features)
    weights = torch.zeros(len(features), dtype=torch.float32, requires_grad=True)
    intercept = torch.zeros((), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([weights, intercept], lr=float(args.lr))
    sample_weights = 1.0 + float(args.improvement_weight) * torch.clamp(
        improvements,
        max=max(float(args.improvement_cap), 0.0),
    )
    for _epoch in range(max(1, int(args.epochs))):
        optimizer.zero_grad()
        logits = x.mv(weights) + intercept
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            labels,
            weight=sample_weights,
        )
        loss = loss + float(args.l2) * weights.pow(2).mean()
        loss.backward()
        optimizer.step()
    return {
        "kind": "linear_logistic",
        "name": Path(args.output).stem if args.output else "t1_selector_override_gate_from_rows",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "features": features,
        "weights": [float(value) for value in weights.detach().tolist()],
        "intercept": float(intercept.detach().item()),
        "means": means,
        "scales": scales,
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "l2": float(args.l2),
        "improvement_weight": float(args.improvement_weight),
        "improvement_cap": float(args.improvement_cap),
        "seed": int(args.seed),
    }


def threshold_grid(step: float) -> list[float]:
    step = max(float(step), 0.001)
    count = int(round(1.0 / step))
    return sorted({round(i * step, 10) for i in range(count + 1)})


def accepts(row: dict[str, Any], gate: dict[str, Any], threshold: float, override_margin: float) -> bool:
    features = {name: float(value) for name, value in (row.get("features") or {}).items()}
    if float(features.get("refined_delta", 0.0)) < float(override_margin):
        return False
    return _linear_gate_probability(gate, features) >= float(threshold)


def summarize(rows: list[dict[str, Any]], gate: dict[str, Any] | None, threshold: float, override_margin: float) -> dict[str, Any]:
    out: dict[str, Any] = {
        "decisions": len(rows),
        "threshold": float(threshold),
        "override_margin": float(override_margin),
        "accepted": 0,
        "model_top1": 0,
        "selector_top1": 0,
        "gated_top1": 0,
        "model_regret": 0.0,
        "selector_regret": 0.0,
        "gated_regret": 0.0,
        "model_max_regret": 0.0,
        "selector_max_regret": 0.0,
        "gated_max_regret": 0.0,
        "bad_override_regret": 0.0,
        "rejected_missed_top1": 0,
    }
    for row in rows:
        accepted = True if gate is None else accepts(row, gate, threshold, override_margin)
        out["accepted"] += 1 if accepted else 0
        best_score = float(row.get("teacher_best_score", 0.0) or 0.0)
        model_score = float(row.get("model_teacher_score", 0.0) or 0.0)
        selector_score_value = float(row.get("selector_teacher_score", 0.0) or 0.0)
        selected_score = selector_score_value if accepted else model_score
        model_regret = max(0.0, best_score - model_score)
        selector_regret = max(0.0, best_score - selector_score_value)
        gated_regret = max(0.0, best_score - selected_score)
        out["model_regret"] += model_regret
        out["selector_regret"] += selector_regret
        out["gated_regret"] += gated_regret
        out["model_max_regret"] = max(float(out["model_max_regret"]), model_regret)
        out["selector_max_regret"] = max(float(out["selector_max_regret"]), selector_regret)
        out["gated_max_regret"] = max(float(out["gated_max_regret"]), gated_regret)
        model_hit = bool(row.get("model_top1_hit"))
        selector_hit = bool(row.get("selector_top1_hit"))
        selected_hit = selector_hit if accepted else model_hit
        out["model_top1"] += 1 if model_hit else 0
        out["selector_top1"] += 1 if selector_hit else 0
        out["gated_top1"] += 1 if selected_hit else 0
        if accepted and selector_score_value < model_score:
            out["bad_override_regret"] += model_score - selector_score_value
        if not accepted and selector_hit and not model_hit:
            out["rejected_missed_top1"] += 1
    denom = max(len(rows), 1)
    for prefix in ("model", "selector", "gated"):
        out[f"{prefix}_top1_recall"] = float(out[f"{prefix}_top1"]) / denom
        out[f"{prefix}_avg_regret"] = float(out[f"{prefix}_regret"]) / denom
    return out


def eval_thresholds(rows: list[dict[str, Any]], gate: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    reports = [summarize(rows, gate, threshold, args.override_margin) for threshold in threshold_grid(args.threshold_step)]
    best = min(
        reports,
        key=lambda row: (
            row["gated_avg_regret"],
            row["gated_max_regret"],
            -row["gated_top1"],
            row["bad_override_regret"],
        ),
    )
    best_top1 = max(
        reports,
        key=lambda row: (
            row["gated_top1"],
            -row["gated_avg_regret"],
            -row["gated_max_regret"],
        ),
    )
    return {"best_regret": best, "best_top1": best_top1, "thresholds": reports}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--eval", nargs="*", default=[])
    parser.add_argument("--selector", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows-output", default="")
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--improvement-weight", type=float, default=0.25)
    parser.add_argument("--improvement-cap", type=float, default=8.0)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--override-margin", type=float, default=0.0)
    parser.add_argument("--min-teacher-margin", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(list(argv) if argv is not None else None)

    selector = json.loads(Path(args.selector).read_text(encoding="utf-8"))
    features = [part.strip() for part in args.features.split(",") if part.strip()]
    train_groups = group_rows([Path(path) for path in args.train], min_teacher_margin=args.min_teacher_margin)
    train_rows = build_decision_rows(train_groups, selector)
    if not train_rows:
        raise SystemExit("No selector-vs-model override decision rows found")
    gate = train_gate(train_rows, features, args)
    train_eval = eval_thresholds(train_rows, gate, args)
    report: dict[str, Any] = {
        "gate": gate,
        "train_inputs": args.train,
        "selector": args.selector,
        "train_decision_rows": len(train_rows),
        "train": train_eval,
        "eval": {},
    }
    if args.rows_output:
        rows_path = Path(args.rows_output)
        rows_path.parent.mkdir(parents=True, exist_ok=True)
        with rows_path.open("w", encoding="utf-8") as f:
            for row in train_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    for eval_path in args.eval:
        eval_groups = group_rows([Path(eval_path)], min_teacher_margin=args.min_teacher_margin)
        eval_rows = build_decision_rows(eval_groups, selector)
        report["eval"][eval_path] = {
            "decision_rows": len(eval_rows),
            "always_accept": summarize(eval_rows, None, 0.0, args.override_margin),
            "gate": eval_thresholds(eval_rows, gate, args),
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
