"""Train a T1 sync-candidate selector from candidate-level rows.

The selector is used before recursive T1 refinement.  It scores the Pool20
candidates with features that are already available at runtime, then the top
K candidates are sent to the 5-second refinement stage.
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


DEFAULT_FEATURES = [
    "model_score",
    "model_raw_score",
    "neg_model_rank",
    "inv_model_rank",
    "rank_le_1",
    "rank_le_3",
    "rank_le_6",
    "rank_le_10",
    "rank_le_15",
    "neg_refinement_rank",
    "inv_refinement_rank",
    "refinement_rank_le_6",
    "refinement_rank_le_10",
    "predicted_bust",
    "predicted_safe",
    "predicted_fl",
    "candidate_fl_score",
    "predicted_qq",
    "predicted_kk",
    "predicted_aa",
    "predicted_trips",
    "premium_fl_sum",
    "insurance_low_bust",
    "insurance_fl",
    "insurance_aux",
    "has_aux_score",
    "aux_model_score",
    "neg_aux_model_rank",
    "aux_rank_le_10",
    "model_score_x_inv_rank",
    "fl_score_x_safe",
]


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def group_rows(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path_index, path in enumerate(paths):
        for row in iter_jsonl(path):
            if int(row.get("turn", -1)) != 1:
                continue
            grouped[f"{path_index}:{int(row.get('line', 0))}"].append(row)
    return {key: rows for key, rows in grouped.items() if any(bool(row.get("is_teacher_best")) for row in rows)}


def _rank(row: dict[str, Any], key: str, default: int = 999) -> float:
    try:
        return max(float(row.get(key, default) or default), 1.0)
    except (TypeError, ValueError):
        return float(default)


def row_features(row: dict[str, Any]) -> dict[str, float]:
    model_rank = _rank(row, "model_rank")
    refinement_rank = _rank(row, "refinement_rank")
    aux_rank = _rank(row, "aux_model_rank")
    model_score = float(row.get("model_score", 0.0) or 0.0)
    model_raw_score = float(row.get("model_raw_score", 0.0) or 0.0)
    predicted_bust = float(row.get("predicted_bust", 0.0) or 0.0)
    predicted_fl = float(row.get("predicted_fl", 0.0) or 0.0)
    predicted_qq = float(row.get("predicted_qq", 0.0) or 0.0)
    predicted_kk = float(row.get("predicted_kk", 0.0) or 0.0)
    predicted_aa = float(row.get("predicted_aa", 0.0) or 0.0)
    predicted_trips = float(row.get("predicted_trips", 0.0) or 0.0)
    fl_score = max(predicted_fl, predicted_qq, predicted_kk, predicted_aa, predicted_trips)
    insurance_reason = str(row.get("insurance_reason") or "")
    has_aux = 1.0 if row.get("aux_model_score") is not None else 0.0
    aux_model_score = float(row.get("aux_model_score", 0.0) or 0.0)
    return {
        "model_score": model_score,
        "model_raw_score": model_raw_score,
        "neg_model_rank": -model_rank,
        "inv_model_rank": 1.0 / model_rank,
        "rank_le_1": 1.0 if model_rank <= 1.0 else 0.0,
        "rank_le_3": 1.0 if model_rank <= 3.0 else 0.0,
        "rank_le_6": 1.0 if model_rank <= 6.0 else 0.0,
        "rank_le_10": 1.0 if model_rank <= 10.0 else 0.0,
        "rank_le_15": 1.0 if model_rank <= 15.0 else 0.0,
        "neg_refinement_rank": -refinement_rank,
        "inv_refinement_rank": 1.0 / refinement_rank,
        "refinement_rank_le_6": 1.0 if refinement_rank <= 6.0 else 0.0,
        "refinement_rank_le_10": 1.0 if refinement_rank <= 10.0 else 0.0,
        "predicted_bust": predicted_bust,
        "predicted_safe": 1.0 - predicted_bust,
        "predicted_fl": predicted_fl,
        "candidate_fl_score": fl_score,
        "predicted_qq": predicted_qq,
        "predicted_kk": predicted_kk,
        "predicted_aa": predicted_aa,
        "predicted_trips": predicted_trips,
        "premium_fl_sum": predicted_kk + predicted_aa + predicted_trips,
        "insurance_low_bust": 1.0 if insurance_reason == "low_bust" else 0.0,
        "insurance_fl": 1.0 if insurance_reason == "fl" else 0.0,
        "insurance_aux": 1.0 if insurance_reason == "aux_shortlist" else 0.0,
        "has_aux_score": has_aux,
        "aux_model_score": aux_model_score,
        "neg_aux_model_rank": -aux_rank if has_aux else 0.0,
        "aux_rank_le_10": 1.0 if has_aux and aux_rank <= 10.0 else 0.0,
        "model_score_x_inv_rank": model_score / model_rank,
        "fl_score_x_safe": fl_score * (1.0 - predicted_bust),
    }


def vectorize(
    groups: dict[str, list[dict[str, Any]]],
    features: list[str],
    means: dict[str, float] | None = None,
    scales: dict[str, float] | None = None,
) -> tuple[list[torch.Tensor], list[int], dict[str, float], dict[str, float]]:
    raw_vectors: list[list[float]] = []
    labels: list[int] = []
    group_vectors: list[list[list[float]]] = []
    for rows in groups.values():
        vectors = [[row_features(row).get(name, 0.0) for name in features] for row in rows]
        label = next((idx for idx, row in enumerate(rows) if bool(row.get("is_teacher_best"))), 0)
        group_vectors.append(vectors)
        raw_vectors.extend(vectors)
        labels.append(label)

    if means is None or scales is None:
        matrix = torch.tensor(raw_vectors, dtype=torch.float32)
        mean_tensor = matrix.mean(dim=0)
        scale_tensor = matrix.std(dim=0, unbiased=False)
        scale_tensor = torch.where(scale_tensor < 1e-9, torch.ones_like(scale_tensor), scale_tensor)
        means = {name: float(mean_tensor[i].item()) for i, name in enumerate(features)}
        scales = {name: float(scale_tensor[i].item()) for i, name in enumerate(features)}

    tensors: list[torch.Tensor] = []
    mean = torch.tensor([means[name] for name in features], dtype=torch.float32)
    scale = torch.tensor([max(scales[name], 1e-9) for name in features], dtype=torch.float32)
    for vectors in group_vectors:
        tensor = torch.tensor(vectors, dtype=torch.float32)
        tensors.append((tensor - mean) / scale)
    return tensors, labels, means, scales


def train_selector(
    groups: dict[str, list[dict[str, Any]]],
    *,
    features: list[str],
    epochs: int,
    lr: float,
    l2: float,
    seed: int,
) -> tuple[torch.Tensor, float, dict[str, float], dict[str, float], list[float]]:
    random.seed(seed)
    torch.manual_seed(seed)
    tensors, labels, means, scales = vectorize(groups, features)
    order = list(range(len(tensors)))
    weights = torch.zeros(len(features), dtype=torch.float32, requires_grad=True)
    intercept = torch.zeros((), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([weights, intercept], lr=lr)
    losses: list[float] = []
    for _epoch in range(max(1, epochs)):
        random.shuffle(order)
        total = torch.zeros((), dtype=torch.float32)
        optimizer.zero_grad()
        for idx in order:
            scores = tensors[idx].mv(weights) + intercept
            total = total + torch.logsumexp(scores, dim=0) - scores[int(labels[idx])]
        loss = total / max(len(tensors), 1) + float(l2) * weights.pow(2).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
    return weights.detach(), float(intercept.detach().item()), means, scales, losses


def selector_score(row: dict[str, Any], model: dict[str, Any]) -> float:
    features = row_features(row)
    total = float(model.get("intercept", 0.0))
    for name, weight in zip(model["features"], model["weights"]):
        raw = float(features.get(name, 0.0))
        raw = (raw - float(model["means"].get(name, 0.0))) / max(float(model["scales"].get(name, 1.0)), 1e-9)
        total += float(weight) * raw
    return total


def selected_rows(rows: list[dict[str, Any]], ranked_rows: list[dict[str, Any]], k: int, high_bust: float) -> list[dict[str, Any]]:
    if any(float(row.get("predicted_bust", 0.0) or 0.0) < high_bust for row in ranked_rows):
        ranked_rows = [row for row in ranked_rows if float(row.get("predicted_bust", 0.0) or 0.0) < high_bust]
    if not ranked_rows:
        ranked_rows = list(rows)
    return ranked_rows[:k]


def recall_report(
    groups: dict[str, list[dict[str, Any]]],
    *,
    model: dict[str, Any] | None,
    sync_ks: list[int],
    high_bust: float,
) -> dict[str, Any]:
    report: dict[str, Any] = {"decisions": len(groups)}
    for name in ("model_rank", "selector"):
        if name == "selector" and model is None:
            continue
        ranked_groups: dict[str, list[dict[str, Any]]] = {}
        for key, rows in groups.items():
            if name == "model_rank":
                ranked = sorted(rows, key=lambda row: (int(row.get("refinement_rank") or row.get("model_rank") or 999), int(row.get("model_rank") or 999)))
            else:
                ranked = sorted(
                    rows,
                    key=lambda row: (selector_score(row, model or {}), -int(row.get("model_rank") or 999)),
                    reverse=True,
                )
            ranked_groups[key] = ranked
        metrics: dict[str, Any] = {}
        for k in sync_ks:
            hits = 0
            for key, ranked in ranked_groups.items():
                chosen = selected_rows(groups[key], ranked, k, high_bust)
                if any(bool(row.get("is_teacher_best")) for row in chosen):
                    hits += 1
            metrics[f"recall_at_{k}"] = hits / max(len(groups), 1)
            metrics[f"hits_at_{k}"] = hits
        report[name] = metrics
    return report


def parse_ks(raw: str) -> list[int]:
    return [int(part) for part in raw.split(",") if part.strip()]


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a T1 sync-candidate selector")
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--eval", nargs="*", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sync-ks", default="3,6,8,10")
    parser.add_argument("--high-bust-threshold", type=float, default=0.999)
    args = parser.parse_args(list(argv) if argv is not None else None)

    features = [part.strip() for part in args.features.split(",") if part.strip()]
    train_paths = [Path(path) for path in args.train]
    eval_paths = [Path(path) for path in args.eval]
    train_groups = group_rows(train_paths)
    if not train_groups:
        raise SystemExit("No T1 teacher-best groups found in training rows")

    weights, intercept, means, scales, losses = train_selector(
        train_groups,
        features=features,
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
        seed=args.seed,
    )
    selector = {
        "name": Path(args.output).stem,
        "kind": "t1_sync_linear_selector",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "features": features,
        "weights": [float(value) for value in weights.tolist()],
        "intercept": intercept,
        "means": means,
        "scales": scales,
        "train_inputs": [str(path) for path in train_paths],
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "l2": float(args.l2),
        "seed": int(args.seed),
        "final_loss": losses[-1] if losses else None,
        "metrics": {
            "train": recall_report(
                train_groups,
                model=None,
                sync_ks=parse_ks(args.sync_ks),
                high_bust=float(args.high_bust_threshold),
            )
        },
    }
    selector["metrics"]["train"].update(
        recall_report(
            train_groups,
            model=selector,
            sync_ks=parse_ks(args.sync_ks),
            high_bust=float(args.high_bust_threshold),
        )
    )

    for eval_path in eval_paths:
        eval_groups = group_rows([eval_path])
        selector["metrics"][str(eval_path)] = recall_report(
            eval_groups,
            model=None,
            sync_ks=parse_ks(args.sync_ks),
            high_bust=float(args.high_bust_threshold),
        )
        selector["metrics"][str(eval_path)].update(
            recall_report(
                eval_groups,
                model=selector,
                sync_ks=parse_ks(args.sync_ks),
                high_bust=float(args.high_bust_threshold),
            )
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selector, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(selector, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
