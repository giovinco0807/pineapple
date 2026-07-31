"""Train a pairwise T1 final selector over refined candidates.

This complements ``train_t1_final_selector.py``.  The listwise trainer learns
to pick the teacher best candidate from the whole refined set.  This trainer
focuses on pairwise mistakes, especially cases where the teacher-best action is
already in the sync/refined set but the runtime final selector chose another
candidate.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.tutor.train_t1_final_selector import (  # noqa: E402
    DEFAULT_FEATURES,
    evaluate_policy,
    group_rows,
    row_features,
    split_groups,
)


def _teacher_index(rows: list[dict[str, Any]]) -> int | None:
    for idx, row in enumerate(rows):
        if bool(row.get("is_teacher_best")):
            return idx
    return None


def _runtime_index(rows: list[dict[str, Any]]) -> int | None:
    for idx, row in enumerate(rows):
        if bool(row.get("is_runtime_best")):
            return idx
    return None


def _score(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def group_is_final_miss(rows: list[dict[str, Any]]) -> bool:
    teacher = _teacher_index(rows)
    runtime = _runtime_index(rows)
    return teacher is not None and runtime is not None and teacher != runtime


def filter_groups(
    groups: dict[str, list[dict[str, Any]]],
    *,
    group_filter: str,
) -> dict[str, list[dict[str, Any]]]:
    if group_filter == "all":
        return groups
    if group_filter == "final_miss":
        return {key: rows for key, rows in groups.items() if group_is_final_miss(rows)}
    raise ValueError(f"unknown group filter: {group_filter}")


def _raw_vectors(
    groups: dict[str, list[dict[str, Any]]],
    features: list[str],
) -> tuple[list[list[list[float]]], list[list[dict[str, Any]]], dict[str, float], dict[str, float]]:
    raw: list[list[float]] = []
    vectors_by_group: list[list[list[float]]] = []
    rows_by_group: list[list[dict[str, Any]]] = []
    for rows in groups.values():
        vectors = [[row_features(row).get(name, 0.0) for name in features] for row in rows]
        vectors_by_group.append(vectors)
        rows_by_group.append(rows)
        raw.extend(vectors)

    if not raw:
        raise ValueError("no rows to vectorize")
    matrix = torch.tensor(raw, dtype=torch.float32)
    mean_tensor = matrix.mean(dim=0)
    scale_tensor = matrix.std(dim=0, unbiased=False)
    scale_tensor = torch.where(scale_tensor < 1e-9, torch.ones_like(scale_tensor), scale_tensor)
    means = {name: float(mean_tensor[i].item()) for i, name in enumerate(features)}
    scales = {name: float(scale_tensor[i].item()) for i, name in enumerate(features)}
    return vectors_by_group, rows_by_group, means, scales


def _normalized_tensors(
    vectors_by_group: list[list[list[float]]],
    features: list[str],
    means: dict[str, float],
    scales: dict[str, float],
) -> list[torch.Tensor]:
    mean = torch.tensor([means[name] for name in features], dtype=torch.float32)
    scale = torch.tensor([max(scales[name], 1e-9) for name in features], dtype=torch.float32)
    return [(torch.tensor(vectors, dtype=torch.float32) - mean) / scale for vectors in vectors_by_group]


def pair_indices(rows: list[dict[str, Any]], pair_mode: str) -> list[tuple[int, int, float]]:
    teacher = _teacher_index(rows)
    if teacher is None:
        return []
    best_score = _score(rows[teacher], "teacher_best_score", _score(rows[teacher], "teacher_score", 0.0))
    pairs: list[tuple[int, int, float]] = []

    if pair_mode == "runtime_best":
        runtime = _runtime_index(rows)
        if runtime is not None and runtime != teacher:
            regret = max(0.0, best_score - _score(rows[runtime], "teacher_score", best_score))
            pairs.append((teacher, runtime, regret))
        return pairs

    if pair_mode != "all":
        raise ValueError(f"unknown pair mode: {pair_mode}")

    for idx, row in enumerate(rows):
        if idx == teacher:
            continue
        regret = max(0.0, best_score - _score(row, "teacher_score", best_score))
        pairs.append((teacher, idx, regret))
    return pairs


def train_pairwise(
    groups: dict[str, list[dict[str, Any]]],
    *,
    features: list[str],
    epochs: int,
    lr: float,
    l2: float,
    seed: int,
    pair_mode: str,
    regret_weight: float,
    regret_cap: float,
    margin: float,
) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    vectors_by_group, rows_by_group, means, scales = _raw_vectors(groups, features)
    tensors = _normalized_tensors(vectors_by_group, features, means, scales)
    pair_groups = [pair_indices(rows, pair_mode) for rows in rows_by_group]
    kept = [(tensor, pairs) for tensor, pairs in zip(tensors, pair_groups) if pairs]
    if not kept:
        raise ValueError("no pairwise training pairs found")

    weights = torch.zeros(len(features), dtype=torch.float32, requires_grad=True)
    intercept = torch.zeros((), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([weights, intercept], lr=lr)
    order = list(range(len(kept)))
    last_loss = 0.0
    for _epoch in range(max(1, epochs)):
        random.shuffle(order)
        total = torch.zeros((), dtype=torch.float32)
        pair_count = 0
        optimizer.zero_grad()
        for idx in order:
            tensor, pairs = kept[idx]
            scores = tensor.mv(weights) + intercept
            for pos, neg, regret in pairs:
                weight = 1.0 + float(regret_weight) * (
                    min(float(regret), float(regret_cap)) if float(regret_cap) > 0.0 else float(regret)
                )
                total = total + float(weight) * torch.nn.functional.softplus(
                    float(margin) - (scores[pos] - scores[neg])
                )
                pair_count += 1
        loss = total / max(pair_count, 1) + float(l2) * weights.pow(2).mean()
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().item())

    pair_count_total = sum(len(pairs) for _tensor, pairs in kept)
    return {
        "name": "",
        "kind": "t1_final_linear_selector",
        "training_kind": "pairwise",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "features": features,
        "weights": [float(value) for value in weights.detach().tolist()],
        "intercept": float(intercept.detach().item()),
        "means": means,
        "scales": scales,
        "epochs": int(epochs),
        "lr": float(lr),
        "l2": float(l2),
        "seed": int(seed),
        "pair_mode": pair_mode,
        "regret_weight": float(regret_weight),
        "regret_cap": float(regret_cap),
        "pairwise_margin": float(margin),
        "pair_groups": len(kept),
        "pair_count": int(pair_count_total),
        "final_loss": last_loss,
    }


def load_selector(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a pairwise T1 final selector")
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--eval", nargs="*", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pair-mode", choices=["all", "runtime_best"], default="runtime_best")
    parser.add_argument("--group-filter", choices=["all", "final_miss"], default="final_miss")
    parser.add_argument("--regret-weight", type=float, default=1.0)
    parser.add_argument("--regret-cap", type=float, default=5.0)
    parser.add_argument("--pairwise-margin", type=float, default=1.0)
    parser.add_argument("--min-teacher-margin", type=float, default=0.0)
    parser.add_argument("--heldout-fraction", type=float, default=0.0)
    parser.add_argument("--heldout-seed", type=int, default=1)
    parser.add_argument("--baseline-selector", default="")
    args = parser.parse_args(list(argv) if argv is not None else None)

    features = [part.strip() for part in args.features.split(",") if part.strip()]
    train_paths = [Path(path) for path in args.train]
    all_groups = group_rows(
        train_paths,
        require_teacher_refined=True,
        min_teacher_margin=args.min_teacher_margin,
    )
    all_groups = filter_groups(all_groups, group_filter=args.group_filter)
    train_groups, heldout_groups = split_groups(
        all_groups,
        heldout_fraction=args.heldout_fraction,
        seed=args.heldout_seed,
    )
    if not train_groups:
        raise SystemExit("No pairwise T1 groups found")

    selector = train_pairwise(
        train_groups,
        features=features,
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
        seed=args.seed,
        pair_mode=args.pair_mode,
        regret_weight=args.regret_weight,
        regret_cap=args.regret_cap,
        margin=args.pairwise_margin,
    )
    selector["name"] = Path(args.output).stem
    selector["train_inputs"] = [str(path) for path in train_paths]
    selector["min_teacher_margin"] = float(args.min_teacher_margin)
    selector["group_filter"] = args.group_filter
    selector["heldout_fraction"] = float(args.heldout_fraction)
    selector["heldout_seed"] = int(args.heldout_seed)
    selector["group_counts"] = {
        "train_total_after_filter": len(all_groups),
        "train_fit": len(train_groups),
        "heldout": len(heldout_groups),
    }
    selector["metrics"] = {
        "train_fit": evaluate_policy(train_groups, selector),
    }
    if heldout_groups:
        selector["metrics"]["heldout"] = evaluate_policy(heldout_groups, selector)

    baseline = load_selector(args.baseline_selector)
    if baseline is not None:
        selector["baseline_selector"] = args.baseline_selector
        selector["metrics"]["train_fit_baseline"] = evaluate_policy(train_groups, baseline)
        if heldout_groups:
            selector["metrics"]["heldout_baseline"] = evaluate_policy(heldout_groups, baseline)

    for eval_path_raw in args.eval:
        eval_path = Path(eval_path_raw)
        eval_groups = group_rows(
            [eval_path],
            require_teacher_refined=False,
            min_teacher_margin=args.min_teacher_margin,
        )
        selector["metrics"][str(eval_path)] = evaluate_policy(eval_groups, selector)
        if baseline is not None:
            selector["metrics"][f"{eval_path}:baseline"] = evaluate_policy(eval_groups, baseline)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selector, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(selector, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
