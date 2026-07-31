"""Train a T2 final selector from saved runtime benchmark outputs.

The T2 serving path already has a strong candidate pool.  This script learns
which already-refined candidate should be selected after partial T3 exact
sampling, using rows from ``benchmark_t2_t3_union_runtime.py`` results.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.tutor.benchmark_t2_t3_union_runtime import candidate_teacher_score
from ai.tutor.exact_late import action_key
from ai.tutor.hybrid_t1t2 import _linear_selector_score, t2_final_selector_features
from ai.tutor.train_t1_final_selector import DEFAULT_FEATURES as T1_FINAL_FEATURES

DEFAULT_FEATURES = [
    *T1_FINAL_FEATURES,
    "forced_bust",
    "not_forced_bust",
    "model_score_x_not_forced_bust",
    "refined_score_x_not_forced_bust",
    "refined_minus_model_per_rank",
    "refined_rank_minus_model_rank",
]


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def teacher_scores_from_payload(payload: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for candidate in payload.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        score = candidate_teacher_score(candidate)
        if score is None:
            continue
        action = {
            "placements": candidate.get("placements") or [],
            "discard": candidate.get("discard"),
        }
        scores[action_key(action)] = float(score)
    return scores


def group_runtime_rows(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path_index, path in enumerate(paths):
        for row in iter_jsonl(path):
            payload = row.get("payload") or {}
            result = row.get("result") or {}
            teacher_scores = teacher_scores_from_payload(payload)
            if not teacher_scores:
                continue
            full_best_score = max(teacher_scores.values())
            group_key = f"{path_index}:{int(row.get('source_line', 0))}"
            for candidate in result.get("candidates") or []:
                if candidate.get("refined_score") is None:
                    continue
                action = candidate.get("action")
                if not isinstance(action, dict):
                    continue
                key = action_key(action)
                if key not in teacher_scores:
                    continue
                out = dict(candidate)
                out.update(
                    {
                        "turn": 2,
                        "group_id": group_key,
                        "line": int(row.get("source_line", 0)),
                        "runtime_path": str(path),
                        "teacher_score": float(teacher_scores[key]),
                        "full_teacher_best_score": float(full_best_score),
                        "is_full_teacher_best": teacher_scores[key] >= full_best_score - 1e-9,
                        "teacher_regret": max(0.0, float(full_best_score) - float(teacher_scores[key])),
                        "is_refined": True,
                    }
                )
                groups[group_key].append(out)
    return {key: rows for key, rows in groups.items() if rows}


def split_groups(
    groups: dict[str, list[dict[str, Any]]],
    *,
    dev_ratio: float,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    if dev_ratio <= 0.0 or len(groups) < 2:
        return groups, {}
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    dev_count = min(len(keys) - 1, max(1, int(round(len(keys) * dev_ratio))))
    dev_keys = set(keys[:dev_count])
    return (
        {key: rows for key, rows in groups.items() if key not in dev_keys},
        {key: rows for key, rows in groups.items() if key in dev_keys},
    )


def pool_best_index(rows: list[dict[str, Any]]) -> int:
    return max(range(len(rows)), key=lambda idx: float(rows[idx].get("teacher_score", 0.0) or 0.0))


def vectorize(
    groups: dict[str, list[dict[str, Any]]],
    features: list[str],
    means: dict[str, float] | None = None,
    scales: dict[str, float] | None = None,
) -> tuple[list[torch.Tensor], list[int], list[torch.Tensor], dict[str, float], dict[str, float]]:
    raw_vectors: list[list[float]] = []
    group_vectors: list[list[list[float]]] = []
    labels: list[int] = []
    group_regrets: list[list[float]] = []
    for rows in groups.values():
        pool_best_score = max(float(row.get("teacher_score", 0.0) or 0.0) for row in rows)
        vectors = [[t2_final_selector_features(row).get(name, 0.0) for name in features] for row in rows]
        regrets = [max(0.0, pool_best_score - float(row.get("teacher_score", 0.0) or 0.0)) for row in rows]
        group_vectors.append(vectors)
        group_regrets.append(regrets)
        raw_vectors.extend(vectors)
        labels.append(pool_best_index(rows))
    if means is None or scales is None:
        matrix = torch.tensor(raw_vectors, dtype=torch.float32)
        mean_tensor = matrix.mean(dim=0)
        scale_tensor = matrix.std(dim=0, unbiased=False)
        scale_tensor = torch.where(scale_tensor < 1e-9, torch.ones_like(scale_tensor), scale_tensor)
        means = {name: float(mean_tensor[i].item()) for i, name in enumerate(features)}
        scales = {name: float(scale_tensor[i].item()) for i, name in enumerate(features)}
    mean = torch.tensor([means[name] for name in features], dtype=torch.float32)
    scale = torch.tensor([max(scales[name], 1e-9) for name in features], dtype=torch.float32)
    tensors = [(torch.tensor(vectors, dtype=torch.float32) - mean) / scale for vectors in group_vectors]
    regret_tensors = [torch.tensor(regrets, dtype=torch.float32) for regrets in group_regrets]
    return tensors, labels, regret_tensors, means, scales


def train_selector(
    groups: dict[str, list[dict[str, Any]]],
    *,
    features: list[str],
    epochs: int,
    lr: float,
    l2: float,
    regret_weight: float,
    regret_scale: float,
    seed: int,
) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    tensors, labels, regrets, means, scales = vectorize(groups, features)
    weights = torch.zeros(len(features), dtype=torch.float32, requires_grad=True)
    intercept = torch.zeros((), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([weights, intercept], lr=lr)
    order = list(range(len(tensors)))
    final_loss = 0.0
    for _epoch in range(max(1, epochs)):
        random.shuffle(order)
        optimizer.zero_grad()
        total = torch.zeros((), dtype=torch.float32)
        for idx in order:
            scores = tensors[idx].mv(weights) + intercept
            ce_loss = torch.logsumexp(scores, dim=0) - scores[int(labels[idx])]
            regret_loss = (torch.softmax(scores, dim=0) * regrets[idx]).sum() / max(float(regret_scale), 1e-9)
            total = total + ce_loss + float(regret_weight) * regret_loss
        loss = total / max(len(tensors), 1) + float(l2) * weights.pow(2).mean()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().item())
    return {
        "kind": "t2_final_linear_selector",
        "features": features,
        "weights": [float(value) for value in weights.detach().tolist()],
        "intercept": float(intercept.detach().item()),
        "means": means,
        "scales": scales,
        "epochs": int(epochs),
        "lr": float(lr),
        "l2": float(l2),
        "regret_weight": float(regret_weight),
        "regret_scale": float(regret_scale),
        "seed": int(seed),
        "final_loss": final_loss,
    }


def select_current(row_group: list[dict[str, Any]]) -> dict[str, Any]:
    def key(row: dict[str, Any]) -> tuple[float, float, float, int]:
        refined = float(row.get("refined_score", float("-inf")) or float("-inf"))
        model = float(row.get("model_score", float("-inf")) or float("-inf"))
        weight = float(row.get("t2_selection_model_weight_used", 0.5) or 0.5)
        return (refined + weight * model, refined, model, -int(row.get("model_rank", 999)))

    return max(row_group, key=key)


def select_refined(row_group: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        row_group,
        key=lambda row: (
            float(row.get("refined_score", float("-inf")) or float("-inf")),
            float(row.get("model_score", float("-inf")) or float("-inf")),
            -int(row.get("model_rank", 999)),
        ),
    )


def select_model(row_group: list[dict[str, Any]]) -> dict[str, Any]:
    return min(row_group, key=lambda row: int(row.get("model_rank", 999)))


def select_selector(row_group: list[dict[str, Any]], selector: dict[str, Any]) -> dict[str, Any]:
    return max(
        row_group,
        key=lambda row: (
            _linear_selector_score(selector, t2_final_selector_features(row)),
            float(row.get("refined_score", float("-inf")) or float("-inf")),
            float(row.get("model_score", float("-inf")) or float("-inf")),
            -int(row.get("model_rank", 999)),
        ),
    )


def evaluate_groups(groups: dict[str, list[dict[str, Any]]], selector: dict[str, Any] | None) -> dict[str, Any]:
    policies = {
        "current": lambda rows: select_current(rows),
        "refined": lambda rows: select_refined(rows),
        "model_top1": lambda rows: select_model(rows),
    }
    if selector is not None:
        policies["selector"] = lambda rows: select_selector(rows, selector)
    report: dict[str, Any] = {"groups": len(groups)}
    for name, chooser in policies.items():
        losses = []
        pool_losses = []
        hits = 0
        full_hits = 0
        for rows in groups.values():
            chosen = chooser(rows)
            pool_best = max(float(row.get("teacher_score", 0.0) or 0.0) for row in rows)
            full_best = max(float(row.get("full_teacher_best_score", 0.0) or 0.0) for row in rows)
            score = float(chosen.get("teacher_score", 0.0) or 0.0)
            pool_loss = max(0.0, pool_best - score)
            loss = max(0.0, full_best - score)
            pool_losses.append(pool_loss)
            losses.append(loss)
            hits += int(pool_loss <= 1e-9)
            full_hits += int(loss <= 1e-9)
        report[name] = {
            "hit_rate": hits / max(len(losses), 1),
            "full_hit_rate": full_hits / max(len(losses), 1),
            "ev_loss_mean": statistics.mean(losses) if losses else 0.0,
            "ev_loss_max": max(losses) if losses else 0.0,
            "ev_loss_ge_0_5": sum(loss >= 0.5 for loss in losses),
            "ev_loss_ge_1_0": sum(loss >= 1.0 for loss in losses),
            "pool_ev_loss_mean": statistics.mean(pool_losses) if pool_losses else 0.0,
            "pool_ev_loss_max": max(pool_losses) if pool_losses else 0.0,
        }
    return report


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-results", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--regret-weight", type=float, default=1.0)
    parser.add_argument("--regret-scale", type=float, default=3.0)
    parser.add_argument("--dev-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(list(argv) if argv is not None else None)

    paths = [Path(path) for path in args.runtime_results]
    groups = group_runtime_rows(paths)
    if not groups:
        raise SystemExit("No refined T2 runtime groups found")
    train_groups, dev_groups = split_groups(groups, dev_ratio=float(args.dev_ratio), seed=int(args.seed))
    features = [part.strip() for part in str(args.features).split(",") if part.strip()]
    selector = train_selector(
        train_groups,
        features=features,
        epochs=int(args.epochs),
        lr=float(args.lr),
        l2=float(args.l2),
        regret_weight=float(args.regret_weight),
        regret_scale=float(args.regret_scale),
        seed=int(args.seed),
    )
    output = Path(args.output)
    selector.update(
        {
            "name": output.stem,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "runtime_results": [str(path) for path in paths],
            "train_groups": len(train_groups),
            "dev_groups": len(dev_groups),
            "metrics": {
                "train": evaluate_groups(train_groups, selector),
                "dev": evaluate_groups(dev_groups, selector) if dev_groups else {},
                "all": evaluate_groups(groups, selector),
            },
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selector, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(selector, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
