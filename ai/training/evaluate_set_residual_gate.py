"""Evaluate simple confidence gates for a set-residual action-value reranker.

The set reranker can improve some T2 groups while hurting others.  This script
searches one-feature threshold gates on a train dataset, then replays the same
gate on independent eval datasets.  A gate switches the whole candidate group
from the base ensemble scores to:

    base + gamma * (set_residual - base)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.models.action_value_set_reranker import ActionValueSetReranker
from ai.training.evaluate_action_value_reranker import parse_topks
from ai.training.evaluate_action_value_set_reranker import predict_groups, summarize_groups
from ai.training.train_action_value_set_reranker import (
    CandidateGroupDataset,
    build_group_bounds,
    collate_groups,
    load_metadata,
)


@dataclass(frozen=True)
class Gate:
    gate_type: str
    feature: str = ""
    direction: str = ""
    threshold: float = 0.0

    def to_dict(self) -> dict:
        out = {"gate_type": self.gate_type}
        if self.feature:
            out["feature"] = self.feature
            out["direction"] = self.direction
            out["threshold"] = float(self.threshold)
        return out


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.name, path
    name, path = value.split("=", 1)
    return name.strip(), Path(path.strip())


def parse_weights(value: str, n: int) -> list[float]:
    if not value.strip():
        return [1.0 / max(n, 1)] * n
    weights = [float(part) for part in value.split(",") if part.strip()]
    if len(weights) != n:
        raise ValueError(f"--weights has {len(weights)} values but {n} checkpoints were provided")
    total = float(sum(weights))
    if total <= 0.0:
        raise ValueError("--weights must sum to a positive value")
    return [weight / total for weight in weights]


def top_margin(values: np.ndarray, n: int) -> float:
    if len(values) <= 1:
        return 0.0
    ordered = np.sort(np.asarray(values, dtype=np.float64))[::-1]
    idx = min(max(int(n), 2), len(ordered)) - 1
    return float(ordered[0] - ordered[idx])


def rank_of(index: int, ordered: np.ndarray) -> int:
    found = np.where(ordered == index)[0]
    return int(found[0]) + 1 if len(found) else len(ordered) + 1


def feature_rows(
    base: np.ndarray,
    challenger: np.ndarray,
    scores: np.ndarray,
    bounds: list[tuple[int, int]],
    positions: np.ndarray | None = None,
    route_tags: np.ndarray | None = None,
    candidate_ranks: np.ndarray | None = None,
    action_indices: np.ndarray | None = None,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    residual = challenger - base
    for group_id, (start, end) in enumerate(bounds):
        idx = np.arange(start, end, dtype=np.int64)
        true = np.asarray(scores[idx], dtype=np.float64)
        base_g = np.asarray(base[idx], dtype=np.float64)
        chal_g = np.asarray(challenger[idx], dtype=np.float64)
        res_g = np.asarray(residual[idx], dtype=np.float64)
        base_order = np.argsort(-base_g)
        chal_order = np.argsort(-chal_g)
        true_best = int(np.argmax(true))
        base_top = int(base_order[0])
        chal_top = int(chal_order[0])
        base_true = float(true[base_top])
        chal_true = float(true[chal_top])
        true_best_score = float(true[true_best])
        base_rank_of_chal = rank_of(chal_top, base_order)
        chal_rank_of_base = rank_of(base_top, chal_order)
        row = {
            "group_id": float(group_id),
            "size": float(len(idx)),
            "switch": float(base_top != chal_top),
            "base_margin_12": top_margin(base_g, 2),
            "base_margin_13": top_margin(base_g, 3),
            "base_margin_15": top_margin(base_g, 5),
            "challenger_margin_12": top_margin(chal_g, 2),
            "challenger_margin_13": top_margin(chal_g, 3),
            "challenger_margin_15": top_margin(chal_g, 5),
            "base_score_std": float(base_g.std()) if len(base_g) else 0.0,
            "challenger_score_std": float(chal_g.std()) if len(chal_g) else 0.0,
            "residual_std": float(res_g.std()) if len(res_g) else 0.0,
            "residual_max": float(res_g.max()) if len(res_g) else 0.0,
            "residual_min": float(res_g.min()) if len(res_g) else 0.0,
            "residual_range": float(res_g.max() - res_g.min()) if len(res_g) else 0.0,
            "base_top_score": float(base_g[base_top]),
            "challenger_top_score": float(chal_g[chal_top]),
            "base_top_residual": float(res_g[base_top]),
            "challenger_top_residual": float(res_g[chal_top]),
            "residual_advantage_chal_over_base": float(res_g[chal_top] - res_g[base_top]),
            "base_margin_against_chal": float(base_g[base_top] - base_g[chal_top]),
            "challenger_margin_over_base": float(chal_g[chal_top] - chal_g[base_top]),
            "base_rank_of_challenger_top": float(base_rank_of_chal),
            "challenger_rank_of_base_top": float(chal_rank_of_base),
            "base_regret": float(true_best_score - base_true),
            "challenger_regret": float(true_best_score - chal_true),
            "challenger_minus_base_true": float(chal_true - base_true),
            "challenger_improves": float(chal_true > base_true + 1e-9),
            "challenger_hurts": float(chal_true < base_true - 1e-9),
        }
        if positions is not None and len(positions) >= end:
            pos = int(positions[start])
            row["position"] = float(pos)
            row["position_is_btn"] = float(pos == 1)
            row["position_is_bb"] = float(pos == 0)
        if candidate_ranks is not None and len(candidate_ranks) >= end:
            row["base_top_candidate_rank"] = float(candidate_ranks[idx[base_top]])
            row["challenger_top_candidate_rank"] = float(candidate_ranks[idx[chal_top]])
            row["candidate_rank_gap_chal_minus_base"] = (
                row["challenger_top_candidate_rank"] - row["base_top_candidate_rank"]
            )
        if action_indices is not None and len(action_indices) >= end:
            row["base_top_action_index"] = float(action_indices[idx[base_top]])
            row["challenger_top_action_index"] = float(action_indices[idx[chal_top]])
            row["action_index_gap_chal_minus_base"] = (
                row["challenger_top_action_index"] - row["base_top_action_index"]
            )
        if route_tags is not None and len(route_tags) >= end:
            route_g = np.asarray(route_tags[idx], dtype=np.int64)
            base_tag = int(route_g[base_top])
            chal_tag = int(route_g[chal_top])
            row["base_top_route_tag"] = float(base_tag)
            row["challenger_top_route_tag"] = float(chal_tag)
            row["route_tag_gap_chal_minus_base"] = float(chal_tag - base_tag)
            for bit in (1, 4, 8, 16, 256):
                row[f"base_top_route_bit_{bit}"] = float((base_tag & bit) != 0)
                row[f"challenger_top_route_bit_{bit}"] = float((chal_tag & bit) != 0)
                row[f"route_bit_{bit}_fraction"] = float(((route_g & bit) != 0).mean())
        rows.append(row)
    return rows


def load_dataset_predictions(
    name: str,
    data_dir: Path,
    model: ActionValueSetReranker,
    gamma: float,
    device: torch.device,
    batch_size: int,
    topks: list[int],
) -> dict:
    meta = load_metadata(data_dir)
    states = np.load(data_dir / "states.npy", mmap_mode="r")
    n_samples = min(int(meta.get("n_samples", states.shape[0])), int(states.shape[0]))
    scores = np.asarray(np.load(data_dir / "scores.npy", mmap_mode="r")[:n_samples], dtype=np.float32)
    group_ids = np.asarray(np.load(data_dir / "group_ids.npy", mmap_mode="r")[:n_samples], dtype=np.int64)
    bounds = build_group_bounds(group_ids)
    base_path = data_dir / "base_scores.npy"
    if not base_path.exists():
        raise FileNotFoundError(f"Missing base scores: {base_path}")
    base = np.asarray(np.load(base_path, mmap_mode="r")[:n_samples], dtype=np.float32)
    dataset = CandidateGroupDataset(data_dir, list(range(len(bounds))), bounds, model.score_mean, model.score_std)
    if int(dataset.input_dim) != int(model.input_dim):
        raise ValueError(f"{name}: data input_dim={dataset.input_dim} does not match model input_dim={model.input_dim}")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_groups,
    )
    set_scores = predict_groups(model, loader, bounds, n_samples, device)
    challenger = base + float(gamma) * (set_scores - base)
    positions = np.load(data_dir / "positions.npy", mmap_mode="r")[:n_samples] if (data_dir / "positions.npy").exists() else None
    route_tags = np.load(data_dir / "route_tags.npy", mmap_mode="r")[:n_samples] if (data_dir / "route_tags.npy").exists() else None
    candidate_ranks = (
        np.load(data_dir / "candidate_ranks.npy", mmap_mode="r")[:n_samples]
        if (data_dir / "candidate_ranks.npy").exists()
        else None
    )
    action_indices = (
        np.load(data_dir / "action_indices.npy", mmap_mode="r")[:n_samples]
        if (data_dir / "action_indices.npy").exists()
        else None
    )
    base_metrics, _ = summarize_groups(scores, base, bounds, topks)
    challenger_metrics, _ = summarize_groups(scores, challenger, bounds, topks)
    return {
        "name": name,
        "data": str(data_dir),
        "scores": scores,
        "bounds": bounds,
        "base": base,
        "set_scores": set_scores,
        "challenger": challenger,
        "features": feature_rows(
            base,
            challenger,
            scores,
            bounds,
            positions=positions,
            route_tags=route_tags,
            candidate_ranks=candidate_ranks,
            action_indices=action_indices,
        ),
        "base_metrics": base_metrics,
        "challenger_metrics": challenger_metrics,
    }


def load_blended_dataset_predictions(
    name: str,
    data_dir: Path,
    models: list[ActionValueSetReranker],
    weights: list[float],
    gamma: float,
    device: torch.device,
    batch_size: int,
    topks: list[int],
) -> dict:
    blended: dict | None = None
    set_scores: np.ndarray | None = None
    for model, weight in zip(models, weights):
        dataset = load_dataset_predictions(name, data_dir, model, gamma=1.0, device=device, batch_size=batch_size, topks=topks)
        if blended is None:
            blended = dict(dataset)
            set_scores = np.asarray(dataset["set_scores"], dtype=np.float32) * float(weight)
        else:
            if blended["name"] != dataset["name"] or blended["data"] != dataset["data"]:
                raise ValueError("Dataset changed while blending checkpoints")
            if len(blended["base"]) != len(dataset["base"]):
                raise ValueError("Dataset sample count changed while blending checkpoints")
            set_scores = set_scores + np.asarray(dataset["set_scores"], dtype=np.float32) * float(weight)
    if blended is None or set_scores is None:
        raise RuntimeError("No model predictions were loaded")

    base = np.asarray(blended["base"], dtype=np.float32)
    challenger = base + float(gamma) * (set_scores - base)
    topks = list(topks)
    challenger_metrics, _ = summarize_groups(blended["scores"], challenger, blended["bounds"], topks)

    # Rebuild features because the challenger top action may change after
    # blending even when each component model was evaluated on the same base.
    data_path = Path(blended["data"])
    n_samples = len(base)
    positions = np.load(data_path / "positions.npy", mmap_mode="r")[:n_samples] if (data_path / "positions.npy").exists() else None
    route_tags = np.load(data_path / "route_tags.npy", mmap_mode="r")[:n_samples] if (data_path / "route_tags.npy").exists() else None
    candidate_ranks = (
        np.load(data_path / "candidate_ranks.npy", mmap_mode="r")[:n_samples]
        if (data_path / "candidate_ranks.npy").exists()
        else None
    )
    action_indices = (
        np.load(data_path / "action_indices.npy", mmap_mode="r")[:n_samples]
        if (data_path / "action_indices.npy").exists()
        else None
    )
    blended["set_scores"] = set_scores
    blended["challenger"] = challenger
    blended["challenger_metrics"] = challenger_metrics
    blended["features"] = feature_rows(
        base,
        challenger,
        blended["scores"],
        blended["bounds"],
        positions=positions,
        route_tags=route_tags,
        candidate_ranks=candidate_ranks,
        action_indices=action_indices,
    )
    return blended


def gate_mask(rows: list[dict[str, float]], gate: Gate) -> np.ndarray:
    if gate.gate_type == "always_base":
        return np.zeros(len(rows), dtype=bool)
    if gate.gate_type == "always_challenger":
        return np.ones(len(rows), dtype=bool)
    if gate.gate_type != "single_feature":
        raise ValueError(f"Unsupported gate type: {gate.gate_type}")
    values = np.asarray([float(row.get(gate.feature, 0.0)) for row in rows], dtype=np.float64)
    if gate.direction == "ge":
        return values >= float(gate.threshold)
    if gate.direction == "le":
        return values <= float(gate.threshold)
    raise ValueError(f"Unsupported direction: {gate.direction}")


def apply_gate_scores(dataset: dict, gate: Gate) -> tuple[np.ndarray, np.ndarray]:
    selected = gate_mask(dataset["features"], gate)
    pred = np.asarray(dataset["base"], dtype=np.float32).copy()
    for group_id, use_challenger in enumerate(selected):
        if not bool(use_challenger):
            continue
        start, end = dataset["bounds"][group_id]
        pred[start:end] = dataset["challenger"][start:end]
    return pred, selected


def evaluate_gate(dataset: dict, gate: Gate, topks: list[int]) -> dict:
    pred, selected = apply_gate_scores(dataset, gate)
    metrics, _ = summarize_groups(dataset["scores"], pred, dataset["bounds"], topks)
    out = gate.to_dict()
    out.update(
        {
            "dataset": dataset["name"],
            "groups": int(metrics["groups"]),
            "switch_groups": int(selected.sum()),
            "switch_fraction": float(selected.mean()) if len(selected) else 0.0,
            "metrics": metrics,
        }
    )
    return out


def candidate_thresholds(values: np.ndarray, max_thresholds: int) -> list[float]:
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float64)
    if len(finite) == 0:
        return []
    unique = np.unique(finite)
    if len(unique) <= max_thresholds:
        return [float(x) for x in unique]
    qs = np.linspace(0.02, 0.98, max_thresholds)
    return [float(x) for x in np.unique(np.quantile(finite, qs))]


def candidate_gates(rows: list[dict[str, float]], max_thresholds: int) -> list[Gate]:
    if not rows:
        return [Gate("always_base"), Gate("always_challenger")]
    skip = {
        "group_id",
        "base_regret",
        "challenger_regret",
        "challenger_minus_base_true",
        "challenger_improves",
        "challenger_hurts",
    }
    features = [key for key in rows[0].keys() if key not in skip]
    gates = [Gate("always_base"), Gate("always_challenger")]
    for feature in features:
        values = np.asarray([float(row.get(feature, 0.0)) for row in rows], dtype=np.float64)
        if len(np.unique(values[np.isfinite(values)])) <= 1:
            continue
        for threshold in candidate_thresholds(values, max_thresholds=max_thresholds):
            gates.append(Gate("single_feature", feature=feature, direction="ge", threshold=threshold))
            gates.append(Gate("single_feature", feature=feature, direction="le", threshold=threshold))
    return gates


def aggregate_results(rows: list[dict]) -> dict:
    groups = sum(int(row["groups"]) for row in rows)
    if groups <= 0:
        return {"groups": 0}
    metric_keys = [
        "group_top1",
        "group_top3",
        "group_top5",
        "group_top10",
        "group_top15",
        "group_top20",
        "group_top1_regret",
        "group_top3_rerank_regret",
        "group_top10_rerank_regret",
    ]
    out = {"groups": int(groups), "switch_groups": int(sum(int(row["switch_groups"]) for row in rows))}
    out["switch_fraction"] = out["switch_groups"] / groups
    for key in metric_keys:
        out[key] = sum(float(row["metrics"].get(key, 0.0)) * int(row["groups"]) for row in rows) / groups
    return out


def score_gate_result(row: dict, min_switches: int) -> tuple:
    metrics = row["metrics"]
    enough = int(row["switch_groups"]) >= int(min_switches) or row["gate_type"] in {"always_base", "always_challenger"}
    return (
        1 if enough else 0,
        float(metrics.get("group_top1", 0.0)),
        -float(metrics.get("group_top1_regret", 0.0)),
        float(metrics.get("group_top3", 0.0)),
        float(metrics.get("group_top10", 0.0)),
        -abs(float(row.get("switch_fraction", 0.0)) - 0.25),
    )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Search and replay simple gates for set-residual predictions")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoints", nargs="*")
    parser.add_argument("--weights", default="")
    parser.add_argument("--eval-dataset", action="append", default=[], help="name=path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--topks", default="1,3,5,10,15,20")
    parser.add_argument("--max-thresholds", type=int, default=64)
    parser.add_argument("--min-switches", type=int, default=20)
    parser.add_argument("--top-gates", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    started = time.time()
    topks = parse_topks(args.topks)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoints = list(args.checkpoints or [])
    if args.checkpoint:
        checkpoints.insert(0, args.checkpoint)
    if not checkpoints:
        raise ValueError("Provide --checkpoint or --checkpoints")
    weights = parse_weights(args.weights, len(checkpoints))
    models = [ActionValueSetReranker.from_checkpoint(checkpoint, map_location=device).to(device) for checkpoint in checkpoints]

    train_name, train_path = parse_named_path(args.train_data)
    train = load_blended_dataset_predictions(train_name, train_path, models, weights, args.gamma, device, args.batch_size, topks)
    gates = candidate_gates(train["features"], max_thresholds=args.max_thresholds)
    train_results = [evaluate_gate(train, gate, topks) for gate in gates]
    train_results.sort(key=lambda row: score_gate_result(row, args.min_switches), reverse=True)
    selected = train_results[0]
    selected_gate = Gate(
        str(selected["gate_type"]),
        str(selected.get("feature", "")),
        str(selected.get("direction", "")),
        float(selected.get("threshold", 0.0)),
    )

    eval_outputs = {}
    for spec in args.eval_dataset:
        name, path = parse_named_path(spec)
        dataset = load_blended_dataset_predictions(name, path, models, weights, args.gamma, device, args.batch_size, topks)
        selected_result = evaluate_gate(dataset, selected_gate, topks)
        baseline_result = evaluate_gate(dataset, Gate("always_base"), topks)
        challenger_result = evaluate_gate(dataset, Gate("always_challenger"), topks)
        top_gate_replay = []
        for row in train_results[: args.top_gates]:
            gate = Gate(
                str(row["gate_type"]),
                str(row.get("feature", "")),
                str(row.get("direction", "")),
                float(row.get("threshold", 0.0)),
            )
            top_gate_replay.append(evaluate_gate(dataset, gate, topks))
        eval_outputs[name] = {
            "data": str(path),
            "baseline": baseline_result,
            "challenger": challenger_result,
            "selected_gate": selected_result,
            "top_train_gate_replay": top_gate_replay,
        }

    selected_eval_rows = [value["selected_gate"] for value in eval_outputs.values()]
    baseline_eval_rows = [value["baseline"] for value in eval_outputs.values()]
    challenger_eval_rows = [value["challenger"] for value in eval_outputs.values()]
    summary = {
        "checkpoint": str(checkpoints[0]) if len(checkpoints) == 1 else "set_checkpoint_blend",
        "checkpoints": [str(path) for path in checkpoints],
        "weights": weights,
        "gamma": float(args.gamma),
        "train_data": str(train_path),
        "topks": topks,
        "selected_gate": selected_gate.to_dict(),
        "train_baseline": evaluate_gate(train, Gate("always_base"), topks),
        "train_challenger": evaluate_gate(train, Gate("always_challenger"), topks),
        "train_selected": selected,
        "train_top_gates": train_results[: args.top_gates],
        "evals": eval_outputs,
        "eval_aggregate": {
            "baseline": aggregate_results(baseline_eval_rows),
            "challenger": aggregate_results(challenger_eval_rows),
            "selected_gate": aggregate_results(selected_eval_rows),
        },
        "elapsed_seconds": time.time() - started,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (out_dir / "train_gate_candidates.jsonl").open("w", encoding="utf-8") as f:
        for row in train_results:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")

    print("set residual gate search")
    print(f"  train: {train_path}")
    print(f"  selected: {selected_gate.to_dict()}")
    for label, row in summary["eval_aggregate"].items():
        print(
            f"  eval aggregate {label}: groups={row.get('groups', 0)} "
            f"switch={row.get('switch_groups', 0)} "
            f"top1={row.get('group_top1', 0.0):.1%} "
            f"top3={row.get('group_top3', 0.0):.1%} "
            f"top10={row.get('group_top10', 0.0):.1%} "
            f"reg1={row.get('group_top1_regret', 0.0):.3f}"
        )
    print(f"  summary={out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
