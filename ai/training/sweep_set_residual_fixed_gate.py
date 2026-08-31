"""Sweep gamma for a fixed set-residual candidate-rank gate.

This is a diagnostic helper for T2 Top1 work.  It computes set-reranker scores
once per dataset, then evaluates:

    base + gamma * (set_residual - base)

with a fixed group-level gate.  The default gate keeps the set-residual
challenger only when its top candidate rank is not worse than the base top
candidate rank.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.models.action_value_set_reranker import ActionValueSetReranker
from ai.training.evaluate_action_value_reranker import parse_topks
from ai.training.evaluate_action_value_set_reranker import predict_groups, summarize_groups
from ai.training.evaluate_set_residual_gate import feature_rows, parse_named_path
from ai.training.train_action_value_set_reranker import (
    CandidateGroupDataset,
    build_group_bounds,
    collate_groups,
    load_metadata,
)


def parse_gammas(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part.strip()]


def parse_weights(value: str, n: int) -> list[float]:
    if not value.strip():
        return [1.0 / max(n, 1)] * n
    weights = [float(part) for part in value.split(",") if part.strip()]
    if len(weights) != n:
        raise ValueError(f"--weights has {len(weights)} values but {n} checkpoints were provided")
    total = float(sum(weights))
    if total <= 0.0:
        raise ValueError("--weights must sum to a positive value")
    return [w / total for w in weights]


def load_raw_predictions(
    name: str,
    data_dir: Path,
    model: ActionValueSetReranker,
    device: torch.device,
    batch_size: int,
) -> dict:
    meta = load_metadata(data_dir)
    states = np.load(data_dir / "states.npy", mmap_mode="r")
    n_samples = min(int(meta.get("n_samples", states.shape[0])), int(states.shape[0]))
    scores = np.asarray(np.load(data_dir / "scores.npy", mmap_mode="r")[:n_samples], dtype=np.float32)
    group_ids = np.asarray(np.load(data_dir / "group_ids.npy", mmap_mode="r")[:n_samples], dtype=np.int64)
    bounds = build_group_bounds(group_ids)
    base_path = data_dir / "base_scores.npy"
    if not base_path.exists():
        raise FileNotFoundError(f"Missing base_scores.npy in {data_dir}")
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
    arrays = {}
    for key in ("positions", "route_tags", "candidate_ranks", "action_indices"):
        path = data_dir / f"{key}.npy"
        arrays[key] = np.load(path, mmap_mode="r")[:n_samples] if path.exists() else None
    return {
        "name": name,
        "data": str(data_dir),
        "scores": scores,
        "bounds": bounds,
        "base": base,
        "set_scores": set_scores,
        **arrays,
    }


def apply_candidate_rank_gate(dataset: dict, challenger: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rows = feature_rows(
        np.asarray(dataset["base"], dtype=np.float32),
        np.asarray(challenger, dtype=np.float32),
        np.asarray(dataset["scores"], dtype=np.float32),
        dataset["bounds"],
        positions=dataset.get("positions"),
        route_tags=dataset.get("route_tags"),
        candidate_ranks=dataset.get("candidate_ranks"),
        action_indices=dataset.get("action_indices"),
    )
    selected = np.asarray(
        [float(row.get("candidate_rank_gap_chal_minus_base", 1.0)) <= 0.0 for row in rows],
        dtype=bool,
    )
    pred = np.asarray(dataset["base"], dtype=np.float32).copy()
    for group_id, enabled in enumerate(selected):
        if not bool(enabled):
            continue
        start, end = dataset["bounds"][group_id]
        pred[start:end] = challenger[start:end]
    return pred, selected, rows


def evaluate_dataset_gamma(dataset: dict, gamma: float, topks: list[int]) -> dict:
    challenger = np.asarray(dataset["base"], dtype=np.float32) + float(gamma) * (
        np.asarray(dataset["set_scores"], dtype=np.float32) - np.asarray(dataset["base"], dtype=np.float32)
    )
    pred, selected, rows = apply_candidate_rank_gate(dataset, challenger)
    metrics, detail_rows = summarize_groups(dataset["scores"], pred, dataset["bounds"], topks)
    misses = [row for row in detail_rows if int(row["teacher_rank_by_model"]) > 1]
    return {
        "dataset": dataset["name"],
        "gamma": float(gamma),
        "groups": int(metrics["groups"]),
        "switch_groups": int(selected.sum()),
        "switch_fraction": float(selected.mean()) if len(selected) else 0.0,
        "metrics": metrics,
        "misses": misses,
        "feature_rows": rows,
    }


def aggregate(rows: list[dict]) -> dict:
    groups = sum(int(row["groups"]) for row in rows)
    if groups <= 0:
        return {"groups": 0}
    out = {
        "groups": int(groups),
        "switch_groups": int(sum(int(row["switch_groups"]) for row in rows)),
        "misses": int(sum(len(row["misses"]) for row in rows)),
    }
    out["switch_fraction"] = out["switch_groups"] / groups
    for key in (
        "group_top1",
        "group_top3",
        "group_top5",
        "group_top10",
        "group_top15",
        "group_top20",
        "group_top1_regret",
        "group_top3_rerank_regret",
        "group_top10_rerank_regret",
    ):
        out[key] = sum(float(row["metrics"].get(key, 0.0)) * int(row["groups"]) for row in rows) / groups
    return out


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sweep gamma for a fixed candidate-rank set-residual gate")
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoints", nargs="*")
    parser.add_argument("--weights", default="")
    parser.add_argument("--dataset", action="append", required=True, help="name=path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gammas", default="0,0.025,0.05,0.075,0.1,0.125,0.15,0.2,0.25,0.3,0.4,0.5,0.75,1.0")
    parser.add_argument("--external-names", default="")
    parser.add_argument("--topks", default="1,3,5,10,15,20")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    started = time.time()
    topks = parse_topks(args.topks)
    gammas = parse_gammas(args.gammas)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoints = list(args.checkpoints or [])
    if args.checkpoint:
        checkpoints.insert(0, args.checkpoint)
    if not checkpoints:
        raise ValueError("Provide --checkpoint or --checkpoints")
    weights = parse_weights(args.weights, len(checkpoints))
    datasets = None
    for checkpoint, weight in zip(checkpoints, weights):
        model = ActionValueSetReranker.from_checkpoint(checkpoint, map_location=device).to(device)
        model_datasets = [load_raw_predictions(*parse_named_path(spec), model, device, args.batch_size) for spec in args.dataset]
        if datasets is None:
            datasets = []
            for dataset in model_datasets:
                item = dict(dataset)
                item["set_scores"] = np.asarray(dataset["set_scores"], dtype=np.float32) * float(weight)
                datasets.append(item)
        else:
            for item, dataset in zip(datasets, model_datasets):
                if item["name"] != dataset["name"] or item["data"] != dataset["data"]:
                    raise ValueError("Dataset order changed while blending set checkpoints")
                item["set_scores"] = np.asarray(item["set_scores"], dtype=np.float32) + (
                    np.asarray(dataset["set_scores"], dtype=np.float32) * float(weight)
                )
        del model
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.empty_cache()
    if datasets is None:
        raise RuntimeError("No datasets loaded")
    external_names = {part.strip() for part in args.external_names.split(",") if part.strip()}

    per_gamma = []
    best_miss_rows: list[dict] = []
    best_gamma = None
    best_score = None
    for gamma in gammas:
        rows = [evaluate_dataset_gamma(dataset, gamma, topks) for dataset in datasets]
        agg_all = aggregate(rows)
        external_rows = [row for row in rows if row["dataset"] in external_names]
        agg_external = aggregate(external_rows) if external_rows else {}
        item = {
            "gamma": float(gamma),
            "aggregate_all": agg_all,
            "aggregate_external": agg_external,
            "datasets": {
                row["dataset"]: {
                    "groups": row["groups"],
                    "switch_groups": row["switch_groups"],
                    "switch_fraction": row["switch_fraction"],
                    "metrics": row["metrics"],
                    "misses": len(row["misses"]),
                }
                for row in rows
            },
        }
        per_gamma.append(item)
        score = (
            float(agg_all.get("group_top1", 0.0)),
            -float(agg_all.get("group_top1_regret", 0.0)),
            float(agg_external.get("group_top1", 0.0)) if agg_external else 0.0,
            -float(agg_external.get("group_top1_regret", 0.0)) if agg_external else 0.0,
        )
        if best_score is None or score > best_score:
            best_score = score
            best_gamma = float(gamma)
            best_miss_rows = []
            for row in rows:
                for miss in row["misses"]:
                    best_miss_rows.append({"dataset": row["dataset"], **miss})

    per_gamma.sort(
        key=lambda row: (
            float(row["aggregate_all"].get("group_top1", 0.0)),
            -float(row["aggregate_all"].get("group_top1_regret", 0.0)),
            float(row["aggregate_external"].get("group_top1", 0.0)) if row["aggregate_external"] else 0.0,
        ),
        reverse=True,
    )
    summary = {
        "checkpoint": str(checkpoints[0]) if len(checkpoints) == 1 else "set_checkpoint_blend",
        "checkpoints": [str(path) for path in checkpoints],
        "weights": weights,
        "datasets": [dataset["data"] for dataset in datasets],
        "external_names": sorted(external_names),
        "gammas": gammas,
        "best_gamma": best_gamma,
        "results": per_gamma,
        "elapsed_seconds": time.time() - started,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (out_dir / "best_gamma_misses.jsonl").open("w", encoding="utf-8") as f:
        for row in sorted(best_miss_rows, key=lambda x: float(x.get("regret", 0.0)), reverse=True):
            f.write(json.dumps(row, separators=(",", ":")) + "\n")

    print("fixed candidate-rank gate gamma sweep")
    print(f"  best_gamma={best_gamma}")
    for row in per_gamma[:8]:
        all_m = row["aggregate_all"]
        ext_m = row["aggregate_external"]
        print(
            f"  gamma={row['gamma']:.3f} all_top1={all_m.get('group_top1', 0.0):.1%} "
            f"all_reg1={all_m.get('group_top1_regret', 0.0):.3f} "
            f"ext_top1={ext_m.get('group_top1', 0.0):.1%} "
            f"ext_reg1={ext_m.get('group_top1_regret', 0.0):.3f} "
            f"switch={all_m.get('switch_groups', 0)}/{all_m.get('groups', 0)}"
        )
    print(f"  summary={out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
