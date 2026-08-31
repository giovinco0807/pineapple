"""Create a hard-negative weighted copy of action-value reranker data.

The source arrays are kept as hardlinks where possible.  Only
``sample_weights.npy``, ``group_sample_weights.npy``, and metadata are written
fresh.  Hard groups are selected from the training split when the teacher-best
candidate is ranked below the requested Top-K by the current model.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.models.action_value_reranker import ActionValueReranker


LINKED_ARRAYS = (
    "action_indices.npy",
    "aux_label_mask.npy",
    "bust.npy",
    "candidate_ranks.npy",
    "fl.npy",
    "fl_types.npy",
    "group_ids.npy",
    "positions.npy",
    "route_tags.npy",
    "scores.npy",
    "states.npy",
    "teacher_gaps.npy",
    "turns.npy",
)


def parse_position(value: str) -> int | None:
    text = value.strip().lower()
    if not text or text in {"all", "mixed"}:
        return None
    if text in {"btn", "button", "second"}:
        return 1
    if text in {"bb", "blind", "first"}:
        return 0
    raise ValueError("--position must be all, btn, or bb")


def load_metadata(data_dir: Path) -> dict:
    path = data_dir / "metadata.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def build_group_bounds(group_ids: np.ndarray) -> list[tuple[int, int]]:
    if len(group_ids) == 0:
        return []
    bounds: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(group_ids)):
        if int(group_ids[i]) != int(group_ids[i - 1]):
            bounds.append((start, i))
            start = i
    bounds.append((start, len(group_ids)))
    return bounds


def validation_groups(group_ids: np.ndarray, val_frac: float, seed: int) -> set[int]:
    unique_groups = np.unique(group_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    n_val_groups = max(1, int(len(unique_groups) * val_frac))
    return {int(group) for group in unique_groups[:n_val_groups]}


def link_or_copy(src: Path, dst: Path) -> str:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


@torch.no_grad()
def predict_scores(
    checkpoint: Path,
    states: np.ndarray,
    turns: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model = ActionValueReranker.from_checkpoint(checkpoint, map_location=device).to(device)
    model.eval()
    pred = np.zeros(int(states.shape[0]), dtype=np.float32)
    for start in range(0, int(states.shape[0]), batch_size):
        end = min(int(states.shape[0]), start + batch_size)
        state = torch.from_numpy(np.array(states[start:end], dtype=np.float32, copy=True)).to(device)
        turn = torch.from_numpy(np.asarray(turns[start:end], dtype=np.int64)).to(device)
        out = model.predict_components(state, turn=turn)
        pred[start:end] = out["score"].detach().cpu().numpy()
    return pred


def create(args: argparse.Namespace) -> None:
    source = Path(args.source)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    meta = load_metadata(source)

    states = np.load(source / "states.npy", mmap_mode="r")
    n_samples = min(int(meta.get("n_samples", states.shape[0])), int(states.shape[0]))
    scores = np.asarray(np.load(source / "scores.npy", mmap_mode="r")[:n_samples], dtype=np.float32)
    turns = np.asarray(np.load(source / "turns.npy", mmap_mode="r")[:n_samples], dtype=np.int64)
    group_ids = np.asarray(np.load(source / "group_ids.npy", mmap_mode="r")[:n_samples], dtype=np.int64)
    base_weights = np.asarray(np.load(source / "sample_weights.npy", mmap_mode="r")[:n_samples], dtype=np.float32)
    positions_path = source / "positions.npy"
    positions = (
        np.asarray(np.load(positions_path, mmap_mode="r")[:n_samples], dtype=np.int8)
        if positions_path.exists()
        else None
    )
    selected_position = parse_position(args.position)
    if selected_position is not None and positions is None:
        raise FileNotFoundError(f"{positions_path} is required when --position is set")

    val_groups = validation_groups(group_ids, args.val_frac, args.seed)
    bounds = build_group_bounds(group_ids)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    pred = predict_scores(Path(args.checkpoint), states[:n_samples], turns, args.batch_size, device)

    sample_weights = base_weights.copy()
    group_weights = np.ones(len(bounds), dtype=np.float32)
    stats = {
        "groups": 0,
        "train_groups": 0,
        "position_train_groups": 0,
        "top1_miss_groups": 0,
        "topk_miss_groups": 0,
        "weighted_candidates": 0,
        "teacher_best_boosted": 0,
        "confusers_boosted": 0,
    }

    for group_index, (start, end) in enumerate(bounds):
        stats["groups"] += 1
        group_id = int(group_ids[start])
        if group_id in val_groups:
            continue
        stats["train_groups"] += 1
        idx = np.arange(start, end, dtype=np.int64)
        if selected_position is not None:
            idx = idx[np.asarray(positions[idx], dtype=np.int8) == int(selected_position)]
        if len(idx) < 2:
            continue
        stats["position_train_groups"] += 1
        true = np.asarray(scores[idx], dtype=np.float64)
        predicted = np.asarray(pred[idx], dtype=np.float64)
        true_best_local = int(np.argmax(true))
        teacher_idx = int(idx[true_best_local])
        ordered = np.argsort(-predicted)
        teacher_rank = int(np.where(ordered == true_best_local)[0][0]) + 1
        topk_kept = ordered[: min(args.target_topk, len(ordered))]
        best_score = float(true[true_best_local])
        topk_best_score = float(true[topk_kept].max())
        topk_regret = max(0.0, best_score - topk_best_score)

        if teacher_rank > 1:
            stats["top1_miss_groups"] += 1
            sample_weights[teacher_idx] += float(args.teacher_best_add)
            stats["teacher_best_boosted"] += 1
            confusers = [int(idx[local]) for local in ordered[: min(args.top_confusers, len(ordered))] if local != true_best_local]
            for confuser_idx in confusers:
                sample_weights[confuser_idx] += float(args.confuser_add)
                stats["confusers_boosted"] += 1

        if teacher_rank > args.target_topk:
            stats["topk_miss_groups"] += 1
            group_weights[group_index] = float(
                1.0
                + args.hard_group_add
                + min(topk_regret, args.regret_cap) * args.regret_group_scale
            )
            sample_weights[teacher_idx] += float(args.topk_teacher_best_add)
            stats["teacher_best_boosted"] += 1
            for local in ordered[: min(args.target_topk, len(ordered))]:
                if int(local) == true_best_local:
                    continue
                sample_weights[int(idx[int(local)])] += float(args.topk_confuser_add)
                stats["confusers_boosted"] += 1

        stats["weighted_candidates"] += int(np.sum(sample_weights[idx] > base_weights[idx]))

    sample_weights = np.clip(sample_weights, 0.01, float(args.max_sample_weight)).astype(np.float32)
    np.save(output / "sample_weights.npy", sample_weights)
    np.save(output / "group_sample_weights.npy", group_weights)

    link_modes: dict[str, str] = {}
    for name in LINKED_ARRAYS:
        src = source / name
        if src.exists():
            link_modes[name] = link_or_copy(src, output / name)

    new_meta = dict(meta)
    new_meta.update(
        {
            "source": str(source),
            "hard_negative_source": str(source),
            "hard_negative_checkpoint": str(args.checkpoint),
            "hard_negative_position": args.position,
            "hard_negative_target_topk": int(args.target_topk),
            "hard_negative_stats": stats,
            "group_sample_weight_mean": float(group_weights.mean()) if len(group_weights) else 0.0,
            "group_sample_weight_max": float(group_weights.max()) if len(group_weights) else 0.0,
            "sample_weight_mean": float(sample_weights.mean()) if len(sample_weights) else 0.0,
            "sample_weight_max": float(sample_weights.max()) if len(sample_weights) else 0.0,
            "link_modes": link_modes,
        }
    )
    with (output / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(new_meta, f, indent=2)
    print(json.dumps(new_meta, indent=2))


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create hard-negative weighted action-value reranker data")
    parser.add_argument("--source", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--position", default="all")
    parser.add_argument("--target-topk", type=int, default=3)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--teacher-best-add", type=float, default=2.0)
    parser.add_argument("--confuser-add", type=float, default=0.75)
    parser.add_argument("--topk-teacher-best-add", type=float, default=6.0)
    parser.add_argument("--topk-confuser-add", type=float, default=1.5)
    parser.add_argument("--top-confusers", type=int, default=5)
    parser.add_argument("--hard-group-add", type=float, default=6.0)
    parser.add_argument("--regret-group-scale", type=float, default=1.5)
    parser.add_argument("--regret-cap", type=float, default=8.0)
    parser.add_argument("--max-sample-weight", type=float, default=16.0)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    create(args)


if __name__ == "__main__":
    main()
