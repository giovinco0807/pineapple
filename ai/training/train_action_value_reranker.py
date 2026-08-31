"""Train an action-value reranker from candidate-level teacher data."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.models.action_value_reranker import ActionValueReranker


FL_TYPE_KEYS = ("qq", "kk", "aa", "trips")


class ActionValueDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        indices: np.ndarray,
        score_mean: float,
        score_std: float,
    ):
        self.data_dir = Path(data_dir)
        self.indices = indices.astype(np.int64, copy=False)
        self.states = np.load(self.data_dir / "states.npy", mmap_mode="r")
        self.scores = np.load(self.data_dir / "scores.npy", mmap_mode="r")
        self.bust = np.load(self.data_dir / "bust.npy", mmap_mode="r")
        self.fl = np.load(self.data_dir / "fl.npy", mmap_mode="r")
        self.fl_types = (
            np.load(self.data_dir / "fl_types.npy", mmap_mode="r")
            if (self.data_dir / "fl_types.npy").exists()
            else None
        )
        self.sample_weights = (
            np.load(self.data_dir / "sample_weights.npy", mmap_mode="r")
            if (self.data_dir / "sample_weights.npy").exists()
            else None
        )
        self.aux_label_mask = (
            np.load(self.data_dir / "aux_label_mask.npy", mmap_mode="r")
            if (self.data_dir / "aux_label_mask.npy").exists()
            else None
        )
        self.turns = np.load(self.data_dir / "turns.npy", mmap_mode="r")
        self.score_mean = float(score_mean)
        self.score_std = max(float(score_std), 1e-6)

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def input_dim(self) -> int:
        return int(self.states.shape[1])

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        score = (float(self.scores[idx]) - self.score_mean) / self.score_std
        fl_types = (
            np.asarray(self.fl_types[idx], dtype=np.float32)
            if self.fl_types is not None
            else np.zeros(4, dtype=np.float32)
        )
        weight = float(self.sample_weights[idx]) if self.sample_weights is not None else 1.0
        aux_label_mask = float(self.aux_label_mask[idx]) if self.aux_label_mask is not None else 1.0
        return {
            "state": torch.from_numpy(np.array(self.states[idx], dtype=np.float32, copy=True)),
            "score": torch.tensor(score, dtype=torch.float32),
            "bust": torch.tensor(float(self.bust[idx]), dtype=torch.float32),
            "fl": torch.tensor(float(self.fl[idx]), dtype=torch.float32),
            "fl_types": torch.from_numpy(np.array(fl_types, dtype=np.float32, copy=True)),
            "weight": torch.tensor(weight, dtype=torch.float32),
            "aux_label_mask": torch.tensor(aux_label_mask, dtype=torch.float32),
            "turn": torch.tensor(int(self.turns[idx]), dtype=torch.long),
            "idx": torch.tensor(idx, dtype=torch.long),
        }


def load_metadata(data_dir: Path) -> dict:
    path = data_dir / "metadata.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_checkpoint_normalization(path: str | Path | None) -> tuple[float, float] | None:
    if not path:
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        return None
    norm = ckpt.get("normalization", {})
    score_mean = norm.get("score_mean", ckpt.get("score_mean"))
    score_std = norm.get("score_std", ckpt.get("score_std"))
    if score_mean is None or score_std is None:
        return None
    return float(score_mean), max(float(score_std), 1e-6)


def load_compatible_state_dict(
    model: ActionValueReranker,
    source_state_dict: dict,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Load checkpoint weights, allowing input feature expansion.

    ``torch.nn.Module.load_state_dict(strict=False)`` still rejects tensors with
    mismatched shapes.  Action-feature experiments expand the input vector while
    keeping the old 520/522 base features first, so the first linear layer can
    reuse the old columns and leave newly added columns at their initialization.
    """
    target_state_dict = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    partial: list[str] = []
    shape_mismatched: list[str] = []
    unexpected = [key for key in source_state_dict if key not in target_state_dict]

    for key, value in source_state_dict.items():
        if key not in target_state_dict:
            continue
        target = target_state_dict[key]
        if tuple(value.shape) == tuple(target.shape):
            filtered[key] = value.to(device=target.device, dtype=target.dtype)
            continue
        if (
            key == "input_proj.0.weight"
            and value.ndim == 2
            and target.ndim == 2
            and value.shape[0] == target.shape[0]
            and value.shape[1] <= target.shape[1]
        ):
            expanded = target.clone()
            expanded[:, : value.shape[1]] = value.to(device=target.device, dtype=target.dtype)
            filtered[key] = expanded
            partial.append(f"{key}:{tuple(value.shape)}->{tuple(target.shape)}")
            continue
        shape_mismatched.append(f"{key}:{tuple(value.shape)}!={tuple(target.shape)}")

    missing, _ = model.load_state_dict(filtered, strict=False)
    return list(missing), unexpected, partial, shape_mismatched


def split_indices(data_dir: Path, n_samples: int, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    group_ids = np.load(data_dir / "group_ids.npy", mmap_mode="r")[:n_samples]
    unique_groups = np.unique(np.asarray(group_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    n_val_groups = max(1, int(len(unique_groups) * val_frac))
    val_groups = set(int(x) for x in unique_groups[:n_val_groups])

    train_parts = []
    val_parts = []
    chunk = 1_000_000
    for start in range(0, n_samples, chunk):
        end = min(n_samples, start + chunk)
        gids = np.asarray(group_ids[start:end])
        mask = np.fromiter((int(g) in val_groups for g in gids), dtype=bool, count=len(gids))
        base = np.arange(start, end, dtype=np.int64)
        val_parts.append(base[mask])
        train_parts.append(base[~mask])

    train_idx = np.concatenate(train_parts) if train_parts else np.empty(0, dtype=np.int64)
    val_idx = np.concatenate(val_parts) if val_parts else np.empty(0, dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def parse_turns(value: str) -> set[int] | None:
    if not value.strip():
        return None
    return {int(part) for part in value.split(",") if part.strip()}


def filter_indices_by_turn(data_dir: Path, indices: np.ndarray, turns: set[int] | None) -> np.ndarray:
    if turns is None or len(indices) == 0:
        return indices
    turn_arr = np.load(data_dir / "turns.npy", mmap_mode="r")
    selected = np.asarray(indices, dtype=np.int64)
    mask = np.isin(np.asarray(turn_arr[selected], dtype=np.int16), list(turns))
    return selected[mask]


def parse_position(value: str) -> int | None:
    text = value.strip().lower()
    if not text or text in {"all", "mixed"}:
        return None
    if text in {"btn", "button", "second", "koukou", "後攻"}:
        return 1
    if text in {"bb", "blind", "first", "senkou", "先行"}:
        return 0
    raise ValueError("--train-position must be one of all, btn, or bb")


def filter_indices_by_position(data_dir: Path, indices: np.ndarray, position: int | None) -> np.ndarray:
    if position is None or len(indices) == 0:
        return indices
    positions_path = data_dir / "positions.npy"
    if not positions_path.exists():
        raise FileNotFoundError(
            f"{positions_path} is required when --train-position is set"
        )
    positions = np.load(positions_path, mmap_mode="r")
    selected = np.asarray(indices, dtype=np.int64)
    mask = np.asarray(positions[selected], dtype=np.int8) == int(position)
    return selected[mask]


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    denom = math.sqrt(float((x * x).sum() * (y * y).sum()))
    if denom <= 1e-12:
        return 0.0
    return float((x * y).sum() / denom)


def weighted_mean(loss: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight.to(loss.device)
    while weight.ndim < loss.ndim:
        weight = weight.unsqueeze(-1)
    weight = weight.expand_as(loss)
    return (loss * weight).sum() / weight.sum().clamp(min=1e-6)


def build_group_bounds(data_dir: Path, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    group_ids = np.asarray(np.load(data_dir / "group_ids.npy", mmap_mode="r")[:n_samples])
    if len(group_ids) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    starts = [0]
    for i in range(1, len(group_ids)):
        if group_ids[i] != group_ids[i - 1]:
            starts.append(i)
    starts_arr = np.asarray(starts, dtype=np.int64)
    ends_arr = np.concatenate([starts_arr[1:], np.asarray([len(group_ids)], dtype=np.int64)])
    return starts_arr, ends_arr


def listnet_loss_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    masked_logits = logits.masked_fill(~valid_mask, -1e9)
    masked_targets = targets.masked_fill(~valid_mask, -1e9)
    target_probs = F.softmax(masked_targets / max(temperature, 1e-6), dim=-1)
    log_probs = F.log_softmax(masked_logits, dim=-1)
    loss = target_probs * (torch.log(target_probs.clamp(min=1e-10)) - log_probs)
    return loss.masked_fill(~valid_mask, 0.0).sum(dim=-1).mean()


def soft_topk_rank_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    target_topk: int,
    temperature: float,
) -> torch.Tensor:
    """Penalize batches where the teacher-best action is softly ranked below top-K."""
    if target_topk <= 0:
        return logits.new_tensor(0.0)

    lengths = valid_mask.sum(dim=1).float()
    masked_targets = targets.masked_fill(~valid_mask, -1e9)
    best_idx = masked_targets.argmax(dim=1)
    best_logits = logits.gather(1, best_idx[:, None])

    competitor_mask = valid_mask.clone()
    competitor_mask.scatter_(1, best_idx[:, None], False)
    soft_above = torch.sigmoid((logits - best_logits) / max(temperature, 1e-6))
    soft_above = soft_above.masked_fill(~competitor_mask, 0.0).sum(dim=1)

    allowed_above = torch.minimum(
        torch.full_like(lengths, float(target_topk - 1)),
        torch.clamp(lengths - 1.0, min=0.0),
    )
    overflow = F.relu(soft_above - allowed_above)
    return (overflow * overflow).mean()


def topk_boundary_margin_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    target_topk: int,
    margin: float,
    temperature: float,
) -> torch.Tensor:
    """Push the teacher-best action above the predicted top-K pruning boundary.

    For top20 pruning, the teacher-best action is safe when at most 19
    competitors score above it.  This loss compares the teacher-best logit with
    the K-th best competitor logit and applies a smooth margin penalty when the
    teacher-best action is too close to, or below, that boundary.
    """
    if target_topk <= 0:
        return logits.new_tensor(0.0)

    masked_targets = targets.masked_fill(~valid_mask, -1e9)
    best_idx = masked_targets.argmax(dim=1)
    best_logits = logits.gather(1, best_idx[:, None]).squeeze(1)

    competitor_mask = valid_mask.clone()
    competitor_mask.scatter_(1, best_idx[:, None], False)
    eligible = competitor_mask.sum(dim=1) >= target_topk
    if not bool(eligible.any()):
        return logits.new_tensor(0.0)

    competitor_logits = logits[eligible].masked_fill(~competitor_mask[eligible], -1e9)
    kth_competitor = torch.topk(competitor_logits, k=target_topk, dim=1).values[:, -1]
    violation = kth_competitor - best_logits[eligible] + float(margin)
    return (
        F.softplus(violation / max(float(temperature), 1e-6))
        * max(float(temperature), 1e-6)
    ).mean()


def train_ranking_batches(
    model: ActionValueReranker,
    optimizer: torch.optim.Optimizer,
    data_dir: Path,
    train_group_ids: np.ndarray,
    group_starts: np.ndarray,
    group_ends: np.ndarray,
    group_sample_weights: np.ndarray | None,
    score_mean: float,
    score_std: float,
    device: torch.device,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> dict:
    if (
        (
            args.ranking_weight <= 0
            and args.topk_rank_weight <= 0
            and args.topk_margin_weight <= 0
        )
        or args.ranking_batches_per_epoch <= 0
        or len(train_group_ids) == 0
    ):
        return {"rank_train_loss": 0.0, "topk_train_loss": 0.0, "topk_margin_train_loss": 0.0}

    states = np.load(data_dir / "states.npy", mmap_mode="r")
    scores = np.load(data_dir / "scores.npy", mmap_mode="r")
    turns = np.load(data_dir / "turns.npy", mmap_mode="r")
    losses = []
    topk_losses = []
    topk_margin_losses = []
    model.train()
    choice_probs = None
    if group_sample_weights is not None and len(train_group_ids) > 0:
        selected_weights = np.asarray(group_sample_weights[train_group_ids], dtype=np.float64)
        selected_weights = np.clip(selected_weights, 0.0, None)
        total_weight = float(selected_weights.sum())
        if total_weight > 0.0:
            choice_probs = selected_weights / total_weight

    for _ in range(args.ranking_batches_per_epoch):
        group_sel = rng.choice(
            train_group_ids,
            size=min(args.group_batch_size, len(train_group_ids)),
            replace=len(train_group_ids) < args.group_batch_size,
            p=choice_probs,
        )
        lengths = [int(group_ends[g] - group_starts[g]) for g in group_sel]
        usable = [i for i, length in enumerate(lengths) if length >= 2]
        if not usable:
            continue
        group_sel = group_sel[usable]
        lengths = [lengths[i] for i in usable]
        max_len = max(lengths)

        state_chunks = []
        target = torch.full((len(group_sel), max_len), -1e9, dtype=torch.float32, device=device)
        mask = torch.zeros((len(group_sel), max_len), dtype=torch.bool, device=device)
        flat_turn_chunks = []
        for row, group_id in enumerate(group_sel):
            start = int(group_starts[group_id])
            end = int(group_ends[group_id])
            length = end - start
            state_chunks.append(np.array(states[start:end], dtype=np.float32, copy=True))
            flat_turn_chunks.append(np.asarray(turns[start:end], dtype=np.int64))
            target[row, :length] = torch.from_numpy(
                ((np.asarray(scores[start:end], dtype=np.float32) - score_mean) / score_std)
            ).to(device)
            mask[row, :length] = True

        flat_states = torch.from_numpy(np.concatenate(state_chunks, axis=0)).to(device)
        flat_turns = torch.from_numpy(np.concatenate(flat_turn_chunks, axis=0)).to(device)
        out = model(flat_states, turn=flat_turns)
        flat_logits = out["score"]
        logits = torch.full((len(group_sel), max_len), -1e9, dtype=torch.float32, device=device)
        pos = 0
        for row, length in enumerate(lengths):
            logits[row, :length] = flat_logits[pos:pos + length]
            pos += length

        rank_loss = listnet_loss_from_logits(logits, target, mask, args.ranking_temperature)
        topk_loss = soft_topk_rank_loss(
            logits,
            target,
            mask,
            args.target_topk,
            args.topk_rank_temperature,
        )
        topk_margin = topk_boundary_margin_loss(
            logits,
            target,
            mask,
            args.target_topk,
            args.topk_margin,
            args.topk_margin_temperature,
        )
        loss = (
            args.ranking_weight * rank_loss
            + args.topk_rank_weight * topk_loss
            + args.topk_margin_weight * topk_margin
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        losses.append(float(rank_loss.item()))
        topk_losses.append(float(topk_loss.item()))
        topk_margin_losses.append(float(topk_margin.item()))

    return {
        "rank_train_loss": float(np.mean(losses)) if losses else 0.0,
        "topk_train_loss": float(np.mean(topk_losses)) if topk_losses else 0.0,
        "topk_margin_train_loss": float(np.mean(topk_margin_losses)) if topk_margin_losses else 0.0,
    }


def select_best_metric(metrics: dict, args: argparse.Namespace) -> float:
    if args.selection_metric == "topk":
        if args.target_topk <= 3:
            hit_key = "group_top3"
            regret_key = "group_top3_rerank_regret"
        elif args.target_topk <= 5:
            hit_key = "group_top5"
            regret_key = "group_top5_rerank_regret"
        elif args.target_topk <= 10:
            hit_key = "group_top10"
            regret_key = "group_top10_rerank_regret"
        elif args.target_topk <= 15:
            hit_key = "group_top15"
            regret_key = "group_top15_rerank_regret"
        else:
            hit_key = "group_top20"
            regret_key = "group_top20_rerank_regret"
        hit_rate = float(metrics.get(hit_key, 0.0))
        topk_regret = float(metrics.get(regret_key, metrics.get("group_regret", 0.0)))
        return topk_regret + (1.0 - hit_rate) * 10.0 + float(metrics.get("score_mae", 0.0)) * 0.01
    return float(metrics["group_regret"]) + float(metrics["score_mae"]) * 0.05


@torch.no_grad()
def evaluate(
    model: ActionValueReranker,
    loader: DataLoader,
    device: torch.device,
    score_mean: float,
    score_std: float,
) -> dict:
    model.eval()
    losses = []
    pred_scores = []
    true_scores = []
    bust_losses = []
    fl_losses = []
    fl_type_losses = []

    for batch in loader:
        state = batch["state"].to(device, non_blocking=True)
        turn = batch["turn"].to(device, non_blocking=True)
        score_t = batch["score"].to(device, non_blocking=True)
        bust_t = batch["bust"].to(device, non_blocking=True)
        fl_t = batch["fl"].to(device, non_blocking=True)
        fl_type_t = batch["fl_types"].to(device, non_blocking=True)
        weight_t = batch["weight"].to(device, non_blocking=True)
        aux_weight_t = weight_t * batch.get("aux_label_mask", torch.ones_like(weight_t)).to(device, non_blocking=True)
        out = model(state, turn=turn)
        score_loss = weighted_mean(F.smooth_l1_loss(out["score"], score_t, reduction="none"), weight_t)
        bust_loss = weighted_mean(
            F.binary_cross_entropy_with_logits(out["bust_logit"], bust_t, reduction="none"),
            aux_weight_t,
        )
        fl_loss = weighted_mean(
            F.binary_cross_entropy_with_logits(out["fl_logit"], fl_t, reduction="none"),
            aux_weight_t,
        )
        fl_type_loss = weighted_mean(
            F.binary_cross_entropy_with_logits(out["fl_type_logits"], fl_type_t, reduction="none"),
            aux_weight_t,
        )
        losses.append(float(score_loss.item()))
        bust_losses.append(float(bust_loss.item()))
        fl_losses.append(float(fl_loss.item()))
        fl_type_losses.append(float(fl_type_loss.item()))
        pred = (out["score"].detach().cpu().numpy() * score_std) + score_mean
        true = (score_t.detach().cpu().numpy() * score_std) + score_mean
        pred_scores.append(pred)
        true_scores.append(true)

    pred_arr = np.concatenate(pred_scores) if pred_scores else np.array([], dtype=np.float32)
    true_arr = np.concatenate(true_scores) if true_scores else np.array([], dtype=np.float32)
    mae = float(np.mean(np.abs(pred_arr - true_arr))) if len(pred_arr) else 0.0
    return {
        "score_loss": float(np.mean(losses)) if losses else 0.0,
        "score_mae": mae,
        "score_corr": pearson_corr(pred_arr, true_arr),
        "bust_bce": float(np.mean(bust_losses)) if bust_losses else 0.0,
        "fl_bce": float(np.mean(fl_losses)) if fl_losses else 0.0,
        "fl_type_bce": float(np.mean(fl_type_losses)) if fl_type_losses else 0.0,
    }


@torch.no_grad()
def group_top1_metrics(
    model: ActionValueReranker,
    data_dir: Path,
    val_idx: np.ndarray,
    score_mean: float,
    score_std: float,
    device: torch.device,
    batch_size: int,
    max_samples: int,
) -> dict:
    if len(val_idx) == 0:
        return {
            "group_top1": 0.0,
            "group_top3": 0.0,
            "group_top5": 0.0,
            "group_top10": 0.0,
            "group_top15": 0.0,
            "group_top20": 0.0,
            "group_regret": 0.0,
            "group_top3_rerank_regret": 0.0,
            "group_top5_rerank_regret": 0.0,
            "group_top10_rerank_regret": 0.0,
            "group_top15_rerank_regret": 0.0,
            "group_top20_rerank_regret": 0.0,
            "groups": 0,
        }
    rng = np.random.default_rng(123)
    idx = val_idx
    if len(idx) > max_samples:
        idx = rng.choice(idx, size=max_samples, replace=False)
    idx = np.sort(idx.astype(np.int64, copy=False))

    states = np.load(data_dir / "states.npy", mmap_mode="r")
    scores = np.load(data_dir / "scores.npy", mmap_mode="r")
    group_ids = np.load(data_dir / "group_ids.npy", mmap_mode="r")
    turns = np.load(data_dir / "turns.npy", mmap_mode="r")

    model.eval()
    preds = np.zeros(len(idx), dtype=np.float32)
    for start in range(0, len(idx), batch_size):
        sel = idx[start:start + batch_size]
        batch = torch.from_numpy(np.array(states[sel], dtype=np.float32, copy=True)).to(device)
        turn = torch.from_numpy(np.asarray(turns[sel], dtype=np.int64)).to(device)
        out = model(batch, turn=turn)
        preds[start:start + len(sel)] = (
            out["score"].detach().cpu().numpy() * score_std + score_mean
        )

    grouped: dict[int, list[tuple[float, float]]] = {}
    true_scores = np.asarray(scores[idx], dtype=np.float32)
    gids = np.asarray(group_ids[idx], dtype=np.int64)
    for gid, pred, true in zip(gids, preds, true_scores):
        grouped.setdefault(int(gid), []).append((float(pred), float(true)))

    top_hits = {1: 0, 3: 0, 5: 0, 10: 0, 15: 0, 20: 0}
    regret = 0.0
    topk_rerank_regret = {3: 0.0, 5: 0.0, 10: 0.0, 15: 0.0, 20: 0.0}
    usable = 0
    for items in grouped.values():
        if len(items) < 2:
            continue
        true_best = max(range(len(items)), key=lambda i: items[i][1])
        ordered = sorted(range(len(items)), key=lambda i: items[i][0], reverse=True)
        pred_best = ordered[0]
        true_rank = ordered.index(true_best) + 1
        for k in top_hits:
            if true_rank <= min(k, len(items)):
                top_hits[k] += 1
        for k in topk_rerank_regret:
            topk_items = ordered[:min(k, len(items))]
            topk_best_true = max(items[i][1] for i in topk_items)
            topk_rerank_regret[k] += items[true_best][1] - topk_best_true
        regret += items[true_best][1] - items[pred_best][1]
        usable += 1

    return {
        "group_top1": top_hits[1] / usable if usable else 0.0,
        "group_top3": top_hits[3] / usable if usable else 0.0,
        "group_top5": top_hits[5] / usable if usable else 0.0,
        "group_top10": top_hits[10] / usable if usable else 0.0,
        "group_top15": top_hits[15] / usable if usable else 0.0,
        "group_top20": top_hits[20] / usable if usable else 0.0,
        "group_regret": regret / usable if usable else 0.0,
        "group_top3_rerank_regret": topk_rerank_regret[3] / usable if usable else 0.0,
        "group_top5_rerank_regret": topk_rerank_regret[5] / usable if usable else 0.0,
        "group_top10_rerank_regret": topk_rerank_regret[10] / usable if usable else 0.0,
        "group_top15_rerank_regret": topk_rerank_regret[15] / usable if usable else 0.0,
        "group_top20_rerank_regret": topk_rerank_regret[20] / usable if usable else 0.0,
        "groups": usable,
    }


def save_checkpoint(
    path: Path,
    model: ActionValueReranker,
    args: argparse.Namespace,
    score_mean: float,
    score_std: float,
    epoch: int,
    metrics: dict,
) -> None:
    ckpt = {
        "model_state_dict": model.state_dict(),
        "input_dim": int(model.input_proj[0].in_features),
        "model_config": {
            "input_dim": int(model.input_proj[0].in_features),
            "hidden": int(args.hidden),
            "n_blocks": int(args.n_blocks),
            "dropout": float(args.dropout),
            "fl_type_keys": list(FL_TYPE_KEYS),
            "turn_specific_heads": bool(args.turn_specific_heads),
            "turn_specific_adapters": bool(args.turn_specific_adapters),
            "adapter_dim": int(args.adapter_dim),
            "num_turns": int(args.num_turns),
        },
        "normalization": {
            "score_mean": float(score_mean),
            "score_std": float(score_std),
        },
        "epoch": int(epoch),
        "metrics": metrics,
    }
    torch.save(ckpt, path)


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    meta = load_metadata(data_dir)
    states_shape = np.load(data_dir / "states.npy", mmap_mode="r").shape
    n_samples = min(int(meta.get("n_samples", states_shape[0])), states_shape[0])
    train_idx, val_idx = split_indices(data_dir, n_samples, args.val_frac, args.seed)
    train_turns = parse_turns(args.train_turns)
    train_position = parse_position(args.train_position)
    train_idx = filter_indices_by_turn(data_dir, train_idx, train_turns)
    val_idx = filter_indices_by_turn(data_dir, val_idx, train_turns)
    train_idx = filter_indices_by_position(data_dir, train_idx, train_position)
    val_idx = filter_indices_by_position(data_dir, val_idx, train_position)
    if len(train_idx) == 0:
        raise ValueError("No training samples remain after turn/position filters")
    if len(val_idx) == 0:
        raise ValueError("No validation samples remain after turn/position filters")

    raw_scores = np.load(data_dir / "scores.npy", mmap_mode="r")
    checkpoint_norm = (
        load_checkpoint_normalization(args.init_checkpoint)
        if args.normalization_source == "checkpoint"
        else None
    )
    if args.normalization_source == "checkpoint" and checkpoint_norm is None:
        raise ValueError("--normalization-source checkpoint requires a checkpoint with score normalization")
    if checkpoint_norm is not None:
        score_mean, score_std = checkpoint_norm
    else:
        score_mean = float(np.asarray(raw_scores[train_idx]).mean())
        score_std = float(np.asarray(raw_scores[train_idx]).std())
        score_std = max(score_std, 1e-6)
    group_starts, group_ends = build_group_bounds(data_dir, n_samples)
    group_ids_arr = np.load(data_dir / "group_ids.npy", mmap_mode="r")
    train_group_ids = np.unique(np.asarray(group_ids_arr[train_idx], dtype=np.int64))
    train_group_ids = train_group_ids[train_group_ids < len(group_starts)]
    group_sample_weights = None
    group_weights_path = data_dir / "group_sample_weights.npy"
    if group_weights_path.exists():
        loaded_group_weights = np.asarray(np.load(group_weights_path, mmap_mode="r"), dtype=np.float32)
        if len(loaded_group_weights) < len(group_starts):
            raise ValueError(
                f"{group_weights_path} has {len(loaded_group_weights)} rows but "
                f"{len(group_starts)} groups are required"
            )
        group_sample_weights = loaded_group_weights[:len(group_starts)]
    rng = np.random.default_rng(args.seed + 1000)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Action-value reranker training")
    print(f"  data: {data_dir}")
    print(f"  samples: {n_samples:,} train={len(train_idx):,} val={len(val_idx):,}")
    if train_turns is not None:
        print(f"  train_turns: {sorted(train_turns)}")
    if train_position is not None:
        print(f"  train_position: {'btn' if train_position == 1 else 'bb'}")
    print(f"  input_dim: {states_shape[1]}")
    print(f"  score norm: mean={score_mean:+.3f} std={score_std:.3f}")
    print(f"  groups: {len(group_starts):,} train_groups={len(train_group_ids):,}")
    if group_sample_weights is not None:
        selected_group_weights = np.asarray(group_sample_weights[train_group_ids], dtype=np.float32)
        print(
            "  group_sample_weights: "
            f"mean={float(selected_group_weights.mean()):.3f} "
            f"max={float(selected_group_weights.max()):.3f}"
        )
    print(f"  device: {device}")

    train_ds = ActionValueDataset(data_dir, train_idx, score_mean, score_std)
    val_ds = ActionValueDataset(data_dir, val_idx, score_mean, score_std)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = ActionValueReranker(
        input_dim=int(states_shape[1]),
        hidden=args.hidden,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
        turn_specific_heads=args.turn_specific_heads,
        turn_specific_adapters=args.turn_specific_adapters,
        adapter_dim=args.adapter_dim,
        num_turns=args.num_turns,
    ).to(device)
    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        missing, unexpected, partial, shape_mismatched = load_compatible_state_dict(model, state_dict)
        if args.turn_specific_heads and any(key.startswith("turn_") for key in missing):
            model.initialize_turn_heads_from_global()
        print(
            f"  init checkpoint: {args.init_checkpoint} "
            f"(missing={len(missing)}, unexpected={len(unexpected)}, "
            f"partial={len(partial)}, shape_mismatched={len(shape_mismatched)})"
        )
        if partial:
            print(f"  partial loads: {partial[:3]}")
        if shape_mismatched:
            print(f"  skipped shape mismatches: {shape_mismatched[:3]}")
    model.score_mean = score_mean
    model.score_std = score_std
    if args.freeze_shared:
        frozen_prefixes = ("input_proj.", "blocks.", "trunk.", "score_head.", "bust_head.", "fl_head.", "fl_type_head.")
        for name, param in model.named_parameters():
            if name.startswith(frozen_prefixes):
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  freeze_shared: trainable_params={trainable:,}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters remain after freezing")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05
    )

    history = []
    best_metric = float("inf")
    best_epoch = 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = []
        for batch in train_loader:
            state = batch["state"].to(device, non_blocking=True)
            turn = batch["turn"].to(device, non_blocking=True)
            score_t = batch["score"].to(device, non_blocking=True)
            bust_t = batch["bust"].to(device, non_blocking=True)
            fl_t = batch["fl"].to(device, non_blocking=True)
            fl_type_t = batch["fl_types"].to(device, non_blocking=True)
            weight_t = batch["weight"].to(device, non_blocking=True)
            aux_weight_t = weight_t * batch.get("aux_label_mask", torch.ones_like(weight_t)).to(device, non_blocking=True)

            out = model(state, turn=turn)
            score_loss = weighted_mean(
                F.smooth_l1_loss(out["score"], score_t, reduction="none"),
                weight_t,
            )
            bust_loss = weighted_mean(
                F.binary_cross_entropy_with_logits(out["bust_logit"], bust_t, reduction="none"),
                aux_weight_t,
            )
            fl_loss = weighted_mean(
                F.binary_cross_entropy_with_logits(out["fl_logit"], fl_t, reduction="none"),
                aux_weight_t,
            )
            fl_type_loss = weighted_mean(
                F.binary_cross_entropy_with_logits(out["fl_type_logits"], fl_type_t, reduction="none"),
                aux_weight_t,
            )
            loss = (
                args.score_weight * score_loss
                + args.bust_weight * bust_loss
                + args.fl_weight * fl_loss
                + args.fl_type_weight * fl_type_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            running.append(float(loss.item()))

        rank_train_metrics = train_ranking_batches(
            model,
            optimizer,
            data_dir,
            train_group_ids,
            group_starts,
            group_ends,
            group_sample_weights,
            score_mean,
            score_std,
            device,
            args,
            rng,
        )
        scheduler.step()
        val_metrics = evaluate(model, val_loader, device, score_mean, score_std)
        group_metrics = group_top1_metrics(
            model,
            data_dir,
            val_idx,
            score_mean,
            score_std,
            device,
            args.batch_size * 4,
            args.eval_max_samples,
        )
        metrics = {
            "epoch": epoch,
            "train_loss": float(np.mean(running)) if running else 0.0,
            **rank_train_metrics,
            **val_metrics,
            **group_metrics,
            "elapsed_s": time.time() - start,
            "lr": float(scheduler.get_last_lr()[0]),
        }
        history.append(metrics)

        print(
            f"  epoch {epoch:03d}: loss={metrics['train_loss']:.4f} "
            f"rank={metrics['rank_train_loss']:.4f} "
            f"topk_loss={metrics['topk_train_loss']:.4f} "
            f"topk_margin={metrics['topk_margin_train_loss']:.4f} "
            f"mae={metrics['score_mae']:.3f} corr={metrics['score_corr']:.3f} "
            f"top1={metrics['group_top1']:.1%} top3={metrics['group_top3']:.1%} "
            f"top5={metrics['group_top5']:.1%} top10={metrics['group_top10']:.1%} "
            f"top15={metrics['group_top15']:.1%} top20={metrics['group_top20']:.1%} "
            f"regret={metrics['group_regret']:.3f} "
            f"t3reg={metrics['group_top3_rerank_regret']:.3f} "
            f"t20reg={metrics['group_top20_rerank_regret']:.3f} "
            f"bust_bce={metrics['bust_bce']:.3f} fl_bce={metrics['fl_bce']:.3f} "
            f"fl_type_bce={metrics['fl_type_bce']:.3f}",
            flush=True,
        )

        rank_metric = select_best_metric(metrics, args)
        if rank_metric < best_metric:
            best_metric = rank_metric
            best_epoch = epoch
            save_checkpoint(
                save_dir / "action_value_best.pt",
                model,
                args,
                score_mean,
                score_std,
                epoch,
                metrics,
            )

        with (save_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if args.max_seconds and time.time() - start >= args.max_seconds:
            print(f"  stopping after {time.time() - start:.0f}s due to --max-seconds")
            break

    final_metrics = history[-1] if history else {}
    save_checkpoint(
        save_dir / "action_value_final.pt",
        model,
        args,
        score_mean,
        score_std,
        int(final_metrics.get("epoch", 0)),
        final_metrics,
    )

    summary = {
        "data": str(data_dir),
        "save_dir": str(save_dir),
        "n_samples": n_samples,
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "train_position": None if train_position is None else ("btn" if train_position == 1 else "bb"),
        "input_dim": int(states_shape[1]),
        "device": str(device),
        "score_mean": score_mean,
        "score_std": score_std,
        "fl_type_keys": list(FL_TYPE_KEYS),
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "final": final_metrics,
        "args": vars(args),
    }
    with (save_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (save_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# Action-value Reranker Training\n\n")
        f.write(f"- data: `{data_dir}`\n")
        f.write(f"- samples: {n_samples:,}\n")
        if train_position is not None:
            f.write(f"- train_position: {'btn' if train_position == 1 else 'bb'}\n")
        f.write(f"- best_epoch: {best_epoch}\n")
        if final_metrics:
            f.write(f"- final score_mae: {final_metrics.get('score_mae', 0.0):.3f}\n")
            f.write(f"- final score_corr: {final_metrics.get('score_corr', 0.0):.3f}\n")
            f.write(f"- final group_top1: {final_metrics.get('group_top1', 0.0):.1%}\n")
            f.write(f"- final group_top3: {final_metrics.get('group_top3', 0.0):.1%}\n")
            f.write(f"- final group_top5: {final_metrics.get('group_top5', 0.0):.1%}\n")
            f.write(f"- final group_top10: {final_metrics.get('group_top10', 0.0):.1%}\n")
            f.write(f"- final group_top15: {final_metrics.get('group_top15', 0.0):.1%}\n")
            f.write(f"- final group_top20: {final_metrics.get('group_top20', 0.0):.1%}\n")
            f.write(f"- final group_regret: {final_metrics.get('group_regret', 0.0):.3f}\n")
            f.write(
                "- final group_top3_rerank_regret: "
                f"{final_metrics.get('group_top3_rerank_regret', 0.0):.3f}\n"
            )
            f.write(
                "- final group_top20_rerank_regret: "
                f"{final_metrics.get('group_top20_rerank_regret', 0.0):.3f}\n"
            )
            f.write(f"- final fl_type_bce: {final_metrics.get('fl_type_bce', 0.0):.3f}\n")

    print(f"\nSaved: {save_dir}")
    print(f"  best:  {save_dir / 'action_value_best.pt'}")
    print(f"  final: {save_dir / 'action_value_final.pt'}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train candidate action-value reranker")
    parser.add_argument("--data", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--init-checkpoint", default=None,
                        help="Optional existing reranker checkpoint to fine-tune")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--n-blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--turn-specific-heads", action="store_true",
                        help="Use separate output heads per turn while sharing the trunk")
    parser.add_argument("--turn-specific-adapters", action="store_true",
                        help="Use zero-initialized residual adapters per turn before the heads")
    parser.add_argument("--adapter-dim", type=int, default=64,
                        help="Bottleneck width for turn-specific adapters")
    parser.add_argument("--num-turns", type=int, default=5)
    parser.add_argument("--freeze-shared", action="store_true",
                        help="Freeze trunk and global heads; train only turn-specific heads")
    parser.add_argument("--normalization-source", choices=("data", "checkpoint"), default="data",
                        help="Use data score stats or reuse init checkpoint score normalization")
    parser.add_argument("--train-turns", default="",
                        help="Optional comma-separated turns to include in training batches")
    parser.add_argument("--train-position", default="all",
                        help="Optional position filter: all, bb/first/senkou, or btn/second/koukou")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--score-weight", type=float, default=1.0)
    parser.add_argument("--bust-weight", type=float, default=0.25)
    parser.add_argument("--fl-weight", type=float, default=0.25)
    parser.add_argument("--fl-type-weight", type=float, default=0.25)
    parser.add_argument("--ranking-weight", type=float, default=0.25)
    parser.add_argument("--ranking-temperature", type=float, default=2.0)
    parser.add_argument("--topk-rank-weight", type=float, default=0.0)
    parser.add_argument("--target-topk", type=int, default=20)
    parser.add_argument("--topk-rank-temperature", type=float, default=0.25)
    parser.add_argument("--topk-margin-weight", type=float, default=0.0,
                        help="Weight for the pruning-boundary margin loss")
    parser.add_argument("--topk-margin", type=float, default=0.05,
                        help="Normalized-score margin above the K-th competitor")
    parser.add_argument("--topk-margin-temperature", type=float, default=0.1,
                        help="Softplus temperature for top-k margin violations")
    parser.add_argument("--selection-metric", choices=("regret", "topk"), default="regret")
    parser.add_argument("--ranking-batches-per-epoch", type=int, default=64)
    parser.add_argument("--group-batch-size", type=int, default=32)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--eval-max-samples", type=int, default=200_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
