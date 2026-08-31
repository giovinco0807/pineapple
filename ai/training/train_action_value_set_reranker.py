"""Train a set/listwise reranker from candidate-level action-value data."""
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.models.action_value_set_reranker import ActionValueSetReranker


def load_metadata(data_dir: Path) -> dict:
    path = data_dir / "metadata.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def parse_position(value: str) -> int | None:
    text = value.strip().lower()
    if not text or text in {"all", "mixed"}:
        return None
    if text in {"btn", "button", "second"}:
        return 1
    if text in {"bb", "blind", "first"}:
        return 0
    raise ValueError("--train-position must be all, btn, or bb")


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    xx = x.astype(np.float64, copy=True)
    yy = y.astype(np.float64, copy=True)
    xx -= xx.mean()
    yy -= yy.mean()
    denom = math.sqrt(float((xx * xx).sum() * (yy * yy).sum()))
    if denom <= 1e-12:
        return 0.0
    return float((xx * yy).sum() / denom)


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


def split_group_indices(group_ids: np.ndarray, bounds: list[tuple[int, int]], val_frac: float, seed: int) -> tuple[set[int], set[int]]:
    unique_groups = np.unique(group_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    n_val_groups = max(1, int(len(unique_groups) * val_frac))
    val_ids = {int(x) for x in unique_groups[:n_val_groups]}
    train_ids = {int(x) for x in unique_groups[n_val_groups:]}
    valid_bound_ids = {int(group_ids[start]) for start, _ in bounds}
    return train_ids & valid_bound_ids, val_ids & valid_bound_ids


class CandidateGroupDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        group_indices: list[int],
        bounds: list[tuple[int, int]],
        score_mean: float,
        score_std: float,
    ):
        self.data_dir = Path(data_dir)
        self.group_indices = list(group_indices)
        self.bounds = bounds
        self.score_mean = float(score_mean)
        self.score_std = max(float(score_std), 1e-6)
        self.states = np.load(self.data_dir / "states.npy", mmap_mode="r")
        self.scores = np.load(self.data_dir / "scores.npy", mmap_mode="r")
        self.group_ids = np.load(self.data_dir / "group_ids.npy", mmap_mode="r")
        self.sample_weights = (
            np.load(self.data_dir / "sample_weights.npy", mmap_mode="r")
            if (self.data_dir / "sample_weights.npy").exists()
            else None
        )
        self.group_sample_weights = (
            np.load(self.data_dir / "group_sample_weights.npy", mmap_mode="r")
            if (self.data_dir / "group_sample_weights.npy").exists()
            else None
        )
        self.base_scores = (
            np.load(self.data_dir / "base_scores.npy", mmap_mode="r")
            if (self.data_dir / "base_scores.npy").exists()
            else None
        )

    def __len__(self) -> int:
        return len(self.group_indices)

    @property
    def input_dim(self) -> int:
        return int(self.states.shape[1]) + (1 if self.base_scores is not None else 0)

    def __getitem__(self, i: int) -> dict:
        group_index = int(self.group_indices[i])
        start, end = self.bounds[group_index]
        states = np.array(self.states[start:end], dtype=np.float32, copy=True)
        raw_scores = np.asarray(self.scores[start:end], dtype=np.float32)
        if self.base_scores is not None:
            base_scores = np.asarray(self.base_scores[start:end], dtype=np.float32)
            base_feature = ((base_scores - self.score_mean) / self.score_std).reshape(-1, 1)
            states = np.concatenate([states, base_feature.astype(np.float32, copy=False)], axis=1)
        scores = (raw_scores - self.score_mean) / self.score_std
        weights = (
            np.asarray(self.sample_weights[start:end], dtype=np.float32)
            if self.sample_weights is not None
            else np.ones(end - start, dtype=np.float32)
        )
        group_weight = (
            float(self.group_sample_weights[group_index])
            if self.group_sample_weights is not None
            else 1.0
        )
        return {
            "states": torch.from_numpy(states),
            "scores": torch.from_numpy(np.array(scores, dtype=np.float32, copy=True)),
            "raw_scores": torch.from_numpy(np.array(raw_scores, dtype=np.float32, copy=True)),
            "weights": torch.from_numpy(np.array(weights, dtype=np.float32, copy=True)),
            "group_weight": torch.tensor(group_weight, dtype=torch.float32),
            "group_index": torch.tensor(group_index, dtype=torch.long),
            "group_id": torch.tensor(int(self.group_ids[start]), dtype=torch.long),
        }


def collate_groups(batch: list[dict]) -> dict:
    batch_size = len(batch)
    max_len = max(int(item["states"].shape[0]) for item in batch)
    input_dim = int(batch[0]["states"].shape[1])
    states = torch.zeros((batch_size, max_len, input_dim), dtype=torch.float32)
    scores = torch.full((batch_size, max_len), -1e9, dtype=torch.float32)
    raw_scores = torch.full((batch_size, max_len), -1e9, dtype=torch.float32)
    weights = torch.zeros((batch_size, max_len), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    group_weight = torch.zeros(batch_size, dtype=torch.float32)
    group_index = torch.zeros(batch_size, dtype=torch.long)
    group_id = torch.zeros(batch_size, dtype=torch.long)
    for row, item in enumerate(batch):
        length = int(item["states"].shape[0])
        states[row, :length] = item["states"]
        scores[row, :length] = item["scores"]
        raw_scores[row, :length] = item["raw_scores"]
        weights[row, :length] = item["weights"]
        mask[row, :length] = True
        group_weight[row] = item["group_weight"]
        group_index[row] = item["group_index"]
        group_id[row] = item["group_id"]
    return {
        "states": states,
        "scores": scores,
        "raw_scores": raw_scores,
        "weights": weights,
        "mask": mask,
        "group_weight": group_weight,
        "group_index": group_index,
        "group_id": group_id,
    }


def listnet_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, temperature: float) -> torch.Tensor:
    masked_logits = logits.masked_fill(~mask, -1e9)
    masked_targets = targets.masked_fill(~mask, -1e9)
    target_probs = F.softmax(masked_targets / max(float(temperature), 1e-6), dim=-1)
    log_probs = F.log_softmax(masked_logits, dim=-1)
    loss = target_probs * (torch.log(target_probs.clamp(min=1e-10)) - log_probs)
    return loss.masked_fill(~mask, 0.0).sum(dim=-1)


def topk_rank_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    target_topk: int,
    temperature: float,
) -> torch.Tensor:
    if target_topk <= 0:
        return torch.zeros(logits.shape[0], dtype=logits.dtype, device=logits.device)
    lengths = mask.sum(dim=1).float()
    best_idx = targets.masked_fill(~mask, -1e9).argmax(dim=1)
    best_logits = logits.gather(1, best_idx[:, None])
    competitor_mask = mask.clone()
    competitor_mask.scatter_(1, best_idx[:, None], False)
    soft_above = torch.sigmoid((logits - best_logits) / max(float(temperature), 1e-6))
    soft_above = soft_above.masked_fill(~competitor_mask, 0.0).sum(dim=1)
    allowed = torch.minimum(
        torch.full_like(lengths, float(target_topk - 1)),
        torch.clamp(lengths - 1.0, min=0.0),
    )
    overflow = F.relu(soft_above - allowed)
    return overflow * overflow


def topk_margin_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    target_topk: int,
    margin: float,
    temperature: float,
) -> torch.Tensor:
    losses = torch.zeros(logits.shape[0], dtype=logits.dtype, device=logits.device)
    if target_topk <= 0:
        return losses
    best_idx = targets.masked_fill(~mask, -1e9).argmax(dim=1)
    best_logits = logits.gather(1, best_idx[:, None]).squeeze(1)
    competitor_mask = mask.clone()
    competitor_mask.scatter_(1, best_idx[:, None], False)
    eligible = competitor_mask.sum(dim=1) >= target_topk
    if not bool(eligible.any()):
        return losses
    competitor_logits = logits[eligible].masked_fill(~competitor_mask[eligible], -1e9)
    kth_competitor = torch.topk(competitor_logits, k=target_topk, dim=1).values[:, -1]
    violation = kth_competitor - best_logits[eligible] + float(margin)
    losses[eligible] = (
        F.softplus(violation / max(float(temperature), 1e-6))
        * max(float(temperature), 1e-6)
    )
    return losses


def pairwise_gap_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    raw_targets: torch.Tensor,
    mask: torch.Tensor,
    min_gap: float,
    gap_cap: float,
    margin_scale: float,
    min_margin: float,
    max_margin: float,
    temperature: float,
) -> torch.Tensor:
    """Penalize inversions where the teacher EV gap is meaningfully large.

    The model logits and ``targets`` are normalized scores, while ``raw_targets``
    stay in EV units.  ``min_gap`` and ``gap_cap`` are therefore specified in EV
    units so this loss can focus on mistakes that cost real value instead of
    near-ties.
    """
    if min_gap <= 0 and gap_cap <= 0:
        return torch.zeros(logits.shape[0], dtype=logits.dtype, device=logits.device)
    valid_pair = mask[:, :, None] & mask[:, None, :]
    raw_gap = raw_targets[:, :, None] - raw_targets[:, None, :]
    pair_mask = valid_pair & (raw_gap >= float(min_gap))
    if not bool(pair_mask.any()):
        return torch.zeros(logits.shape[0], dtype=logits.dtype, device=logits.device)

    norm_gap = targets[:, :, None] - targets[:, None, :]
    desired_margin = torch.clamp(
        norm_gap * float(margin_scale),
        min=float(min_margin),
        max=float(max_margin),
    )
    violation = logits[:, None, :] - logits[:, :, None] + desired_margin
    temp = max(float(temperature), 1e-6)
    pair_loss = F.softplus(violation / temp) * temp

    capped_gap = torch.clamp(raw_gap, min=0.0, max=max(float(gap_cap), float(min_gap)))
    pair_weight = torch.clamp(capped_gap / max(float(gap_cap), 1e-6), min=0.05)
    pair_loss = pair_loss * pair_weight
    pair_loss = pair_loss.masked_fill(~pair_mask, 0.0)
    denom = pair_weight.masked_fill(~pair_mask, 0.0).sum(dim=(1, 2)).clamp(min=1e-6)
    return pair_loss.sum(dim=(1, 2)) / denom


def weighted_row_mean(loss: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight.to(device=loss.device, dtype=loss.dtype)
    return (loss * weight).sum() / weight.sum().clamp(min=1e-6)


def score_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    loss = F.smooth_l1_loss(logits, targets, reduction="none").masked_fill(~mask, 0.0)
    weights = weights.to(loss.device).masked_fill(~mask, 0.0)
    return (loss * weights).sum() / weights.sum().clamp(min=1e-6)


@torch.no_grad()
def evaluate(
    model: ActionValueSetReranker,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    topks = (1, 3, 5, 10, 15, 20)
    hits = {k: 0 for k in topks}
    regrets = {k: 0.0 for k in topks}
    pred_top1_regret = 0.0
    pred_scores_all: list[np.ndarray] = []
    true_scores_all: list[np.ndarray] = []
    groups = 0

    for batch in loader:
        states = batch["states"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        raw_scores = batch["raw_scores"].to(device, non_blocking=True)
        pred = model.predict_scores(states, mask)
        for row in range(states.shape[0]):
            valid = mask[row]
            true = raw_scores[row, valid].detach().cpu().numpy().astype(np.float64)
            predicted = pred[row, valid].detach().cpu().numpy().astype(np.float64)
            if len(true) < 1:
                continue
            true_best = int(np.argmax(true))
            order = np.argsort(-predicted)
            pred_best = int(order[0])
            rank = int(np.where(order == true_best)[0][0]) + 1
            best_score = float(true[true_best])
            pred_top1_regret += max(0.0, best_score - float(true[pred_best]))
            for k in topks:
                kk = min(k, len(order))
                if rank <= kk:
                    hits[k] += 1
                regrets[k] += max(0.0, best_score - float(true[order[:kk]].max()))
            pred_scores_all.append(predicted)
            true_scores_all.append(true)
            groups += 1

    denom = max(groups, 1)
    pred_arr = np.concatenate(pred_scores_all) if pred_scores_all else np.array([], dtype=np.float32)
    true_arr = np.concatenate(true_scores_all) if true_scores_all else np.array([], dtype=np.float32)
    metrics = {
        "groups": int(groups),
        "score_mae": float(np.mean(np.abs(pred_arr - true_arr))) if len(pred_arr) else 0.0,
        "score_corr": pearson_corr(pred_arr, true_arr),
        "group_regret": pred_top1_regret / denom,
    }
    for k in topks:
        metrics[f"group_top{k}"] = hits[k] / denom
        metrics[f"group_top{k}_rerank_regret"] = regrets[k] / denom
    return metrics


def select_metric(metrics: dict, target_topk: int, selection_topk: int = 0) -> float:
    if selection_topk > 0:
        k = int(selection_topk)
    elif target_topk <= 1:
        k = 1
    elif target_topk <= 3:
        k = 3
    elif target_topk <= 5:
        k = 5
    elif target_topk <= 10:
        k = 10
    elif target_topk <= 15:
        k = 15
    else:
        k = 20
    return (
        float(metrics.get(f"group_top{k}_rerank_regret", metrics.get("group_regret", 0.0)))
        + (1.0 - float(metrics.get(f"group_top{k}", 0.0))) * 10.0
        + float(metrics.get("score_mae", 0.0)) * 0.01
    )


def save_checkpoint(
    path: Path,
    model: ActionValueSetReranker,
    args: argparse.Namespace,
    score_mean: float,
    score_std: float,
    epoch: int,
    metrics: dict,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "input_dim": int(model.input_dim),
                "hidden": int(args.hidden),
                "n_blocks": int(args.n_blocks),
                "n_layers": int(args.n_layers),
                "n_heads": int(args.n_heads),
                "dropout": float(args.dropout),
                "base_score_residual": bool(args.base_score_residual),
            },
            "normalization": {
                "score_mean": float(score_mean),
                "score_std": float(score_std),
            },
            "epoch": int(epoch),
            "metrics": metrics,
        },
        path,
    )


def group_indices_for_position(
    bounds: list[tuple[int, int]],
    group_ids: np.ndarray,
    positions: np.ndarray | None,
    train_ids: set[int],
    val_ids: set[int],
    selected_position: int | None,
) -> tuple[list[int], list[int]]:
    train: list[int] = []
    val: list[int] = []
    for group_index, (start, end) in enumerate(bounds):
        if selected_position is not None:
            if positions is None:
                raise FileNotFoundError("positions.npy is required when --train-position is set")
            group_positions = np.asarray(positions[start:end], dtype=np.int8)
            if not bool(np.all(group_positions == int(selected_position))):
                continue
        group_id = int(group_ids[start])
        if group_id in train_ids:
            train.append(group_index)
        elif group_id in val_ids:
            val.append(group_index)
    return train, val


def make_loader(
    dataset: CandidateGroupDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    weighted: bool,
) -> DataLoader:
    sampler = None
    if weighted and dataset.group_sample_weights is not None:
        weights = np.asarray(dataset.group_sample_weights[dataset.group_indices], dtype=np.float64)
        weights = np.clip(weights, 0.0, None)
        if float(weights.sum()) > 0.0:
            sampler = WeightedRandomSampler(
                torch.as_tensor(weights, dtype=torch.double),
                num_samples=len(dataset),
                replacement=True,
            )
            shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_groups,
    )


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    meta = load_metadata(data_dir)
    states_shape = np.load(data_dir / "states.npy", mmap_mode="r").shape
    n_samples = min(int(meta.get("n_samples", states_shape[0])), int(states_shape[0]))
    group_ids = np.asarray(np.load(data_dir / "group_ids.npy", mmap_mode="r")[:n_samples], dtype=np.int64)
    bounds = build_group_bounds(group_ids)
    positions_path = data_dir / "positions.npy"
    positions = (
        np.asarray(np.load(positions_path, mmap_mode="r")[:n_samples], dtype=np.int8)
        if positions_path.exists()
        else None
    )
    train_ids, val_ids = split_group_indices(group_ids, bounds, args.val_frac, args.seed)
    selected_position = parse_position(args.train_position)
    train_groups, val_groups = group_indices_for_position(
        bounds,
        group_ids,
        positions,
        train_ids,
        val_ids,
        selected_position,
    )
    if not train_groups or not val_groups:
        raise ValueError("No train/validation groups remain after filters")

    init_ckpt = None
    if args.init_checkpoint:
        init_ckpt = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(init_ckpt, dict) or "model_state_dict" not in init_ckpt:
            raise ValueError(f"--init-checkpoint is not an action-value set checkpoint: {args.init_checkpoint}")

    raw_scores = np.load(data_dir / "scores.npy", mmap_mode="r")
    train_sample_indices = np.concatenate(
        [np.arange(bounds[g][0], bounds[g][1], dtype=np.int64) for g in train_groups]
    )
    score_mean = float(np.asarray(raw_scores[train_sample_indices]).mean())
    score_std = max(float(np.asarray(raw_scores[train_sample_indices]).std()), 1e-6)
    if init_ckpt is not None and args.normalization_source == "checkpoint":
        norm = init_ckpt.get("normalization", {})
        score_mean = float(norm.get("score_mean", init_ckpt.get("score_mean", score_mean)))
        score_std = max(float(norm.get("score_std", init_ckpt.get("score_std", score_std))), 1e-6)

    train_ds = CandidateGroupDataset(data_dir, train_groups, bounds, score_mean, score_std)
    val_ds = CandidateGroupDataset(data_dir, val_groups, bounds, score_mean, score_std)
    train_loader = make_loader(
        train_ds,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        weighted=args.weighted_groups,
    )
    val_loader = make_loader(
        val_ds,
        args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        weighted=False,
    )

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = ActionValueSetReranker(
        input_dim=int(train_ds.input_dim),
        hidden=args.hidden,
        n_blocks=args.n_blocks,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        base_score_residual=args.base_score_residual,
    ).to(device)
    model.score_mean = score_mean
    model.score_std = score_std
    if init_ckpt is not None:
        ckpt_config = init_ckpt.get("model_config", {})
        expected = {
            "input_dim": int(train_ds.input_dim),
            "hidden": int(args.hidden),
            "n_blocks": int(args.n_blocks),
            "n_layers": int(args.n_layers),
            "n_heads": int(args.n_heads),
            "base_score_residual": bool(args.base_score_residual),
        }
        for key, value in expected.items():
            if key in ckpt_config and ckpt_config[key] != value:
                raise ValueError(
                    f"--init-checkpoint {key}={ckpt_config[key]!r} does not match requested {value!r}"
                )
        model.load_state_dict(init_ckpt["model_state_dict"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.lr * 0.05,
    )

    print("Action-value set reranker training")
    print(f"  data: {data_dir}")
    print(f"  groups: train={len(train_groups):,} val={len(val_groups):,}")
    print(f"  samples: {n_samples:,}")
    print(f"  train_position: {args.train_position}")
    print(f"  input_dim: {train_ds.input_dim}")
    print(f"  base_scores: {(data_dir / 'base_scores.npy').exists()}")
    print(f"  base_score_residual: {args.base_score_residual}")
    print(f"  score norm: mean={score_mean:+.3f} std={score_std:.3f}")
    if args.init_checkpoint:
        print(f"  init_checkpoint: {args.init_checkpoint}")
        print(f"  normalization_source: {args.normalization_source}")
    print(f"  model: hidden={args.hidden} layers={args.n_layers} heads={args.n_heads}")
    print(f"  device: {device}")

    history: list[dict] = []
    best_metric = float("inf")
    best_epoch = 0
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        rank_losses: list[float] = []
        topk_losses: list[float] = []
        margin_losses: list[float] = []
        pairwise_losses: list[float] = []
        score_losses: list[float] = []
        for batch in train_loader:
            states = batch["states"].to(device, non_blocking=True)
            targets = batch["scores"].to(device, non_blocking=True)
            raw_targets = batch["raw_scores"].to(device, non_blocking=True)
            weights = batch["weights"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            group_weight = batch["group_weight"].to(device, non_blocking=True)
            logits = model(states, mask)
            s_loss = score_loss(logits, targets, mask, weights)
            r_loss_rows = listnet_loss(logits, targets, mask, args.ranking_temperature)
            t_loss_rows = topk_rank_loss(
                logits,
                targets,
                mask,
                args.target_topk,
                args.topk_rank_temperature,
            )
            m_loss_rows = topk_margin_loss(
                logits,
                targets,
                mask,
                args.target_topk,
                args.topk_margin,
                args.topk_margin_temperature,
            )
            p_loss_rows = pairwise_gap_loss(
                logits,
                targets,
                raw_targets,
                mask,
                args.pairwise_min_gap,
                args.pairwise_gap_cap,
                args.pairwise_margin_scale,
                args.pairwise_min_margin,
                args.pairwise_max_margin,
                args.pairwise_temperature,
            )
            r_loss = weighted_row_mean(r_loss_rows, group_weight)
            t_loss = weighted_row_mean(t_loss_rows, group_weight)
            m_loss = weighted_row_mean(m_loss_rows, group_weight)
            p_loss = weighted_row_mean(p_loss_rows, group_weight)
            loss = (
                args.score_weight * s_loss
                + args.ranking_weight * r_loss
                + args.topk_rank_weight * t_loss
                + args.topk_margin_weight * m_loss
                + args.pairwise_weight * p_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_losses.append(float(loss.item()))
            score_losses.append(float(s_loss.item()))
            rank_losses.append(float(r_loss.item()))
            topk_losses.append(float(t_loss.item()))
            margin_losses.append(float(m_loss.item()))
            pairwise_losses.append(float(p_loss.item()))
        scheduler.step()

        metrics = evaluate(model, val_loader, device)
        metrics.update(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
                "score_loss": float(np.mean(score_losses)) if score_losses else 0.0,
                "rank_train_loss": float(np.mean(rank_losses)) if rank_losses else 0.0,
                "topk_train_loss": float(np.mean(topk_losses)) if topk_losses else 0.0,
                "topk_margin_train_loss": float(np.mean(margin_losses)) if margin_losses else 0.0,
                "pairwise_train_loss": float(np.mean(pairwise_losses)) if pairwise_losses else 0.0,
                "elapsed_s": time.time() - start_time,
                "lr": float(scheduler.get_last_lr()[0]),
            }
        )
        history.append(metrics)
        print(
            f"  epoch {epoch:03d}: loss={metrics['train_loss']:.4f} "
            f"rank={metrics['rank_train_loss']:.4f} "
            f"topk={metrics['topk_train_loss']:.4f} "
            f"margin={metrics['topk_margin_train_loss']:.4f} "
            f"pair={metrics['pairwise_train_loss']:.4f} "
            f"mae={metrics['score_mae']:.3f} corr={metrics['score_corr']:.3f} "
            f"top1={metrics['group_top1']:.1%} top3={metrics['group_top3']:.1%} "
            f"top5={metrics['group_top5']:.1%} top10={metrics['group_top10']:.1%} "
            f"top15={metrics['group_top15']:.1%} regret={metrics['group_regret']:.3f}",
            flush=True,
        )
        metric = select_metric(metrics, args.target_topk, args.selection_topk)
        if metric < best_metric:
            best_metric = metric
            best_epoch = epoch
            save_checkpoint(
                save_dir / "action_value_set_best.pt",
                model,
                args,
                score_mean,
                score_std,
                epoch,
                metrics,
            )
        with (save_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        if args.max_seconds and time.time() - start_time >= args.max_seconds:
            print(f"  stopping after {time.time() - start_time:.0f}s due to --max-seconds")
            break

    final = history[-1] if history else {}
    save_checkpoint(
        save_dir / "action_value_set_final.pt",
        model,
        args,
        score_mean,
        score_std,
        int(final.get("epoch", 0)),
        final,
    )
    summary = {
        "data": str(data_dir),
        "save_dir": str(save_dir),
        "train_position": args.train_position,
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "n_samples": n_samples,
        "input_dim": int(train_ds.input_dim),
        "device": str(device),
        "score_mean": score_mean,
        "score_std": score_std,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "final": final,
        "args": vars(args),
    }
    with (save_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {save_dir}")
    print(f"  best:  {save_dir / 'action_value_set_best.pt'}")
    print(f"  final: {save_dir / 'action_value_set_final.pt'}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train set/listwise action-value reranker")
    parser.add_argument("--data", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--normalization-source", choices=("data", "checkpoint"), default="data")
    parser.add_argument("--train-position", default="all")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--base-score-residual", action="store_true",
                        help="Add the final input feature as a base normalized score residual")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--score-weight", type=float, default=0.35)
    parser.add_argument("--ranking-weight", type=float, default=0.7)
    parser.add_argument("--ranking-temperature", type=float, default=2.0)
    parser.add_argument("--topk-rank-weight", type=float, default=1.0)
    parser.add_argument("--target-topk", type=int, default=3)
    parser.add_argument("--selection-topk", type=int, default=0,
                        help="Checkpoint selection K; defaults to target-topk buckets. Use 1 for Top1 work.")
    parser.add_argument("--topk-rank-temperature", type=float, default=0.18)
    parser.add_argument("--topk-margin-weight", type=float, default=0.25)
    parser.add_argument("--topk-margin", type=float, default=0.10)
    parser.add_argument("--topk-margin-temperature", type=float, default=0.08)
    parser.add_argument("--pairwise-weight", type=float, default=0.0)
    parser.add_argument("--pairwise-min-gap", type=float, default=0.25)
    parser.add_argument("--pairwise-gap-cap", type=float, default=8.0)
    parser.add_argument("--pairwise-margin-scale", type=float, default=0.5)
    parser.add_argument("--pairwise-min-margin", type=float, default=0.02)
    parser.add_argument("--pairwise-max-margin", type=float, default=1.5)
    parser.add_argument("--pairwise-temperature", type=float, default=0.10)
    parser.add_argument("--weighted-groups", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260605)
    args = parser.parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
