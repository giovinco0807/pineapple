"""Train a neural T2 reranker inside a selector-feature TopK union pool.

The current selector-feature and pairwise pool models use tree learners.  This
script keeps the same runtime-available pool construction, then learns a
group-wise neural scorer for candidates inside that pool.  Teacher EV is used
only as the supervised target.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.training.evaluate_action_value_reranker import parse_topks
from ai.training.evaluate_action_value_set_reranker import summarize_groups
from ai.training.evaluate_t2_selector_feature_union import FeatureSelectorSpec
from ai.training.train_t2_selector_feature_pool_reranker import (
    compact_metrics,
    feature_matrix,
    load_dataset,
    pool_local_features,
    union_ceiling,
    union_members,
)
from ai.training.train_t2_selector_feature_ranker import DataSpec, ModelSpec, SelectorData, parse_selector


SELECTOR_DEFINITIONS = {
    "old_pair_g115": "old_pair_g115=old_resid_l31+old_score_l31,1.15",
    "old_all4_g130": "old_all4_g130=old_resid_l31+old_score_l31+old_score_l15+old_extra_d10,1.30",
    "new_score_g115": "new_score_g115=new_score_l31,1.15",
    "new_score_g100": "new_score_g100=new_score_l31,1.00",
    "new_resid_g115": "new_resid_g115=new_resid_l31,1.15",
}


@dataclass
class PoolGroup:
    dataset: str
    group_index: int
    indices: np.ndarray
    x: np.ndarray
    scores: np.ndarray


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.scale).astype(np.float32)


class MLPScorer(nn.Module):
    def __init__(self, input_dim: int, hidden: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for width in hidden:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.LayerNorm(width))
            layers.append(nn.SiLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class AttentionScorer(nn.Module):
    """Small set-aware scorer for candidates inside one union pool."""

    def __init__(self, input_dim: int, d_model: int, num_heads: int, num_layers: int, dropout: float):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=max(d_model * 4, 64),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, max(d_model // 2, 32)),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(max(d_model // 2, 32), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.input(x).unsqueeze(0)
        encoded = self.encoder(encoded).squeeze(0)
        return self.output(encoded).squeeze(-1)


def parse_hidden(value: str) -> list[int]:
    out = [int(part) for part in value.lower().replace("x", ",").split(",") if part.strip()]
    if not out:
        raise ValueError("hidden layers must not be empty")
    return out


def load_summary_specs(summary_path: Path) -> tuple[list[DataSpec], list[DataSpec], dict[str, ModelSpec], list, list, list[FeatureSelectorSpec]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    train_specs = [DataSpec(name=name, path=Path(path)) for name, path in summary["train_data"].items()]
    eval_specs = [DataSpec(name=name, path=Path(path)) for name, path in summary["eval_data"].items()]
    model_specs = {
        name: ModelSpec(name=name, path=Path(item["path"]), target=item["target"])
        for name, item in summary["models"].items()
    }
    lgbm_names = set(summary.get("lgbm_selectors", {}))
    feature_names = set(summary.get("feature_selectors", {}))
    selectors = []
    for name in summary.get("selectors", []):
        if name == "base_scores" or name in lgbm_names or name in feature_names:
            continue
        if name not in SELECTOR_DEFINITIONS:
            raise ValueError(f"unknown selector definition for {name!r}; add it to SELECTOR_DEFINITIONS")
        selectors.append(parse_selector(SELECTOR_DEFINITIONS[name]))
    lgbm_selectors = [(name, Path(path)) for name, path in summary.get("lgbm_selectors", {}).items()]
    feature_selectors = [
        FeatureSelectorSpec(name=name, path=Path(item["path"]), scope=item["scope"])
        for name, item in summary.get("feature_selectors", {}).items()
    ]
    return train_specs, eval_specs, model_specs, selectors, lgbm_selectors, feature_selectors


def load_sets(
    specs: list[DataSpec],
    model_specs: dict[str, ModelSpec],
    selectors: list,
    lgbm_selectors: list,
    feature_selectors: list[FeatureSelectorSpec],
) -> list[SelectorData]:
    loaded_models = {name: joblib.load(spec.path) for name, spec in model_specs.items()}
    loaded_lgbm = {name: joblib.load(path) for name, path in lgbm_selectors}
    loaded_feature = {spec.name: joblib.load(spec.path) for spec in feature_selectors}
    return [
        load_dataset(
            spec,
            model_specs,
            selectors,
            lgbm_selectors,
            loaded_models,
            loaded_lgbm,
            feature_selectors,
            loaded_feature,
            include_base=True,
        )
        for spec in specs
    ]


def build_groups(
    datasets: list[SelectorData],
    pool_k: int,
    feature_scope: str,
    include_pool_local: bool,
) -> list[PoolGroup]:
    groups: list[PoolGroup] = []
    for ds in datasets:
        x_all = feature_matrix(ds, feature_scope)
        for group_index, (start, end) in enumerate(ds.bounds):
            members = union_members(ds, start, end, pool_k)
            if not members:
                continue
            idx = np.asarray(members, dtype=np.int64)
            x = x_all[idx].astype(np.float32)
            if include_pool_local:
                x = np.concatenate([x, pool_local_features(ds, members)], axis=1).astype(np.float32)
            groups.append(
                PoolGroup(
                    dataset=ds.spec.name,
                    group_index=group_index,
                    indices=idx,
                    x=x,
                    scores=np.asarray(ds.scores[idx], dtype=np.float32),
                )
            )
    return groups


def make_standardizer(groups: list[PoolGroup]) -> Standardizer:
    x = np.concatenate([group.x for group in groups], axis=0).astype(np.float64)
    mean = x.mean(axis=0).astype(np.float32)
    scale = x.std(axis=0).astype(np.float32)
    scale[scale < 1e-4] = 1.0
    return Standardizer(mean=mean, scale=scale)


def split_groups(groups: list[PoolGroup], val_frac: float, seed: int) -> tuple[list[PoolGroup], list[PoolGroup]]:
    order = list(range(len(groups)))
    rng = random.Random(seed)
    rng.shuffle(order)
    val_count = max(1, int(round(len(order) * val_frac))) if val_frac > 0 else 0
    val_ids = set(order[:val_count])
    train = [group for idx, group in enumerate(groups) if idx not in val_ids]
    val = [group for idx, group in enumerate(groups) if idx in val_ids]
    return train, val


def group_loss(
    pred: torch.Tensor,
    scores: torch.Tensor,
    *,
    hard_weight: float,
    soft_weight: float,
    pairwise_weight: float,
    mse_weight: float,
    temperature: float,
    ev_scale: float,
    margin: float,
) -> torch.Tensor:
    best = torch.max(scores)
    gap = scores - best
    best_index = torch.argmax(scores).view(1)
    loss = pred.sum() * 0.0
    if hard_weight:
        loss = loss + float(hard_weight) * F.cross_entropy(pred.view(1, -1), best_index)
    if soft_weight:
        target = torch.softmax(gap / max(float(temperature), 1e-4), dim=0)
        logp = torch.log_softmax(pred, dim=0)
        loss = loss + float(soft_weight) * (-(target * logp).sum())
    if pairwise_weight and len(pred) > 1:
        best_pred = pred[best_index.item()]
        ev_loss = torch.clamp(best - scores, min=0.0, max=max(float(ev_scale), 1e-4))
        pair_weights = 1.0 + ev_loss / max(float(ev_scale), 1e-4)
        hinge = F.softplus(float(margin) - (best_pred - pred))
        hinge = hinge * (torch.arange(len(pred), device=pred.device) != best_index.item()).float()
        denom = torch.clamp((torch.arange(len(pred), device=pred.device) != best_index.item()).float().sum(), min=1.0)
        loss = loss + float(pairwise_weight) * ((hinge * pair_weights).sum() / denom)
    if mse_weight:
        target_gap = torch.clamp(gap / max(float(ev_scale), 1e-4), min=-4.0, max=0.0)
        pred_centered = pred - torch.max(pred)
        loss = loss + float(mse_weight) * F.smooth_l1_loss(pred_centered, target_gap)
    return loss


@torch.no_grad()
def predict_group(model: nn.Module, group: PoolGroup, standardizer: Standardizer, device: torch.device) -> np.ndarray:
    model.eval()
    x = torch.from_numpy(standardizer.transform(group.x)).to(device)
    return model(x).detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def evaluate_group_list(model: nn.Module, groups: list[PoolGroup], standardizer: Standardizer, device: torch.device) -> dict[str, float]:
    hits1 = hits3 = hits5 = 0
    reg1: list[float] = []
    reg3: list[float] = []
    reg5: list[float] = []
    for group in groups:
        pred = predict_group(model, group, standardizer, device)
        scores = group.scores
        best = float(np.max(scores))
        order = np.argsort(-pred)
        hits1 += int(abs(float(scores[order[0]]) - best) <= 1e-6)
        hits3 += int(any(abs(float(scores[i]) - best) <= 1e-6 for i in order[: min(3, len(order))]))
        hits5 += int(any(abs(float(scores[i]) - best) <= 1e-6 for i in order[: min(5, len(order))]))
        reg1.append(max(0.0, best - float(scores[order[0]])))
        reg3.append(max(0.0, best - float(max(scores[i] for i in order[: min(3, len(order))]))))
        reg5.append(max(0.0, best - float(max(scores[i] for i in order[: min(5, len(order))]))))
    n = max(len(groups), 1)
    return {
        "groups": int(len(groups)),
        "top1": float(hits1 / n),
        "top3": float(hits3 / n),
        "top5": float(hits5 / n),
        "reg1": float(np.mean(reg1)) if reg1 else 0.0,
        "reg3": float(np.mean(reg3)) if reg3 else 0.0,
        "reg5": float(np.mean(reg5)) if reg5 else 0.0,
    }


def validation_key(metrics: dict[str, float]) -> tuple[float, float, float]:
    return (float(metrics["top1"]), -float(metrics["reg1"]), float(metrics["top3"]))


def make_model(args: argparse.Namespace, input_dim: int, hidden: list[int]) -> nn.Module:
    if args.architecture == "mlp":
        return MLPScorer(input_dim, hidden, args.dropout)
    if args.architecture == "attention":
        return AttentionScorer(input_dim, args.d_model, args.num_heads, args.num_layers, args.dropout)
    raise ValueError(f"unknown architecture: {args.architecture}")


def train_one(args: argparse.Namespace, groups: list[PoolGroup], standardizer: Standardizer, hidden: list[int], seed: int) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    train_groups, val_groups = split_groups(groups, args.val_frac, seed)
    model = make_model(args, groups[0].x.shape[1], hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    best_metrics = {"top1": -1.0, "top3": 0.0, "top5": 0.0, "reg1": 1.0e9, "reg3": 1.0e9, "reg5": 1.0e9}
    best_epoch = 0
    no_improve = 0
    history: list[dict[str, Any]] = []
    order = list(range(len(train_groups)))
    for epoch in range(1, args.epochs + 1):
        random.shuffle(order)
        model.train()
        losses: list[float] = []
        for start in range(0, len(order), args.batch_groups):
            opt.zero_grad(set_to_none=True)
            loss = None
            for group_idx in order[start : start + args.batch_groups]:
                group = train_groups[group_idx]
                x = torch.from_numpy(standardizer.transform(group.x)).to(device)
                scores = torch.from_numpy(group.scores).to(device)
                pred = model(x)
                group_loss_value = group_loss(
                    pred,
                    scores,
                    hard_weight=args.hard_weight,
                    soft_weight=args.soft_weight,
                    pairwise_weight=args.pairwise_weight,
                    mse_weight=args.mse_weight,
                    temperature=args.temperature,
                    ev_scale=args.ev_scale,
                    margin=args.margin,
                )
                loss = group_loss_value if loss is None else loss + group_loss_value
            assert loss is not None
            loss = loss / max(1, min(args.batch_groups, len(order) - start))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = evaluate_group_list(model, val_groups or train_groups, standardizer, device)
            metrics["epoch"] = epoch
            metrics["loss"] = float(np.mean(losses)) if losses else 0.0
            history.append(metrics)
            if validation_key(metrics) > validation_key(best_metrics):
                best_metrics = metrics
                best_epoch = epoch
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += args.eval_every
            if args.patience > 0 and no_improve >= args.patience:
                break
    model.load_state_dict(best_state)
    return model, {
        "seed": int(seed),
        "architecture": args.architecture,
        "hidden": hidden,
        "d_model": int(args.d_model),
        "num_heads": int(args.num_heads),
        "num_layers": int(args.num_layers),
        "device": str(device),
        "train_groups": int(len(train_groups)),
        "val_groups": int(len(val_groups)),
        "best_epoch": int(best_epoch),
        "best_validation": best_metrics,
        "history_tail": history[-8:],
    }


@torch.no_grad()
def predict_dataset(
    ds: SelectorData,
    model: nn.Module,
    standardizer: Standardizer,
    pool_k: int,
    feature_scope: str,
    include_pool_local: bool,
    device: torch.device,
) -> np.ndarray:
    x_all = feature_matrix(ds, feature_scope)
    pred = np.full(len(ds.scores), -1.0e9, dtype=np.float32)
    model.eval()
    for start, end in ds.bounds:
        members = union_members(ds, start, end, pool_k)
        if not members:
            continue
        idx = np.asarray(members, dtype=np.int64)
        local_x = x_all[idx].astype(np.float32)
        if include_pool_local:
            local_x = np.concatenate([local_x, pool_local_features(ds, members)], axis=1).astype(np.float32)
        x = torch.from_numpy(standardizer.transform(local_x)).to(device)
        pred[idx] = model(x).detach().cpu().numpy().astype(np.float32)
    return pred


def evaluate_pred(datasets: list[SelectorData], pred_by_dataset: dict[str, np.ndarray], topks: list[int]) -> tuple[dict, dict]:
    all_scores: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_bounds: list[tuple[int, int]] = []
    per_dataset: dict[str, dict] = {}
    offset = 0
    for ds in datasets:
        pred = pred_by_dataset[ds.spec.name]
        metrics, _rows = summarize_groups(ds.scores, pred, ds.bounds, topks)
        per_dataset[ds.spec.name] = compact_metrics(metrics)
        all_scores.append(ds.scores)
        all_pred.append(pred)
        for start, end in ds.bounds:
            all_bounds.append((start + offset, end + offset))
        offset += len(ds.scores)
    aggregate, _rows = summarize_groups(np.concatenate(all_scores), np.concatenate(all_pred), all_bounds, topks)
    return compact_metrics(aggregate), per_dataset


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# T2 Neural Pool Reranker",
        "",
        f"- pool_k: `{summary['pool_k']}`",
        f"- feature_scope: `{summary['feature_scope']}`",
        f"- train_groups: `{summary['train_groups']}`",
        f"- eval_groups: `{summary['eval_groups']}`",
        f"- input_dim: `{summary['input_dim']}`",
        "",
        "| model | seed | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg3 | Reg10 | best epoch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["eval_rows"]:
        m = row["aggregate"]
        lines.append(
            f"| {row['model']} | {row['seed']} | "
            f"{float(m.get('group_top1', 0.0)):.1%} | "
            f"{float(m.get('group_top3', 0.0)):.1%} | "
            f"{float(m.get('group_top5', 0.0)):.1%} | "
            f"{float(m.get('group_top10', 0.0)):.1%} | "
            f"{float(m.get('group_top20', 0.0)):.1%} | "
            f"{float(m.get('group_top1_regret', 0.0)):.3f} | "
            f"{float(m.get('group_top3_rerank_regret', 0.0)):.3f} | "
            f"{float(m.get('group_top10_rerank_regret', 0.0)):.3f} | "
            f"{int(row['training']['best_epoch'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sort_key(row: dict[str, Any]) -> tuple[float, float, float]:
    m = row["aggregate"]
    return (
        float(m.get("group_top1", 0.0)),
        -float(m.get("group_top1_regret", 0.0)),
        float(m.get("group_top3", 0.0)),
    )


def run(args: argparse.Namespace) -> None:
    started = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    topks = parse_topks(args.topks)
    summary_path = Path(args.base_summary)
    base_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    pool_k = int(args.pool_k or base_summary.get("pool_k", 10))
    feature_scope = args.feature_scope or base_summary.get("feature_scope", "both")

    train_specs, eval_specs, model_specs, selectors, lgbm_selectors, feature_selectors = load_summary_specs(summary_path)
    train_sets = load_sets(train_specs, model_specs, selectors, lgbm_selectors, feature_selectors)
    eval_sets = load_sets(eval_specs, model_specs, selectors, lgbm_selectors, feature_selectors)
    train_groups = build_groups(train_sets, pool_k, feature_scope, args.pool_local_features)
    if not train_groups:
        raise ValueError("no training pool groups built")
    standardizer = make_standardizer(train_groups)

    eval_rows: list[dict[str, Any]] = []
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    for hidden_text in args.hidden:
        hidden = parse_hidden(hidden_text)
        for seed in args.seed:
            model_name = f"{args.architecture}_{hidden_text}_seed{seed}"
            t0 = time.time()
            model, train_info = train_one(args, train_groups, standardizer, hidden, seed)
            model = model.to(device)
            pred = {
                ds.spec.name: predict_dataset(
                    ds,
                    model,
                    standardizer,
                    pool_k,
                    feature_scope,
                    args.pool_local_features,
                    device,
                )
                for ds in eval_sets
            }
            aggregate, per_dataset = evaluate_pred(eval_sets, pred, topks)
            model_path = out_dir / f"{model_name}.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_dim": int(train_groups[0].x.shape[1]),
                    "hidden": hidden,
                    "dropout": float(args.dropout),
                    "pool_k": int(pool_k),
                    "feature_scope": feature_scope,
                    "pool_local_features": bool(args.pool_local_features),
                    "architecture": args.architecture,
                    "d_model": int(args.d_model),
                    "num_heads": int(args.num_heads),
                    "num_layers": int(args.num_layers),
                    "standardizer_mean": standardizer.mean,
                    "standardizer_scale": standardizer.scale,
                    "base_summary": str(summary_path),
                },
                model_path,
            )
            eval_rows.append(
                {
                    "model": model_name,
                    "path": str(model_path),
                    "seed": int(seed),
                    "fit_seconds": time.time() - t0,
                    "training": train_info,
                    "aggregate": aggregate,
                    "datasets": per_dataset,
                }
            )
    eval_rows.sort(key=sort_key, reverse=True)
    summary = {
        "base_summary": str(summary_path),
        "pool_k": int(pool_k),
        "feature_scope": feature_scope,
        "pool_local_features": bool(args.pool_local_features),
        "architecture": args.architecture,
        "train_data": {spec.name: str(spec.path) for spec in train_specs},
        "eval_data": {spec.name: str(spec.path) for spec in eval_specs},
        "selectors": list(train_sets[0].selector_scores),
        "topks": topks,
        "train_groups": int(sum(len(ds.bounds) for ds in train_sets)),
        "eval_groups": int(sum(len(ds.bounds) for ds in eval_sets)),
        "train_pool_groups": int(len(train_groups)),
        "input_dim": int(train_groups[0].x.shape[1]),
        "train_union_ceiling": union_ceiling(train_sets, pool_k),
        "eval_union_ceiling": union_ceiling(eval_sets, pool_k),
        "hyperparameters": {
            "hidden": args.hidden,
            "d_model": int(args.d_model),
            "num_heads": int(args.num_heads),
            "num_layers": int(args.num_layers),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "dropout": float(args.dropout),
            "val_frac": float(args.val_frac),
            "batch_groups": int(args.batch_groups),
            "hard_weight": float(args.hard_weight),
            "soft_weight": float(args.soft_weight),
            "pairwise_weight": float(args.pairwise_weight),
            "mse_weight": float(args.mse_weight),
            "temperature": float(args.temperature),
            "ev_scale": float(args.ev_scale),
            "margin": float(args.margin),
        },
        "eval_rows": eval_rows,
        "elapsed_seconds": time.time() - started,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(out_dir / "summary.md", summary)
    print(json.dumps({"summary": str(out_dir / "summary.json"), "elapsed_seconds": summary["elapsed_seconds"]}, indent=2))


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pool-k", type=int, default=0)
    parser.add_argument("--feature-scope", choices=("both", "selector", "state"), default="")
    parser.add_argument("--pool-local-features", action="store_true")
    parser.add_argument("--architecture", choices=("mlp", "attention"), default="mlp")
    parser.add_argument("--hidden", action="append", default=None)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--seed", action="append", type=int, default=[20260621])
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--patience", type=int, default=45)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--batch-groups", type=int, default=32)
    parser.add_argument("--val-frac", type=float, default=0.12)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--hard-weight", type=float, default=1.0)
    parser.add_argument("--soft-weight", type=float, default=0.65)
    parser.add_argument("--pairwise-weight", type=float, default=0.35)
    parser.add_argument("--mse-weight", type=float, default=0.12)
    parser.add_argument("--temperature", type=float, default=0.55)
    parser.add_argument("--ev-scale", type=float, default=2.0)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--grad-clip", type=float, default=4.0)
    parser.add_argument("--device", default="")
    parser.add_argument("--topks", default="1,3,5,10,15,20")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.hidden is None:
        args.hidden = ["256x128"]
    run(args)


if __name__ == "__main__":
    main()
