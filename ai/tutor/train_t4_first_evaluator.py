"""Train the T4 first-seat evaluator on exact uniform-deal EV labels.

Architecture follows the regular track's `t4_model.rs`, whose header records
why it is a dense net rather than the boosted ensemble that scored marginally
better: at batch size one the ensemble cost 9,006 us against an exact
first-seat solve of roughly 1,100 us, so it was slower than the thing it would
replace.  Four matrix multiplies also port without dragging a tree format along.

The dataset's fit/dev/test split was fixed by root seed before any label was
computed, so the holdout here cannot be reselected.  Dev picks the checkpoint;
test is touched exactly once, at the end.

Usage:
    python -m ai.tutor.train_t4_first_evaluator --data-dir <dir> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

MODEL_SCHEMA = "ofc_t4_first_evaluator/v1"
HIDDEN = (256, 128, 64)


def parse_hidden(value: str) -> tuple[int, ...]:
    """Parse a reproducible comma-separated dense architecture."""
    try:
        hidden = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--hidden must contain integers") from exc
    if not hidden or any(size <= 0 for size in hidden):
        raise argparse.ArgumentTypeError("--hidden sizes must be positive")
    return hidden


class T4FirstEvaluator(nn.Module):
    """Dense regressor from the 101-dim encoder to the exact EV."""

    def __init__(self, input_dim: int, hidden: tuple[int, ...] = HIDDEN) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for size in hidden:
            layers.append(nn.Linear(previous, size))
            layers.append(nn.ReLU())
            previous = size
        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def load_split(data_dir: Path, name: str, device: torch.device):
    payload = np.load(data_dir / f"{name}.npz")
    x = torch.tensor(payload["x"], dtype=torch.float32, device=device)
    y = torch.tensor(payload["y"], dtype=torch.float32, device=device)
    jokers = torch.tensor(payload["jokers"].astype(np.int64), device=device)
    return x, y, jokers


def root_groups(data_dir: Path, name: str) -> list[np.ndarray] | None:
    """Row indices grouped by the root they came from, if the split says so."""
    payload = np.load(data_dir / f"{name}.npz")
    if "roots" not in payload:
        return None
    roots = payload["roots"]
    order = np.argsort(roots, kind="stable")
    boundaries = np.flatnonzero(np.diff(roots[order])) + 1
    return [group for group in np.split(order, boundaries) if group.size > 1]


def split_roots(data_dir: Path, name: str) -> np.ndarray | None:
    """The raw root column used to prove fit/dev content separation."""
    payload = np.load(data_dir / f"{name}.npz")
    return payload["roots"] if "roots" in payload else None


def validate_fit_dev(
    fit_x: torch.Tensor,
    dev_x: torch.Tensor,
    fit_roots: np.ndarray | None,
    dev_roots: np.ndarray | None,
    *,
    require_roots: bool = False,
) -> None:
    """Refuse a drifted feature width or a provable fit/dev root leak."""
    if fit_x.ndim != 2 or dev_x.ndim != 2:
        raise ValueError("fit and dev features must both be two-dimensional")
    if fit_x.shape[1] != dev_x.shape[1]:
        raise ValueError(
            f"fit/dev feature width differs: {fit_x.shape[1]} != {dev_x.shape[1]}"
        )
    if fit_roots is None and dev_roots is None and not require_roots:
        return
    if fit_roots is None or dev_roots is None:
        raise ValueError("fit and dev splits both need roots for overlap validation")
    overlap = np.intersect1d(np.unique(fit_roots), np.unique(dev_roots))
    if overlap.size:
        raise ValueError(
            f"fit/dev root overlap: {overlap.size} roots, first={int(overlap[0])}"
        )


def charged_regret(
    prediction: torch.Tensor, target: torch.Tensor, groups: list[np.ndarray]
) -> float:
    """What the model's pick costs against the best action, per root.

    This is the quantity the gate reports, so it is also the quantity worth
    selecting a checkpoint on.  Dev MAE has twice picked a checkpoint that
    tracked the level and misranked the actions -- the level is shared by
    every action at a root and cancels in exactly the comparison that
    decides the play.
    """
    total = 0.0
    for group in groups:
        truth = target[group]
        pick = group[int(torch.argmax(prediction[group]).item())]
        total += float(truth.max().item() - target[pick].item())
    return total / max(len(groups), 1)


def soft_regret_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    group_lengths: list[int],
    temperature: float,
) -> torch.Tensor:
    """Differentiable regret of a soft choice, averaged over decision roots.

    Pointwise EV regression spends equal effort on errors shared by every move
    at a root, even though those errors cancel when the model chooses.  This
    term instead charges probability assigned to a move by exactly the EV it
    gives up to that root's best move.  It is an optional addition: weight zero
    retains the historical lap-two training path bit for bit.
    """
    if temperature <= 0:
        raise ValueError("rank temperature must be positive")
    losses: list[torch.Tensor] = []
    start = 0
    for length in group_lengths:
        end = start + length
        truth = target[start:end]
        predicted = prediction[start:end]
        probability = torch.softmax(predicted / temperature, dim=0)
        losses.append(torch.sum(probability * (truth.max() - truth)))
        start = end
    if start != prediction.numel():
        raise ValueError("group lengths do not partition the predictions")
    if not losses:
        return prediction.sum() * 0.0
    return torch.stack(losses).mean()


def best_action_ranking_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    group_lengths: list[int],
    temperature: float,
    best_positions: list[int] | None = None,
) -> torch.Tensor:
    """Pair every inferior action with the teacher-best action.

    The loss is shift-invariant within a root and weights each comparison by
    its true EV gap.  Unlike pointwise regression, a large common level error
    cannot consume capacity here; unlike the soft-choice loss, the teacher's
    best action remains the explicit anchor even when several alternatives
    currently receive similar model scores.
    """
    if temperature <= 0:
        raise ValueError("best-rank temperature must be positive")
    total_length = sum(group_lengths)
    if total_length != prediction.numel():
        raise ValueError("group lengths do not partition the predictions")
    if not group_lengths:
        return prediction.sum() * 0.0
    offsets = np.cumsum([0, *group_lengths[:-1]]).tolist()
    if best_positions is None:
        best_positions = [
            offset + int(torch.argmax(target[offset : offset + length]).item())
            for offset, length in zip(offsets, group_lengths)
        ]
    if len(best_positions) != len(group_lengths):
        raise ValueError("best positions must have one entry per group")

    device = prediction.device
    lengths = torch.tensor(group_lengths, dtype=torch.long, device=device)
    group_ids = torch.repeat_interleave(
        torch.arange(len(group_lengths), device=device), lengths
    )
    best_index = torch.tensor(best_positions, dtype=torch.long, device=device)
    best_prediction = torch.repeat_interleave(prediction[best_index], lengths)
    best_target = torch.repeat_interleave(target[best_index], lengths)
    gaps = best_target - target
    gap_totals = torch.zeros(len(group_lengths), device=device).scatter_add_(
        0, group_ids, gaps
    )
    denominators = torch.repeat_interleave(gap_totals.clamp_min(1e-12), lengths)
    weighted = (
        torch.nn.functional.softplus((prediction - best_prediction) / temperature)
        * gaps
        / denominators
    )
    return weighted.sum() / len(group_lengths)


def metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    error = prediction - target
    mae = error.abs().mean().item()
    rmse = torch.sqrt((error**2).mean()).item()
    if target.numel() > 1 and target.std() > 0 and prediction.std() > 0:
        stacked = torch.stack([prediction, target])
        correlation = torch.corrcoef(stacked)[0, 1].item()
    else:
        correlation = float("nan")
    return {"mae": mae, "rmse": rmse, "correlation": correlation}


def run(
    *,
    data_dir: Path,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device_name: str,
    select_on: str = "mae",
    rank_weight: float = 0.0,
    rank_temperature: float = 1.0,
    best_rank_weight: float = 0.0,
    seed: int | None = None,
    evaluate_test: bool = True,
    hidden: tuple[int, ...] = HIDDEN,
    weight_decay: float = 0.01,
    dev_data_dir: Path | None = None,
) -> dict:
    if rank_weight < 0:
        raise ValueError("rank weight cannot be negative")
    if best_rank_weight < 0:
        raise ValueError("best-rank weight cannot be negative")
    if rank_temperature <= 0:
        raise ValueError("rank temperature must be positive")
    if weight_decay < 0:
        raise ValueError("weight decay cannot be negative")
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    device = torch.device(device_name)
    effective_dev_dir = dev_data_dir or data_dir
    fit_x, fit_y, _ = load_split(data_dir, "fit", device)
    dev_x, dev_y, _ = load_split(effective_dev_dir, "dev", device)
    fit_roots = split_roots(data_dir, "fit")
    dev_roots = split_roots(effective_dev_dir, "dev")
    validate_fit_dev(
        fit_x,
        dev_x,
        fit_roots,
        dev_roots,
        require_roots=dev_data_dir is not None,
    )

    # Standardize on the fit split only; dev and test never inform the scaler.
    mean = fit_x.mean(dim=0)
    std = fit_x.std(dim=0)
    std = torch.where(std > 1e-6, std, torch.ones_like(std))
    fit_x = (fit_x - mean) / std
    dev_x = (dev_x - mean) / std

    model = T4FirstEvaluator(fit_x.shape[1], hidden).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.SmoothL1Loss()

    fit_groups = root_groups(data_dir, "fit")
    dev_groups = root_groups(effective_dev_dir, "dev")
    if select_on == "regret" and not dev_groups:
        raise SystemExit("--select-on regret needs a `roots` array in dev.npz")
    if (rank_weight or best_rank_weight) and not fit_groups:
        raise SystemExit("ranking losses need a `roots` array in fit.npz")
    fit_best_local = None
    if best_rank_weight:
        fit_y_cpu = fit_y.detach().cpu().numpy()
        fit_best_local = [int(np.argmax(fit_y_cpu[group])) for group in fit_groups]

    out_dir.mkdir(parents=True, exist_ok=True)
    best = {"dev_mae": float("inf"), "epoch": -1, "score": float("inf")}
    history = []
    rows = fit_x.shape[0]
    started = time.time()

    for epoch in range(epochs):
        model.train()
        total = 0.0
        rank_total = 0.0
        rank_roots = 0
        if rank_weight or best_rank_weight:
            group_order = torch.randperm(len(fit_groups)).cpu().tolist()
            for first in range(0, len(group_order), max(1, batch_size // 24)):
                groups = [fit_groups[i] for i in group_order[first : first + max(1, batch_size // 24)]]
                flat = np.concatenate(groups)
                index = torch.tensor(flat, dtype=torch.long, device=device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(fit_x[index])
                point_loss = loss_fn(prediction, fit_y[index])
                lengths = [len(group) for group in groups]
                rank_loss = (
                    soft_regret_loss(prediction, fit_y[index], lengths, rank_temperature)
                    if rank_weight else prediction.sum() * 0.0
                )
                best_positions = None
                if best_rank_weight:
                    assert fit_best_local is not None
                    offsets = np.cumsum([0, *lengths[:-1]]).tolist()
                    best_positions = [
                        offset + fit_best_local[group_id]
                        for offset, group_id in zip(
                            offsets, group_order[first : first + len(groups)]
                        )
                    ]
                best_rank_loss = (
                    best_action_ranking_loss(
                        prediction,
                        fit_y[index],
                        lengths,
                        rank_temperature,
                        best_positions,
                    )
                    if best_rank_weight else prediction.sum() * 0.0
                )
                loss = (
                    point_loss
                    + rank_weight * rank_loss
                    + best_rank_weight * best_rank_loss
                )
                loss.backward()
                optimizer.step()
                total += point_loss.item() * index.numel()
                rank_total += rank_loss.item() * len(groups)
                rank_roots += len(groups)
        else:
            permutation = torch.randperm(rows, device=device)
            for start in range(0, rows, batch_size):
                index = permutation[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(fit_x[index]), fit_y[index])
                loss.backward()
                optimizer.step()
                total += loss.item() * index.numel()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            dev_prediction = model(dev_x)
            dev_metrics = metrics(dev_prediction, dev_y)
            if dev_groups:
                dev_metrics["regret"] = charged_regret(dev_prediction, dev_y, dev_groups)
        epoch_metrics = {
            "epoch": epoch,
            "fit_loss": total / rows,
            **{f"dev_{k}": v for k, v in dev_metrics.items()},
        }
        if rank_roots:
            epoch_metrics["fit_soft_regret"] = rank_total / rank_roots
        history.append(epoch_metrics)
        score = dev_metrics[select_on]
        if score < best["score"]:
            best = {"score": score, "dev_mae": dev_metrics["mae"], "epoch": epoch,
                    **dev_metrics}
            torch.save(
                {
                    "schema": MODEL_SCHEMA,
                    "model_state_dict": model.state_dict(),
                    "input_mean": mean.cpu(),
                    "input_std": std.cpu(),
                    "input_dim": int(fit_x.shape[1]),
                    "hidden": list(hidden),
                    "selected_on": select_on,
                    "rank_weight": rank_weight,
                    "rank_temperature": rank_temperature,
                    "best_rank_weight": best_rank_weight,
                    "weight_decay": weight_decay,
                    "seed": seed,
                    "data_dir": str(data_dir),
                    "dev_data_dir": str(effective_dev_dir),
                },
                out_dir / "evaluator_best.pt",
            )
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(
                f"epoch {epoch:3d}  fit_loss {total / rows:.4f}  "
                f"dev MAE {dev_metrics['mae']:.4f}  corr {dev_metrics['correlation']:.4f}"
                + (f"  regret {dev_metrics['regret']:.4f}" if "regret" in dev_metrics else ""),
                flush=True,
            )

    # Test is read once, after the dev-selected checkpoint is frozen.  Tuning
    # runs can explicitly skip it so choosing a rank weight does not turn the
    # final holdout into another dev set.
    checkpoint = torch.load(out_dir / "evaluator_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    test_metrics = None
    by_joker = None
    test_rows = None
    if evaluate_test:
        test_x, test_y, test_jokers = load_split(data_dir, "test", device)
        test_x = (test_x - mean) / std
        test_rows = int(test_x.shape[0])
        with torch.no_grad():
            prediction = model(test_x)
            test_metrics = metrics(prediction, test_y)
            by_joker = {}
            for value in sorted(set(test_jokers.tolist())):
                mask = test_jokers == value
                if mask.sum() > 1:
                    by_joker[str(value)] = {
                        "rows": int(mask.sum().item()),
                        **metrics(prediction[mask], test_y[mask]),
                    }

    report = {
        "schema": MODEL_SCHEMA,
        "data_dir": str(data_dir),
        "dev_data_dir": str(effective_dev_dir),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "hidden": list(hidden),
        "weight_decay": weight_decay,
        "rank_weight": rank_weight,
        "rank_temperature": rank_temperature,
        "best_rank_weight": best_rank_weight,
        "seed": seed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "fit_rows": int(rows),
        "dev_rows": int(dev_x.shape[0]),
        "test_rows": test_rows,
        "selected_epoch": best["epoch"],
        "dev_best": best,
        "test": test_metrics,
        "test_by_visible_jokers": by_joker,
        "elapsed_seconds": time.time() - started,
        "test_touched_once": evaluate_test,
        "history": history,
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--dev-data-dir",
        type=Path,
        default=None,
        help="optional separate encoded corpus supplying dev.npz; fit remains in --data-dir",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--hidden",
        type=parse_hidden,
        default=HIDDEN,
        metavar="N,N,...",
        help="Dense hidden widths (default: 256,128,64).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW weight decay (default: 0.01).",
    )
    parser.add_argument(
        "--select-on", choices=["mae", "regret"], default="mae",
        help="dev metric the checkpoint is chosen by; regret needs `roots` in the npz",
    )
    parser.add_argument(
        "--rank-weight", type=float, default=0.0,
        help="Add this multiple of within-root soft regret to SmoothL1 (default: off).",
    )
    parser.add_argument(
        "--rank-temperature", type=float, default=1.0,
        help="Temperature used by the ranking losses.",
    )
    parser.add_argument(
        "--best-rank-weight", type=float, default=0.0,
        help="Add EV-gap-weighted best-vs-rest ranking loss (default: off).",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--skip-test", action="store_true",
        help="Do not open test.npz; use for dev-only hyperparameter selection.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    report = run(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device_name=args.device,
        select_on=args.select_on,
        rank_weight=args.rank_weight,
        rank_temperature=args.rank_temperature,
        best_rank_weight=args.best_rank_weight,
        seed=args.seed,
        evaluate_test=not args.skip_test,
        hidden=args.hidden,
        weight_decay=args.weight_decay,
        dev_data_dir=args.dev_data_dir,
    )
    print(
        json.dumps(
            {
                "parameters": report["parameters"],
                "selected_epoch": report["selected_epoch"],
                "test": report["test"],
                "test_by_visible_jokers": report["test_by_visible_jokers"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
