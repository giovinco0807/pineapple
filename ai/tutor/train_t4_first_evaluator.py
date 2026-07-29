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
) -> dict:
    device = torch.device(device_name)
    fit_x, fit_y, _ = load_split(data_dir, "fit", device)
    dev_x, dev_y, _ = load_split(data_dir, "dev", device)

    # Standardize on the fit split only; dev and test never inform the scaler.
    mean = fit_x.mean(dim=0)
    std = fit_x.std(dim=0)
    std = torch.where(std > 1e-6, std, torch.ones_like(std))
    fit_x = (fit_x - mean) / std
    dev_x = (dev_x - mean) / std

    model = T4FirstEvaluator(fit_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.SmoothL1Loss()

    out_dir.mkdir(parents=True, exist_ok=True)
    best = {"dev_mae": float("inf"), "epoch": -1}
    history = []
    rows = fit_x.shape[0]
    started = time.time()

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(rows, device=device)
        total = 0.0
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
            dev_metrics = metrics(model(dev_x), dev_y)
        history.append(
            {"epoch": epoch, "fit_loss": total / rows, **{f"dev_{k}": v for k, v in dev_metrics.items()}}
        )
        if dev_metrics["mae"] < best["dev_mae"]:
            best = {"dev_mae": dev_metrics["mae"], "epoch": epoch, **dev_metrics}
            torch.save(
                {
                    "schema": MODEL_SCHEMA,
                    "model_state_dict": model.state_dict(),
                    "input_mean": mean.cpu(),
                    "input_std": std.cpu(),
                    "input_dim": int(fit_x.shape[1]),
                    "hidden": list(HIDDEN),
                },
                out_dir / "evaluator_best.pt",
            )
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(
                f"epoch {epoch:3d}  fit_loss {total / rows:.4f}  "
                f"dev MAE {dev_metrics['mae']:.4f}  corr {dev_metrics['correlation']:.4f}",
                flush=True,
            )

    # Test is read once, after the dev-selected checkpoint is frozen.
    checkpoint = torch.load(out_dir / "evaluator_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    test_x, test_y, test_jokers = load_split(data_dir, "test", device)
    test_x = (test_x - mean) / std
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
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "parameters": sum(p.numel() for p in model.parameters()),
        "fit_rows": int(rows),
        "dev_rows": int(dev_x.shape[0]),
        "test_rows": int(test_x.shape[0]),
        "selected_epoch": best["epoch"],
        "dev_best": best,
        "test": test_metrics,
        "test_by_visible_jokers": by_joker,
        "elapsed_seconds": time.time() - started,
        "test_touched_once": True,
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
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
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
