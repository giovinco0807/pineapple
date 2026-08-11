"""Train a HU Turn1 listwise neural candidate-generator model.

This trainer keeps the existing HU action-value runtime interface, but trains
the scalar action utility with a state-level softmax/listwise objective instead
of independent action classification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .hu_turn3_model import HuTorchActionValueModel, _build_torch_mlp, _import_torch, read_teacher_samples, sample_to_matrix, split_samples
from .train_hu_turn1_candidate_generator import evaluate_candidate_generator, infer_artifact_stage
from .hu_turn1_training_augmentation import augment_samples_by_suit
from .train_hu_turn3 import parse_source_weights, sample_source, source_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--validation-input",
        type=Path,
        help="Optional explicit state-level validation JSONL; disables random holdout splitting.",
    )
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--baseline-model", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026062608)
    parser.add_argument(
        "--suit-augmentations",
        type=int,
        default=0,
        help="Non-identity global suit permutations appended to the training split only (0-23).",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-sizes", default="256,128")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--listwise-temperature", type=float, default=4.0)
    parser.add_argument("--mse-weight", type=float, default=0.1)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after this many non-improving validation epochs; 0 disables.",
    )
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-predict-size", type=int, default=8192)
    parser.add_argument(
        "--source-weight",
        action="append",
        default=[],
        metavar="SOURCE=WEIGHT",
        help="Per-sample source weight. May be repeated or comma-separated.",
    )
    return parser.parse_args()


def parse_hidden_sizes(raw: str) -> tuple[int, ...]:
    sizes = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("--hidden-sizes must contain positive integers")
    return sizes


def materialize_samples(samples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    materialized: list[dict[str, Any]] = []
    for sample in samples:
        features, targets = sample_to_matrix(sample)
        if features.shape[0] == 0:
            continue
        materialized.append(
            {
                "sample": sample,
                "features": features.astype(np.float32, copy=False),
                "targets": targets.astype(np.float32, copy=False),
            }
        )
    if not materialized:
        raise ValueError("no materialized action rows")
    return materialized


def normalize_stats(materialized: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, float, float]:
    features = np.vstack([row["features"] for row in materialized]).astype(np.float32, copy=False)
    targets = np.concatenate([row["targets"] for row in materialized]).astype(np.float32, copy=False)
    feature_mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale = features.std(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale[feature_scale < 1e-6] = 1.0
    target_mean = float(targets.mean(dtype=np.float64))
    target_scale = float(targets.std(dtype=np.float64))
    if target_scale < 1e-6:
        target_scale = 1.0
    return feature_mean, feature_scale, target_mean, target_scale


def train_epoch(
    *,
    torch: Any,
    net: Any,
    optimizer: Any,
    materialized: Sequence[dict[str, Any]],
    order: np.ndarray,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    target_mean: float,
    target_scale: float,
    source_weights: dict[str, float],
    batch_size: int,
    listwise_temperature: float,
    mse_weight: float,
    device: str,
) -> float:
    net.train()
    total_loss = 0.0
    total_weight = 0.0
    mean_tensor = torch.from_numpy(feature_mean).to(device)
    scale_tensor = torch.from_numpy(feature_scale).to(device)
    temperature = max(1e-6, float(listwise_temperature))
    for start in range(0, len(order), batch_size):
        batch_indices = order[start : start + batch_size]
        optimizer.zero_grad(set_to_none=True)
        batch_loss = None
        batch_weight = 0.0
        for index in batch_indices:
            row = materialized[int(index)]
            features = torch.from_numpy(row["features"]).to(device)
            targets = torch.from_numpy(row["targets"]).to(device)
            x = (features - mean_tensor) / scale_tensor
            pred_norm = net(x).squeeze(-1)
            target_norm = (targets - target_mean) / target_scale
            shifted_targets = (targets - torch.max(targets)) / temperature
            target_prob = torch.softmax(shifted_targets, dim=0)
            listwise = -(target_prob * torch.log_softmax(pred_norm, dim=0)).sum()
            mse = torch.mean((pred_norm - target_norm) ** 2)
            weight = float(source_weights.get(sample_source(row["sample"]), 1.0))
            sample_loss = (listwise + float(mse_weight) * mse) * weight
            batch_loss = sample_loss if batch_loss is None else batch_loss + sample_loss
            batch_weight += weight
        if batch_loss is None:
            continue
        normalized_loss = batch_loss / max(1e-6, batch_weight)
        normalized_loss.backward()
        optimizer.step()
        total_loss += float(normalized_loss.detach().cpu()) * batch_weight
        total_weight += batch_weight
    return total_loss / max(1e-6, total_weight)


def evaluate_epoch_loss(
    *,
    torch: Any,
    net: Any,
    materialized: Sequence[dict[str, Any]],
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    target_mean: float,
    target_scale: float,
    source_weights: dict[str, float],
    listwise_temperature: float,
    mse_weight: float,
    device: str,
) -> float:
    net.eval()
    total_loss = 0.0
    total_weight = 0.0
    mean_tensor = torch.from_numpy(feature_mean).to(device)
    scale_tensor = torch.from_numpy(feature_scale).to(device)
    temperature = max(1e-6, float(listwise_temperature))
    inference_guard = getattr(torch, "inference_mode", torch.no_grad)
    with inference_guard():
        for row in materialized:
            features = torch.from_numpy(row["features"]).to(device)
            targets = torch.from_numpy(row["targets"]).to(device)
            pred_norm = net((features - mean_tensor) / scale_tensor).squeeze(-1)
            target_norm = (targets - target_mean) / target_scale
            target_prob = torch.softmax((targets - torch.max(targets)) / temperature, dim=0)
            listwise = -(target_prob * torch.log_softmax(pred_norm, dim=0)).sum()
            mse = torch.mean((pred_norm - target_norm) ** 2)
            weight = float(source_weights.get(sample_source(row["sample"]), 1.0))
            total_loss += float((listwise + float(mse_weight) * mse).detach().cpu()) * weight
            total_weight += weight
    return total_loss / max(1e-6, total_weight)


def build_model_from_net(
    *,
    net: Any,
    hidden_sizes: tuple[int, ...],
    dropout: float,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    target_mean: float,
    target_scale: float,
    batch_size: int,
) -> HuTorchActionValueModel:
    return HuTorchActionValueModel(
        state_dict={key: value.detach().cpu() for key, value in net.state_dict().items()},
        hidden_layer_sizes=hidden_sizes,
        dropout=float(dropout),
        batch_size=int(batch_size),
        feature_mean=feature_mean.astype(np.float32, copy=False),
        feature_scale=feature_scale.astype(np.float32, copy=False),
        target_mean=float(target_mean),
        target_scale=float(target_scale),
    )


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.early_stopping_patience < 0:
        raise SystemExit("--early-stopping-patience must be non-negative")
    if args.early_stopping_min_delta < 0.0:
        raise SystemExit("--early-stopping-min-delta must be non-negative")
    try:
        hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
        source_weights = parse_source_weights(args.source_weight)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    samples = read_teacher_samples(args.input, max_samples=args.max_samples)
    if not samples:
        raise SystemExit("no HU T0/T1 teacher samples")
    artifact_stage = infer_artifact_stage(samples)
    if args.validation_input is not None:
        train_samples_raw = samples
        holdout_samples = read_teacher_samples(args.validation_input)
        if not holdout_samples:
            raise SystemExit("explicit validation input has no HU T0/T1 teacher samples")
        validation_stage = infer_artifact_stage(holdout_samples)
        if validation_stage != artifact_stage:
            raise SystemExit(
                f"training/validation artifact stage mismatch: {artifact_stage} != {validation_stage}"
            )
        split_strategy = "explicit_validation_input"
    else:
        train_samples_raw, holdout_samples = split_samples(
            samples,
            holdout_fraction=args.holdout,
            seed=args.seed,
        )
        split_strategy = "random_state_holdout"
    try:
        train_samples = augment_samples_by_suit(
            train_samples_raw,
            count=int(args.suit_augmentations),
            seed=int(args.seed),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    train_rows = materialize_samples(train_samples)
    validation_rows = materialize_samples(holdout_samples) if args.early_stopping_patience > 0 else []
    feature_mean, feature_scale, target_mean, target_scale = normalize_stats(train_rows)

    torch = _import_torch()
    torch.manual_seed(int(args.seed))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = _build_torch_mlp(torch, int(feature_mean.shape[0]), hidden_sizes, float(args.dropout)).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    rng = np.random.default_rng(int(args.seed))
    losses: list[dict[str, float]] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0
    stopped_early = False
    for epoch in range(1, int(args.epochs) + 1):
        order = rng.permutation(len(train_rows))
        loss = train_epoch(
            torch=torch,
            net=net,
            optimizer=optimizer,
            materialized=train_rows,
            order=order,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            target_mean=target_mean,
            target_scale=target_scale,
            source_weights=source_weights,
            batch_size=int(args.batch_size),
            listwise_temperature=float(args.listwise_temperature),
            mse_weight=float(args.mse_weight),
            device=device,
        )
        loss_row = {"epoch": float(epoch), "train_loss": float(loss)}
        if args.early_stopping_patience > 0:
            validation_loss = evaluate_epoch_loss(
                torch=torch,
                net=net,
                materialized=validation_rows,
                feature_mean=feature_mean,
                feature_scale=feature_scale,
                target_mean=target_mean,
                target_scale=target_scale,
                source_weights=source_weights,
                listwise_temperature=float(args.listwise_temperature),
                mse_weight=float(args.mse_weight),
                device=device,
            )
            loss_row["validation_loss"] = float(validation_loss)
            if validation_loss < best_validation_loss - float(args.early_stopping_min_delta):
                best_validation_loss = float(validation_loss)
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in net.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        losses.append(loss_row)
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            stopped_early = True
            break

    if best_state is not None:
        net.load_state_dict(best_state)

    model = build_model_from_net(
        net=net,
        hidden_sizes=hidden_sizes,
        dropout=float(args.dropout),
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        batch_size=int(args.batch_predict_size),
    )
    model.save(args.model_output)

    baseline_eval: dict[str, Any] | None = None
    if args.baseline_model is not None:
        from .hu_turn3_model import load_hu_action_value_model

        baseline_model = load_hu_action_value_model(args.baseline_model)
        baseline_eval = {
            "path": str(args.baseline_model),
            "train": evaluate_candidate_generator(baseline_model, train_samples, accept_regret=0.25),
            "holdout": evaluate_candidate_generator(baseline_model, holdout_samples, accept_regret=0.25),
        }

    metrics = {
        "schema": f"{artifact_stage}_listwise_torch_training_metrics_v1",
        "artifact_stage": artifact_stage,
        "input": str(args.input),
        "validation_input": str(args.validation_input) if args.validation_input else None,
        "split_strategy": split_strategy,
        "model_output": str(args.model_output),
        "training_objective": "state_listwise_softmax",
        "total_samples": len(samples),
        "raw_train_samples": len(train_samples_raw),
        "train_samples": len(train_samples),
        "holdout_samples": len(holdout_samples),
        "suit_augmentations": int(args.suit_augmentations),
        "hidden_sizes": list(hidden_sizes),
        "dropout": float(args.dropout),
        "epochs": int(args.epochs),
        "epochs_trained": len(losses),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "listwise_temperature": float(args.listwise_temperature),
        "mse_weight": float(args.mse_weight),
        "early_stopping": {
            "enabled": bool(args.early_stopping_patience > 0),
            "patience": int(args.early_stopping_patience),
            "min_delta": float(args.early_stopping_min_delta),
            "best_epoch": int(best_epoch),
            "best_validation_loss": (
                float(best_validation_loss) if np.isfinite(best_validation_loss) else None
            ),
            "stopped_early": bool(stopped_early),
            "restored_best_state": bool(best_state is not None),
        },
        "source_weights": source_weights,
        "source_counts": source_counts(samples),
        "train_source_counts": source_counts(train_samples),
        "holdout_source_counts": source_counts(holdout_samples),
        "device": device,
        "target_mean": float(target_mean),
        "target_scale": float(target_scale),
        "losses": losses,
        "train": evaluate_candidate_generator(model, train_samples, accept_regret=0.25),
        "holdout": evaluate_candidate_generator(model, holdout_samples, accept_regret=0.25),
        "baseline_model": baseline_eval,
        "decision": "candidate_generator_only_not_production_runtime",
    }
    print(json.dumps(metrics, indent=2))
    if args.metrics_output is not None:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
