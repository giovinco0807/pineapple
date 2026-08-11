"""Score HU T2 Stage8c rows with a saved risk-head model.

This is inference-only infrastructure for calibration/replay prep. It can score
rows that were not trainable for the model target, so the resulting probability
must not be treated as observed performance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .train_hu_turn2_stage8c_risk_head import (
    SPLIT_MODE_SOURCE_LOG,
    SPLIT_MODE_SOURCE_SEED,
    risk_feature_vector_for_row,
    safe_float,
    safe_int,
    split_group_key,
    target_label_for_row,
)
from .train_torch_action_value import select_device
from .turn3_model import _build_torch_mlp, _import_torch


DEFAULT_OUTPUT_DIR = Path("outputs/training/hu_turn2_stage8c_risk_head_predictions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument(
        "--include-feature-errors",
        action="store_true",
        help="Emit rows with feature errors in the skipped CSV instead of only counting them.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def recommended_use(row: dict[str, Any]) -> str:
    return str(row.get("recommended_training_use") or row.get("target_recommended_use") or "")


def metadata_for_row(row: dict[str, Any], *, row_index: int, target_mode: str) -> dict[str, Any]:
    target = target_label_for_row(row, target_mode)
    return {
        "row_index": row_index,
        "source_log": row.get("source_log", ""),
        "config_id": row.get("config_id", ""),
        "hand_seed": row.get("hand_seed", ""),
        "split_group_source_log": split_group_key(row, SPLIT_MODE_SOURCE_LOG),
        "split_group_source_seed": split_group_key(row, SPLIT_MODE_SOURCE_SEED),
        "seat": row.get("seat", ""),
        "seat_swap": row.get("seat_swap", ""),
        "candidate_source": row.get("candidate_source", ""),
        "recommended_training_use": recommended_use(row),
        "risk_target_group": target[1] if target is not None else "",
        "label": target[0] if target is not None else "",
        "state_signature": row.get("state_signature", ""),
        "action_signature": row.get("action_signature", ""),
        "baseline_action_signature": row.get("baseline_action_signature", ""),
        "candidate_index": safe_int(row.get("candidate_index", row.get("candidate_action_index")), -1),
        "baseline_index": safe_int(row.get("baseline_index", row.get("baseline_action_index")), -1),
        "realized_delta": safe_float(row.get("realized_delta")),
        "realized_delta_observed": int(
            truthy(
                row.get(
                    "realized_delta_observed",
                    truthy(row.get("override_fired")) and truthy(row.get("realized_delta_valid")),
                )
            )
        ),
        "realized_loss": max(0.0, -safe_float(row.get("realized_delta"))),
        "local_replay_status": row.get("local_replay_status", ""),
        "local_replay_action_mapping_status": row.get("local_replay_action_mapping_status", ""),
        "local_replay_bucket": row.get("local_replay_bucket", ""),
        "local_replay_label": row.get("local_replay_label", ""),
        "local_replay_delta": safe_float(row.get("local_replay_delta")),
        "confirm_delta": safe_float(row.get("confirm_delta")),
        "confirm_delta_se": safe_float(row.get("confirm_delta_se")),
        "predicted_delta": safe_float(row.get("predicted_delta")),
        "gate_probability": safe_float(row.get("gate_probability")),
        "candidate_ev_rank": safe_int(row.get("candidate_ev_rank"), 9999),
    }


def materialize_prediction_rows(
    rows: list[dict[str, Any]],
    *,
    feature_mode: str,
    target_mode: str,
    include_feature_errors: bool = False,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    features: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for index, row in enumerate(rows):
        try:
            feature_row = risk_feature_vector_for_row(row, feature_mode)
        except Exception as exc:  # pragma: no cover - surfaced in artifact.
            reason = f"feature_error:{type(exc).__name__}"
            counters[reason] += 1
            if include_feature_errors:
                skipped.append(metadata_for_row(row, row_index=index, target_mode=target_mode) | {"skip_reason": reason})
            continue
        features.append(feature_row)
        metadata.append(metadata_for_row(row, row_index=index, target_mode=target_mode))
    if not features:
        return np.empty((0, 0), dtype=np.float32), metadata, skipped, dict(counters)
    return np.vstack(features).astype(np.float32), metadata, skipped, dict(counters)


def predict_probabilities(
    features: np.ndarray,
    *,
    payload: dict[str, Any],
    device_choice: str,
    batch_size: int,
) -> np.ndarray:
    if features.size == 0:
        return np.asarray([], dtype=np.float32)
    torch = _import_torch()
    device = select_device(torch, device_choice)
    net = _build_torch_mlp(
        torch,
        int(payload["feature_dim"]),
        tuple(int(value) for value in payload.get("hidden_layer_sizes", (256, 128))),
        float(payload.get("dropout", 0.0)),
    ).to(device)
    net.load_state_dict(payload["state_dict"])
    net.eval()
    feature_mean = np.asarray(payload.get("feature_mean"), dtype=np.float32)
    feature_scale = np.asarray(payload.get("feature_scale"), dtype=np.float32)
    feature_scale[feature_scale < 1e-6] = 1.0
    if features.shape[1] != int(payload["feature_dim"]):
        raise ValueError(f"feature dim mismatch: rows={features.shape[1]} model={payload['feature_dim']}")
    x_norm = ((features - feature_mean) / feature_scale).astype(np.float32)
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, int(x_norm.shape[0]), batch_size):
            batch = torch.from_numpy(x_norm[start : start + batch_size]).to(device)
            logits = net(batch).squeeze(-1)
            probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
            probabilities.append(probs)
    return np.concatenate(probabilities) if probabilities else np.asarray([], dtype=np.float32)


def finite_probability(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else 0.0


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    torch = _import_torch()
    payload = torch.load(args.model, map_location="cpu", weights_only=False)
    feature_mode = str(payload.get("feature_mode") or "")
    target_mode = str(payload.get("target_mode") or "")
    if not feature_mode or not target_mode:
        raise ValueError("model payload must include feature_mode and target_mode")
    rows: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    for path in args.input_jsonl:
        loaded = read_jsonl(path)
        source_counts[str(path)] = len(loaded)
        rows.extend(loaded)
    loaded_rows = len(rows)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    features, metadata, skipped, skipped_counts = materialize_prediction_rows(
        rows,
        feature_mode=feature_mode,
        target_mode=target_mode,
        include_feature_errors=args.include_feature_errors,
    )
    probabilities = predict_probabilities(
        features,
        payload=payload,
        device_choice=args.device,
        batch_size=args.batch_size,
    )
    prediction_rows = [
        row | {"risk_probability": finite_probability(float(probabilities[index]))}
        for index, row in enumerate(metadata)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "risk_head_predictions.csv", prediction_rows)
    write_csv(args.output_dir / "risk_head_prediction_skipped.csv", skipped)
    manifest = {
        "schema": "hu_turn2_stage8c_risk_head_prediction_manifest_v1",
        "input_jsonl": [str(path) for path in args.input_jsonl],
        "model": str(args.model),
        "output_dir": str(args.output_dir),
        "source_counts": source_counts,
        "loaded_rows": loaded_rows,
        "sampled_rows": len(rows),
        "predicted_rows": len(prediction_rows),
        "skipped_rows": len(rows) - len(prediction_rows),
        "skipped_counts": skipped_counts,
        "feature_mode": feature_mode,
        "target_mode": target_mode,
        "feature_dim": int(payload["feature_dim"]),
        "production_p2_fixed": "No-Go",
        "teacher_50k": "No-Go",
        "t1_training": "No-Go",
    }
    (args.output_dir / "risk_head_prediction_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
