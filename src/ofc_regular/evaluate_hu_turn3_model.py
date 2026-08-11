"""Evaluate a HU Turn3 action-value model on teacher JSONL."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .hu_turn3_model import evaluate_model, load_hu_action_value_model, read_teacher_samples, sample_to_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--batch-samples", type=int, default=512)
    parser.add_argument(
        "--breakdown-key",
        choices=("source", "source_input_path"),
        default="source_input_path",
        help="Metadata key used for breakdown metrics.",
    )
    return parser.parse_args()


def sample_source(sample: dict[str, Any], *, breakdown_key: str = "source_input_path") -> str:
    if breakdown_key == "source_input_path":
        value = sample.get("source_input_path") or (sample.get("source_state") or {}).get("source_input_path")
        if value:
            return Path(str(value)).name
    return str(sample.get("source") or (sample.get("source_state") or {}).get("source") or "unknown")


def source_breakdown(
    model: Any,
    samples: list[dict[str, Any]],
    *,
    batch_samples: int,
    breakdown_key: str = "source_input_path",
) -> dict[str, dict[str, float]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_source[sample_source(sample, breakdown_key=breakdown_key)].append(sample)
    return {
        source: evaluate_model(model, source_samples, batch_samples=batch_samples)
        for source, source_samples in sorted(by_source.items())
    }


def sample_rows(
    model: Any,
    samples: list[dict[str, Any]],
    *,
    breakdown_key: str = "source_input_path",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        features, targets = sample_to_matrix(sample)
        predictions = model.predict_matrix(features)
        predicted_sorted_index = int(np.argmax(predictions))
        best_sorted_index = int(np.argmax(targets))
        predicted_action = sample["actions"][predicted_sorted_index]
        best_action = sample["actions"][best_sorted_index]
        best_score = float(targets[best_sorted_index])
        predicted_score = float(targets[predicted_sorted_index])
        rows.append(
            {
                "sample_id": sample.get("sample_id"),
                "source": sample_source(sample, breakdown_key="source"),
                "breakdown_source": sample_source(sample, breakdown_key=breakdown_key),
                "state_id": (sample.get("source_state") or {}).get("state_id"),
                "hand_seed": (sample.get("source_state") or {}).get("hand_seed"),
                "seat": sample.get("seat"),
                "future_count": sample.get("future_count"),
                "legal_action_count": sample.get("legal_action_count"),
                "best_original_index": best_action.get("original_index"),
                "predicted_original_index": predicted_action.get("original_index"),
                "best_score": best_score,
                "predicted_score": predicted_score,
                "regret": best_score - predicted_score,
                "predicted_model_score": float(predictions[predicted_sorted_index]),
                "best_model_score": float(predictions[best_sorted_index]),
            }
        )
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    samples = read_teacher_samples(args.input, max_samples=args.max_samples)
    if not samples:
        raise SystemExit("no HU Turn3 teacher samples")
    model = load_hu_action_value_model(args.model)
    metrics = {
        "model": str(args.model),
        "input": str(args.input),
        "samples": len(samples),
        "breakdown_key": args.breakdown_key,
        "overall": evaluate_model(model, samples, batch_samples=args.batch_samples),
        "source_breakdown": source_breakdown(
            model,
            samples,
            batch_samples=args.batch_samples,
            breakdown_key=args.breakdown_key,
        ),
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.rows_output is not None:
        write_csv(args.rows_output, sample_rows(model, samples, breakdown_key=args.breakdown_key))
    print(json.dumps(metrics, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
