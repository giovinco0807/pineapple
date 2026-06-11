"""Benchmark HU Turn3 Stage3 feature generation from a replay artifact."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .hu_turn3_model import load_hu_action_value_model
from .hu_turn3_stage3_feature_manifest import write_hu_turn3_stage3_feature_manifest
from .hu_turn3_stage3_feature_fast import (
    FEATURE_SCHEMA_VERSION,
    Stage3StateFeatureCache,
    build_hu_turn3_stage3_feature_matrix_batch,
)
from .hu_turn3_stage3_feature_replay import (
    expected_scores_path_for_replay,
    load_stage3_feature_replay,
)


def benchmark_replay(
    *,
    input_path: Path,
    schema_path: Path | None,
    output_path: Path,
    batch_size: int,
    limit_states: int,
    repeat: int,
    check_model_scores: bool,
    encoder_mode: str,
    compare_encoder_modes: bool,
    manifest_output: Path | None,
    include_rust: bool = False,
) -> dict[str, Any]:
    schema = _load_schema(schema_path) if schema_path is not None else {}
    if manifest_output is not None:
        write_hu_turn3_stage3_feature_manifest(manifest_output)
    replay = load_stage3_feature_replay(input_path, limit_states=limit_states)
    if not replay.states:
        raise RuntimeError("feature replay has no states")
    if compare_encoder_modes:
        mode_summaries = {}
        compare_modes = _compare_modes(include_rust)
        for mode in compare_modes:
            mode_summaries[mode] = _benchmark_one_mode(
                replay=replay,
                schema=schema,
                input_path=input_path,
                batch_size=batch_size,
                repeat=repeat,
                check_model_scores=check_model_scores,
                encoder_mode=mode,
            )
        parity = _parity_vs_scalar(
            replay=replay,
            schema=schema,
            batch_size=batch_size,
            include_rust=include_rust,
        )
        summary = {
            "input": str(input_path),
            "schema": str(schema_path) if schema_path is not None else "",
            "compare_encoder_modes": True,
            "include_rust": include_rust,
            "mode_results": mode_summaries,
            "speedups_vs_scalar_fast": _speedups_vs_scalar(mode_summaries),
            "feature_parity_vs_scalar_fast": parity["feature_parity"],
            "reference_parity_vs_scalar_fast": parity["reference_parity"],
            "attribution_by_mode": {
                mode: _attribution_summary(mode_summary)
                for mode, mode_summary in mode_summaries.items()
            },
            "metadata": replay.metadata,
            "manifest_output": str(manifest_output) if manifest_output is not None else "",
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return summary

    summary = _benchmark_one_mode(
        replay=replay,
        schema=schema,
        input_path=input_path,
        batch_size=batch_size,
        repeat=repeat,
        check_model_scores=check_model_scores,
        encoder_mode=encoder_mode,
    )
    summary["input"] = str(input_path)
    summary["schema"] = str(schema_path) if schema_path is not None else ""
    summary["metadata"] = replay.metadata
    summary["manifest_output"] = str(manifest_output) if manifest_output is not None else ""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _benchmark_one_mode(
    *,
    replay: Any,
    schema: dict[str, Any],
    input_path: Path,
    batch_size: int,
    repeat: int,
    check_model_scores: bool,
    encoder_mode: str,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    last_batch = None
    for _run_index in range(repeat):
        cache = Stage3StateFeatureCache(max_size=0)
        started_at = time.perf_counter()
        feature_batch = build_hu_turn3_stage3_feature_matrix_batch(
            replay.states,
            replay.actions_by_state,
            FEATURE_SCHEMA_VERSION,
            state_keys=replay.state_keys,
            state_feature_cache=cache,
            include_action_encodings=False,
            encoder_mode=encoder_mode,
        )
        elapsed = time.perf_counter() - started_at
        profile = dict(feature_batch.profile)
        profile["wall_seconds"] = elapsed
        profile["rows_per_second_wall"] = (
            feature_batch.X.shape[0] / elapsed if elapsed else 0.0
        )
        runs.append(profile)
        last_batch = feature_batch

    assert last_batch is not None
    score_check = {}
    if check_model_scores:
        score_check = _check_expected_scores(
            input_path=input_path,
            schema=schema,
            features=last_batch.X,
            batch_size=batch_size,
        )
    summary = {
        "encoder_mode": encoder_mode,
        "states": len(replay.states),
        "actions": int(last_batch.X.shape[0]),
        "feature_columns": int(last_batch.X.shape[1]) if last_batch.X.ndim == 2 else 0,
        "feature_dtype": str(last_batch.X.dtype),
        "repeat": repeat,
        "runs": runs,
        "aggregate": _aggregate_runs(runs),
        "attribution": _attribution_summary({"runs": runs}),
        "score_check": score_check,
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--schema", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--limit-states", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--check-model-scores", action="store_true")
    parser.add_argument(
        "--encoder-mode",
        choices=("scalar_fast", "numpy_direct_partial", "numpy_direct_full", "rust_direct"),
        default="scalar_fast",
    )
    parser.add_argument("--compare-encoder-modes", action="store_true")
    parser.add_argument("--include-rust", action="store_true")
    parser.add_argument("--manifest-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.repeat <= 0:
        raise SystemExit("--repeat must be positive")
    summary = benchmark_replay(
        input_path=args.input,
        schema_path=args.schema,
        output_path=args.output,
        batch_size=args.batch_size,
        limit_states=args.limit_states,
        repeat=args.repeat,
        check_model_scores=args.check_model_scores,
        encoder_mode=args.encoder_mode,
        compare_encoder_modes=args.compare_encoder_modes,
        manifest_output=args.manifest_output,
        include_rust=args.include_rust,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _load_schema(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, float]:
    values = [float(run.get("wall_seconds", 0.0)) for run in runs]
    feature_values = [float(run.get("stage3_feature_generation_total", 0.0)) for run in runs]
    rows = float(runs[-1].get("stage3_feature_rows", 0.0)) if runs else 0.0
    mean_wall = sum(values) / len(values) if values else 0.0
    mean_feature = sum(feature_values) / len(feature_values) if feature_values else 0.0
    return {
        "wall_seconds_mean": mean_wall,
        "wall_seconds_min": min(values) if values else 0.0,
        "wall_seconds_max": max(values) if values else 0.0,
        "feature_seconds_mean": mean_feature,
        "rows_per_second_mean": rows / mean_feature if mean_feature else 0.0,
    }


def _compare_modes(include_rust: bool) -> tuple[str, ...]:
    modes = ["scalar_fast", "numpy_direct_partial", "numpy_direct_full"]
    if include_rust:
        modes.append("rust_direct")
    return tuple(modes)


def _attribution_summary(summary: dict[str, Any]) -> dict[str, Any]:
    runs = summary.get("runs", [])
    if not runs:
        return {}
    feature_seconds = _mean_run_value(runs, "stage3_feature_generation_total")
    groups = {
        "after_board": _mean_run_value(runs, "stage3_after_board_construction_seconds"),
        "row_summary": _mean_run_value(runs, "stage3_row_summary_seconds"),
        "global_summary": _mean_run_value(runs, "stage3_global_summary_seconds"),
        "action_delta": _mean_run_value(runs, "stage3_action_delta_seconds"),
        "scalar_fallback": _mean_run_value(runs, "stage3_scalar_fallback_seconds"),
        "cache_lookup_update": _mean_run_value(runs, "stage3_cache_lookup_update_seconds"),
        "column_validation": _mean_run_value(runs, "stage3_column_validation_seconds"),
        "numpy_allocation": _mean_run_value(runs, "stage3_numpy_allocation_seconds"),
        "hgb_input_preparation": _mean_run_value(runs, "stage3_hgb_input_preparation_seconds"),
        "non_encoder_overhead": _mean_run_value(runs, "stage3_non_encoder_overhead_seconds"),
    }
    slowest_group, slowest_seconds = max(groups.items(), key=lambda item: item[1])
    last = runs[-1]
    return {
        "feature_seconds_mean": feature_seconds,
        "direct_column_count": int(float(last.get("direct_column_count", 0))),
        "scalar_fallback_column_count": int(float(last.get("scalar_fallback_column_count", 0))),
        "direct_column_coverage_ratio": float(last.get("direct_column_coverage_ratio", 0.0)),
        "group_seconds_mean": groups,
        "group_shares": {
            key: (value / feature_seconds if feature_seconds else 0.0)
            for key, value in groups.items()
        },
        "slowest_group": slowest_group,
        "slowest_group_seconds": slowest_seconds,
        "slowest_group_share": slowest_seconds / feature_seconds if feature_seconds else 0.0,
    }


def _mean_run_value(runs: list[dict[str, Any]], key: str) -> float:
    values = [float(run.get(key, 0.0)) for run in runs]
    return sum(values) / len(values) if values else 0.0


def _speedups_vs_scalar(mode_summaries: dict[str, dict[str, Any]]) -> dict[str, float]:
    baseline = float(
        mode_summaries.get("scalar_fast", {})
        .get("aggregate", {})
        .get("feature_seconds_mean", 0.0)
    )
    speedups: dict[str, float] = {}
    for mode, summary in mode_summaries.items():
        current = float(summary.get("aggregate", {}).get("feature_seconds_mean", 0.0))
        speedups[mode] = baseline / current if baseline and current else 0.0
    return speedups


def _parity_vs_scalar(
    *,
    replay: Any,
    schema: dict[str, Any],
    batch_size: int,
    include_rust: bool,
) -> dict[str, Any]:
    scalar = build_hu_turn3_stage3_feature_matrix_batch(
        replay.states,
        replay.actions_by_state,
        FEATURE_SCHEMA_VERSION,
        state_keys=replay.state_keys,
        state_feature_cache=Stage3StateFeatureCache(max_size=0),
        include_action_encodings=False,
        encoder_mode="scalar_fast",
    )
    scalar_predictions = _predict_matrix_for_parity(schema, scalar.X, batch_size)
    feature_parity: dict[str, Any] = {}
    reference_parity: dict[str, Any] = {}
    modes = ["numpy_direct_partial", "numpy_direct_full"]
    if include_rust:
        modes.append("rust_direct")
    for mode in modes:
        current = build_hu_turn3_stage3_feature_matrix_batch(
            replay.states,
            replay.actions_by_state,
            FEATURE_SCHEMA_VERSION,
            state_keys=replay.state_keys,
            state_feature_cache=Stage3StateFeatureCache(max_size=0),
            include_action_encodings=False,
            encoder_mode=mode,
        )
        diff = np.abs(scalar.X - current.X)
        feature_parity[mode] = {
            "rows": int(current.X.shape[0]),
            "columns": int(current.X.shape[1]) if current.X.ndim == 2 else 0,
            "max_abs_diff": float(diff.max()) if diff.size else 0.0,
            "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
            "allclose_1e_6": bool(np.allclose(scalar.X, current.X, rtol=0.0, atol=1e-6, equal_nan=True)),
        }
        current_predictions = _predict_matrix_for_parity(schema, current.X, batch_size)
        reference_parity[mode] = _reference_parity_from_predictions(
            replay.actions_by_state,
            scalar.row_to_state_index,
            scalar.row_to_action_index,
            scalar_predictions,
            current_predictions,
        )
    return {
        "feature_parity": feature_parity,
        "reference_parity": reference_parity,
    }


def _predict_matrix_for_parity(
    schema: dict[str, Any],
    features: np.ndarray,
    batch_size: int,
) -> np.ndarray | None:
    model_path = schema.get("metadata", {}).get("stage3_reference_model_path")
    if not model_path:
        return None
    model = load_hu_action_value_model(model_path)
    predictions: list[np.ndarray] = []
    for start in range(0, features.shape[0], batch_size):
        end = min(start + batch_size, features.shape[0])
        predictions.append(np.asarray(model.predict_matrix(features[start:end]), dtype=np.float64))
    return np.concatenate(predictions) if predictions else np.zeros(0, dtype=np.float64)


def _reference_parity_from_predictions(
    actions_by_state: list[list[Any]],
    row_to_state_index: np.ndarray,
    row_to_action_index: np.ndarray,
    scalar_predictions: np.ndarray | None,
    current_predictions: np.ndarray | None,
) -> dict[str, Any]:
    if scalar_predictions is None or current_predictions is None:
        return {"checked": False, "reason": "missing_model_path"}
    scalar_by_state = _predictions_by_state(
        actions_by_state,
        row_to_state_index,
        row_to_action_index,
        scalar_predictions,
    )
    current_by_state = _predictions_by_state(
        actions_by_state,
        row_to_state_index,
        row_to_action_index,
        current_predictions,
    )
    action_mismatches = 0
    max_margin_diff = 0.0
    max_score_diff = 0.0
    for scalar_scores, current_scores in zip(scalar_by_state, current_by_state):
        scalar_best, scalar_margin = _best_index_and_margin(scalar_scores)
        current_best, current_margin = _best_index_and_margin(current_scores)
        if scalar_best != current_best:
            action_mismatches += 1
        max_margin_diff = max(max_margin_diff, abs(scalar_margin - current_margin))
        max_score_diff = max(max_score_diff, float(np.max(np.abs(scalar_scores - current_scores))))
    state_count = len(scalar_by_state)
    return {
        "checked": True,
        "states": state_count,
        "action_mismatches": action_mismatches,
        "action_match_rate": (state_count - action_mismatches) / state_count if state_count else 0.0,
        "max_margin_diff": max_margin_diff,
        "max_score_diff": max_score_diff,
    }


def _predictions_by_state(
    actions_by_state: list[list[Any]],
    row_to_state_index: np.ndarray,
    row_to_action_index: np.ndarray,
    flat: np.ndarray,
) -> list[np.ndarray]:
    values = [
        np.full(len(actions), np.nan, dtype=np.float64)
        for actions in actions_by_state
    ]
    for row_index, score in enumerate(flat):
        state_index = int(row_to_state_index[row_index])
        action_index = int(row_to_action_index[row_index])
        values[state_index][action_index] = float(score)
    return values


def _best_index_and_margin(values: np.ndarray) -> tuple[int, float]:
    if values.size == 0:
        return -1, 0.0
    order = np.argsort(values)
    best = int(order[-1])
    if values.size == 1:
        return best, 0.0
    return best, float(values[order[-1]] - values[order[-2]])


def _check_expected_scores(
    *,
    input_path: Path,
    schema: dict[str, Any],
    features: np.ndarray,
    batch_size: int,
) -> dict[str, Any]:
    score_path = expected_scores_path_for_replay(input_path)
    model_path = schema.get("metadata", {}).get("stage3_reference_model_path")
    if not score_path.exists() or not model_path:
        return {"checked": False, "reason": "missing_expected_scores_or_model_path"}
    expected = np.load(score_path)
    if expected["row_index"].shape[0] == 0:
        return {"checked": False, "reason": "empty_expected_scores"}
    model = load_hu_action_value_model(model_path)
    predictions: list[np.ndarray] = []
    for start in range(0, features.shape[0], batch_size):
        end = min(start + batch_size, features.shape[0])
        predictions.append(np.asarray(model.predict_matrix(features[start:end]), dtype=np.float64))
    flat = np.concatenate(predictions) if predictions else np.zeros(0, dtype=np.float64)
    original_row_indices = expected["row_index"].astype(np.int64)
    mask = original_row_indices < flat.shape[0]
    row_indices = original_row_indices[mask]
    if row_indices.size == 0:
        return {"checked": False, "reason": "expected_rows_outside_replay"}
    expected_scores = expected["score"].astype(np.float64)[mask]
    diff = np.abs(flat[row_indices] - expected_scores)
    return {
        "checked": True,
        "rows": int(row_indices.size),
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
    }


if __name__ == "__main__":
    main()
