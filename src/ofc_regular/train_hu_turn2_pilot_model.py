"""Train and evaluate a HU Turn2 pilot multi-head model from a feature cache."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .build_hu_turn2_pilot_feature_cache import (
    FEATURE_CACHE_SCHEMA,
    FEATURE_VALUE_SCHEMA,
    REGULAR_RULES_DIGEST,
    SPLIT_ID_TO_NAME,
    SPLIT_NAME_TO_ID,
    cache_identity_payload,
    sha256_file,
)
from .action_key import ACTION_KEY_SCHEMA
from .hu_infoset import OBSERVATION_SCHEMA, POLICY_FEATURE_SAMPLE_SCHEMA
from .hu_turn3_model import HU_FEATURE_DIM
from .teacher import DEFAULT_FL_EV
from .train_torch_action_value import parse_hidden_layers, select_device
from .turn3_model import _build_torch_mlp

REGRESSION_HEADS = ("ev", "delta_vs_baseline", "delta_vs_reference", "rank_score")
GATE_ID_TO_LABEL = {0: "negative", 1: "gray", 2: "positive"}
THRESHOLD_T2_VALUES = (5.0, 8.0, 10.0, 12.0)
THRESHOLD_REFERENCE_VALUES = (10.0, 15.0, 20.0, 25.0)
CURRENT_FL_EV_14 = float(DEFAULT_FL_EV[14])
RISK_ONLY_LABEL_FIELDS = (
    "requires_separate_whole_game_risk_head",
    "do_not_use_as_local_ev_hard_negative",
    "use_for_whole_game_risk_head",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("models/hu_turn2_stage8_pilot_2000_mc512_reference_override_cached_rank_wide.pt"),
    )
    parser.add_argument(
        "--init-from-model",
        type=Path,
        default=None,
        help=(
            "Optional existing hu_turn2_pilot_multihead_mlp checkpoint to fine-tune from. "
            "The architecture must match --hidden-layer-sizes and the cache feature dimension."
        ),
    )
    parser.add_argument(
        "--init-stats-source",
        choices=("init_model", "cache"),
        default="init_model",
        help=(
            "Normalization stats to use when --init-from-model is set. "
            "init_model preserves the checkpoint's input/output scale for fine-tuning."
        ),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-action-rows", type=int, default=32768)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--hidden-layer-sizes", default="1024,512,256")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026061702)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--ranking-loss-weight", type=float, default=0.05)
    parser.add_argument("--listwise-loss-weight", type=float, default=0.05)
    parser.add_argument("--gate-loss-weight", type=float, default=0.2)
    parser.add_argument("--gate-negative-weight", type=float, default=2.0)
    parser.add_argument(
        "--early-stop-metric",
        choices=(
            "val_avg_regret",
            "val_top3_recall",
            "val_top5_avg_regret",
            "val_top5_recall",
            "val_top10_avg_regret",
            "val_top10_recall",
        ),
        default="val_avg_regret",
        help="Metric used for best checkpoint selection. Use val_top5_avg_regret for Stage9 candidate generation.",
    )
    parser.add_argument(
        "--auxiliary-source-bucket",
        action="append",
        default=[],
        help=(
            "Treat matching source_bucket rows as auxiliary partial teachers. "
            "By default these rows keep gate loss but do not affect EV/ranking/listwise losses."
        ),
    )
    parser.add_argument("--auxiliary-regression-weight", type=float, default=0.0)
    parser.add_argument("--auxiliary-ranking-weight", type=float, default=0.0)
    parser.add_argument("--auxiliary-listwise-weight", type=float, default=0.0)
    parser.add_argument("--auxiliary-gate-weight", type=float, default=1.0)
    parser.add_argument(
        "--stage8b-labels-csv",
        type=Path,
        default=None,
        help="Optional Stage8b state-level safe override label CSV.",
    )
    parser.add_argument(
        "--gate-label-column",
        default="safe_lcb196_gate_label_id",
        help="Column in --stage8b-labels-csv to use as gate_label_id override.",
    )
    parser.add_argument(
        "--gate-weight-column",
        default="stage8b_gate_weight",
        help="Optional per-state gate loss weight column in --stage8b-labels-csv.",
    )
    parser.add_argument(
        "--allow-missing-scoring-metadata",
        action="store_true",
        help="Allow training from a cache without scoring_objective.fl_ev_14 metadata.",
    )
    parser.add_argument(
        "--allow-scoring-mismatch",
        action="store_true",
        help="Allow training from a cache whose fl_ev_14 differs from the current config.",
    )
    parser.add_argument(
        "--candidate-generator-training-jsonl",
        action="append",
        type=Path,
        default=[],
        help=(
            "Optional Stage9 candidate-generator training rows JSONL. "
            "Rows with action_row_index/state_index/weight boost action regression and state ranking/listwise losses."
        ),
    )
    parser.add_argument(
        "--candidate-pairwise-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Extra Stage9 candidate-generator loss. High-MC positive rows are pushed above the current TopK boundary; "
            "high-MC hard negatives are pushed below baseline. This is off by default."
        ),
    )
    parser.add_argument("--candidate-pairwise-margin", type=float, default=0.25)
    parser.add_argument("--candidate-pairwise-topk", type=int, default=5)
    parser.add_argument("--threshold-split", choices=("val", "test", "holdout"), default="test")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def truthy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def load_cache(cache_dir: Path) -> dict[str, Any]:
    metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    expected = {
        "schema": FEATURE_CACHE_SCHEMA,
        "observation_schema": OBSERVATION_SCHEMA,
        "policy_feature_sample_schema": POLICY_FEATURE_SAMPLE_SCHEMA,
        "action_key_schema": ACTION_KEY_SCHEMA,
        "feature_value_schema": FEATURE_VALUE_SCHEMA,
        "rules_digest": REGULAR_RULES_DIGEST,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"feature cache {key} mismatch: expected {value!r}, got {metadata.get(key)!r}"
            )
    if int(metadata.get("feature_dim", -1)) != HU_FEATURE_DIM:
        raise ValueError("feature cache dimension mismatch")
    identity_encoded = json.dumps(
        cache_identity_payload(metadata),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    expected_manifest_digest = hashlib.sha256(identity_encoded).hexdigest()
    if metadata.get("cache_manifest_digest") != expected_manifest_digest:
        raise ValueError("feature cache manifest digest mismatch")
    for item in metadata.get("input_files", ()):
        if not isinstance(item, dict) or not isinstance(item.get("sha256"), str):
            raise ValueError("feature cache input file digest is missing")
        path = Path(str(item.get("path", "")))
        if path.is_file() and sha256_file(path) != item["sha256"]:
            raise ValueError(f"feature cache input SHA-256 mismatch: {path}")
    state_count = int(metadata["state_count"])
    action_count = int(metadata["action_count"])
    if state_count <= 0 or action_count <= 0:
        raise ValueError("feature cache counts must be positive")
    feature_dtype = np.dtype(metadata.get("feature_dtype", "float32"))
    features_path = cache_dir / f"features.{feature_dtype.name}.mmap"
    expected_feature_bytes = action_count * HU_FEATURE_DIM * feature_dtype.itemsize
    if features_path.stat().st_size != expected_feature_bytes:
        raise ValueError("feature cache mmap size mismatch")
    features = np.memmap(
        features_path,
        dtype=feature_dtype,
        mode="r",
        shape=(action_count, int(metadata.get("feature_dim", HU_FEATURE_DIM))),
    )
    return {
        "metadata": metadata,
        "state_metadata": read_jsonl(cache_dir / "state_metadata.jsonl"),
        "features": features,
        "target_ev": np.memmap(cache_dir / "target_ev.float32.mmap", dtype=np.float32, mode="r", shape=(action_count,)),
        "target_delta_baseline": np.memmap(cache_dir / "target_delta_baseline.float32.mmap", dtype=np.float32, mode="r", shape=(action_count,)),
        "target_delta_reference": np.memmap(cache_dir / "target_delta_reference.float32.mmap", dtype=np.float32, mode="r", shape=(action_count,)),
        "target_rank_score": np.memmap(cache_dir / "target_rank_score.float32.mmap", dtype=np.float32, mode="r", shape=(action_count,)),
        "action_ev_se": np.memmap(cache_dir / "action_ev_se.float32.mmap", dtype=np.float32, mode="r", shape=(action_count,)),
        "offsets": np.load(cache_dir / "sample_offsets.npy"),
        "action_counts": np.load(cache_dir / "sample_action_counts.npy"),
        "split": np.load(cache_dir / "state_split.npy"),
        "baseline_action_index": np.load(cache_dir / "baseline_action_index.npy"),
        "reference_action_index": np.load(cache_dir / "reference_action_index.npy"),
        "fallback_action_index": np.load(cache_dir / "fallback_action_index.npy"),
        "best_action_index": np.load(cache_dir / "best_action_index.npy"),
        "second_best_action_index": np.load(cache_dir / "second_best_action_index.npy"),
        "gate_label_id": np.load(cache_dir / "gate_label_id.npy"),
    }


def _optional_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def _candidate_row_allows_target_override(row: dict[str, Any], label: str) -> bool:
    schema = str(row.get("schema", ""))
    if schema == "hu_turn2_stage9_high_mc_training_row_v1":
        return True
    if label.startswith("high_mc_"):
        return True
    try:
        mc_n = int(row.get("mc_n", 0))
    except (TypeError, ValueError):
        mc_n = 0
    return mc_n > 0


def load_candidate_generator_training_adjustments(
    cache: dict[str, Any],
    paths: Iterable[Path],
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, dict[str, Any] | None]:
    paths = [Path(path) for path in paths if path]
    if not paths:
        return None, None, None, None
    action_count = int(cache["metadata"]["action_count"])
    state_count = int(cache["metadata"]["state_count"])
    action_weights = np.ones(action_count, dtype=np.float32)
    state_weights = np.ones(state_count, dtype=np.float32)
    target_overrides = np.full((action_count, len(REGRESSION_HEADS)), np.nan, dtype=np.float32)
    label_counts: Counter[str] = Counter()
    invalid_rows = 0
    applied_rows = 0
    target_override_rows = 0
    target_override_cells = 0
    max_weight = 1.0
    target_fields = {
        "target_ev": 0,
        "delta_vs_baseline": 1,
        "target_delta_baseline": 1,
        "delta_vs_reference": 2,
        "target_delta_reference": 2,
        "target_rank_score": 3,
        "rank_score": 3,
    }
    for path in paths:
        for row in read_jsonl(path):
            label = str(row.get("label", "unknown"))
            label_counts[label] += 1
            try:
                action_row = int(row.get("action_row_index"))
                state_index = int(row.get("state_index"))
                weight = float(row.get("weight", 1.0))
            except (TypeError, ValueError):
                invalid_rows += 1
                continue
            if not math.isfinite(weight) or weight <= 0.0:
                invalid_rows += 1
                continue
            if not (0 <= action_row < action_count) or not (0 <= state_index < state_count):
                invalid_rows += 1
                continue
            action_weights[action_row] = max(float(action_weights[action_row]), weight)
            state_weights[state_index] = max(float(state_weights[state_index]), min(weight, 4.0))
            row_override_cells = 0
            if _candidate_row_allows_target_override(row, label):
                for field, column_index in target_fields.items():
                    value = _optional_float(row.get(field))
                    if value is None:
                        continue
                    target_overrides[action_row, column_index] = value
                    row_override_cells += 1
            if row_override_cells:
                target_override_rows += 1
                target_override_cells += row_override_cells
            max_weight = max(max_weight, weight)
            applied_rows += 1
    metadata = {
        "source": [str(path) for path in paths],
        "applied_rows": applied_rows,
        "invalid_rows": invalid_rows,
        "label_counts": dict(sorted(label_counts.items())),
        "boosted_action_rows": int(np.sum(action_weights > 1.0)),
        "boosted_states": int(np.sum(state_weights > 1.0)),
        "target_override_rows": int(target_override_rows),
        "target_override_cells": int(target_override_cells),
        "max_weight": float(max_weight),
    }
    if target_override_rows == 0:
        target_overrides = None
    return action_weights, state_weights, target_overrides, metadata


def load_candidate_generator_training_weights(
    cache: dict[str, Any],
    paths: Iterable[Path],
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any] | None]:
    action_weights, state_weights, _target_overrides, metadata = load_candidate_generator_training_adjustments(
        cache,
        paths,
    )
    return action_weights, state_weights, metadata


def _is_high_mc_positive_label(label: str) -> bool:
    return label.startswith("high_mc_lcb") and label.endswith("_positive")


def _is_high_mc_negative_label(label: str) -> bool:
    return label == "high_mc_hard_negative"


def load_candidate_pairwise_training_labels(
    cache: dict[str, Any],
    paths: Iterable[Path],
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any] | None]:
    paths = [Path(path) for path in paths if path]
    if not paths:
        return None, None, None
    action_count = int(cache["metadata"]["action_count"])
    state_count = int(cache["metadata"]["state_count"])
    positive_weights = np.zeros(action_count, dtype=np.float32)
    negative_weights = np.zeros(action_count, dtype=np.float32)
    label_counts: Counter[str] = Counter()
    invalid_rows = 0
    positive_rows = 0
    negative_rows = 0
    positive_states: set[int] = set()
    negative_states: set[int] = set()
    for path in paths:
        for row in read_jsonl(path):
            label = str(row.get("label", "unknown"))
            if not (_is_high_mc_positive_label(label) or _is_high_mc_negative_label(label)):
                continue
            label_counts[label] += 1
            try:
                action_row = int(row.get("action_row_index"))
                state_index = int(row.get("state_index"))
                weight = float(row.get("weight", 1.0))
            except (TypeError, ValueError):
                invalid_rows += 1
                continue
            if not math.isfinite(weight) or weight <= 0.0:
                invalid_rows += 1
                continue
            if not (0 <= action_row < action_count) or not (0 <= state_index < state_count):
                invalid_rows += 1
                continue
            if _is_high_mc_positive_label(label):
                positive_weights[action_row] = max(float(positive_weights[action_row]), weight)
                positive_rows += 1
                positive_states.add(state_index)
            elif _is_high_mc_negative_label(label):
                negative_weights[action_row] = max(float(negative_weights[action_row]), weight)
                negative_rows += 1
                negative_states.add(state_index)
    metadata = {
        "source": [str(path) for path in paths],
        "invalid_rows": invalid_rows,
        "label_counts": dict(sorted(label_counts.items())),
        "positive_rows": int(positive_rows),
        "negative_rows": int(negative_rows),
        "positive_states": int(len(positive_states)),
        "negative_states": int(len(negative_states)),
    }
    if positive_rows == 0 and negative_rows == 0:
        return None, None, metadata
    return positive_weights, negative_weights, metadata


def apply_candidate_target_overrides(targets: np.ndarray, overrides: np.ndarray | None) -> np.ndarray:
    if overrides is None:
        return targets
    if overrides.shape != targets.shape:
        raise ValueError(f"target override shape {overrides.shape} does not match targets {targets.shape}")
    output = np.array(targets, dtype=np.float32, copy=True)
    mask = np.isfinite(overrides)
    output[mask] = overrides[mask]
    return output


def scoring_metadata_status(
    metadata: dict[str, Any],
    *,
    expected_fl_ev_14: float = CURRENT_FL_EV_14,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    scoring = metadata.get("scoring_objective")
    if not isinstance(scoring, dict):
        return {
            "status": "missing",
            "training_allowed": False,
            "expected_fl_ev_14": expected_fl_ev_14,
            "cache_fl_ev_14": None,
            "reason": "metadata.scoring_objective is missing",
        }
    value = scoring.get("fl_ev_14")
    if value in (None, ""):
        return {
            "status": "missing",
            "training_allowed": False,
            "expected_fl_ev_14": expected_fl_ev_14,
            "cache_fl_ev_14": None,
            "reason": "metadata.scoring_objective.fl_ev_14 is missing or not unique",
            "fl_ev_14_values": scoring.get("fl_ev_14_values", {}),
            "fl_ev_14_status": scoring.get("fl_ev_14_status"),
        }
    try:
        cache_value = float(value)
    except (TypeError, ValueError):
        return {
            "status": "invalid",
            "training_allowed": False,
            "expected_fl_ev_14": expected_fl_ev_14,
            "cache_fl_ev_14": value,
            "reason": "metadata.scoring_objective.fl_ev_14 is not numeric",
        }
    matches = abs(cache_value - expected_fl_ev_14) <= tolerance
    return {
        "status": "match" if matches else "mismatch",
        "training_allowed": matches,
        "expected_fl_ev_14": expected_fl_ev_14,
        "cache_fl_ev_14": cache_value,
        "reason": "ok" if matches else "cache fl_ev_14 differs from current config",
        "fl_ev_14_values": scoring.get("fl_ev_14_values", {}),
        "fl_ev_14_status": scoring.get("fl_ev_14_status"),
    }


def validate_cache_scoring_for_training(
    cache: dict[str, Any],
    *,
    allow_missing: bool = False,
    allow_mismatch: bool = False,
) -> dict[str, Any]:
    status = scoring_metadata_status(cache["metadata"])
    if status["status"] == "match":
        return status
    if status["status"] == "missing" and allow_missing:
        return status | {"training_allowed": True, "override": "allow_missing_scoring_metadata"}
    if status["status"] == "mismatch" and allow_mismatch:
        return status | {"training_allowed": True, "override": "allow_scoring_mismatch"}
    raise ValueError(
        "feature cache scoring metadata is not compatible with current training objective: "
        f"{status}. Rebuild/reroll the cache under current FL EV or pass an explicit override for analysis only."
    )


def load_stage8b_gate_labels(
    labels_csv: Path,
    *,
    state_count: int,
    label_column: str,
    weight_column: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    labels = np.full(state_count, 1, dtype=np.int8)
    weights = np.ones(state_count, dtype=np.float32)
    seen = np.zeros(state_count, dtype=np.bool_)
    counts: Counter[str] = Counter()
    with labels_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "state_index" not in (reader.fieldnames or ()):
            raise ValueError(f"{labels_csv} is missing state_index")
        if label_column not in (reader.fieldnames or ()):
            raise ValueError(f"{labels_csv} is missing {label_column}")
        for row in reader:
            if (
                str(row.get("recommended_training_use", "")).strip() == "whole_game_risk_only"
                or any(truthy(row.get(field)) for field in RISK_ONLY_LABEL_FIELDS)
            ):
                raise ValueError(
                    f"{labels_csv} contains whole-game risk-only rows; "
                    "do not use them as local EV/safe-LCB gate labels"
                )
            state_index = int(row["state_index"])
            if state_index < 0 or state_index >= state_count:
                raise ValueError(f"state_index out of range in {labels_csv}: {state_index}")
            label_id = int(float(row[label_column]))
            if label_id not in (0, 1, 2):
                raise ValueError(f"invalid gate label id in {labels_csv}: {label_id}")
            labels[state_index] = label_id
            if weight_column and weight_column in row and row[weight_column] not in (None, ""):
                weights[state_index] = max(0.0, float(row[weight_column]))
            seen[state_index] = True
            counts[str(label_id)] += 1
    missing = int(np.size(seen) - int(np.sum(seen)))
    if missing:
        raise ValueError(f"{labels_csv} is missing labels for {missing} states")
    return labels, weights, {"label_column": label_column, "weight_column": weight_column, "counts": dict(counts)}


def apply_stage8b_gate_labels(
    cache: dict[str, Any],
    labels_csv: Path,
    *,
    label_column: str,
    weight_column: str,
) -> dict[str, Any]:
    labels, weights, metadata = load_stage8b_gate_labels(
        labels_csv,
        state_count=len(cache["state_metadata"]),
        label_column=label_column,
        weight_column=weight_column,
    )
    cache["gate_label_id"] = labels
    cache["gate_label_weight"] = weights
    cache["gate_label_metadata"] = metadata | {"source": str(labels_csv)}
    return metadata


def build_loss_weight_metadata(
    state_metadata: list[dict[str, Any]],
    *,
    auxiliary_source_buckets: Iterable[str],
    auxiliary_regression_weight: float,
    auxiliary_ranking_weight: float,
    auxiliary_listwise_weight: float,
    auxiliary_gate_weight: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    source_set = {str(item).strip() for item in auxiliary_source_buckets if str(item).strip()}
    state_count = len(state_metadata)
    regression = np.ones(state_count, dtype=np.float32)
    ranking = np.ones(state_count, dtype=np.float32)
    listwise = np.ones(state_count, dtype=np.float32)
    gate = np.ones(state_count, dtype=np.float32)
    auxiliary_indices: list[int] = []
    auxiliary_source_counts: Counter[str] = Counter()
    if source_set:
        for index, item in enumerate(state_metadata):
            source_bucket = str(item.get("source_bucket", ""))
            if source_bucket in source_set:
                regression[index] = max(0.0, float(auxiliary_regression_weight))
                ranking[index] = max(0.0, float(auxiliary_ranking_weight))
                listwise[index] = max(0.0, float(auxiliary_listwise_weight))
                gate[index] = max(0.0, float(auxiliary_gate_weight))
                auxiliary_indices.append(index)
                auxiliary_source_counts[source_bucket] += 1
    arrays = {
        "regression": regression,
        "ranking": ranking,
        "listwise": listwise,
        "gate": gate,
    }
    metadata = {
        "auxiliary_source_buckets": sorted(source_set),
        "auxiliary_state_count": len(auxiliary_indices),
        "auxiliary_source_counts": dict(auxiliary_source_counts),
        "auxiliary_regression_weight": float(auxiliary_regression_weight),
        "auxiliary_ranking_weight": float(auxiliary_ranking_weight),
        "auxiliary_listwise_weight": float(auxiliary_listwise_weight),
        "auxiliary_gate_weight": float(auxiliary_gate_weight),
    }
    return arrays, metadata


def state_indices_for_split(split: np.ndarray, name: str) -> np.ndarray:
    if name == "holdout":
        return np.where(split != SPLIT_NAME_TO_ID["train"])[0]
    return np.where(split == SPLIT_NAME_TO_ID[name])[0]


def action_indices_for_states(offsets: np.ndarray, state_indices: Iterable[int]) -> np.ndarray:
    parts = [np.arange(int(offsets[i]), int(offsets[i + 1]), dtype=np.int64) for i in state_indices]
    if not parts:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(parts)


def action_indices_for_weighted_states(
    offsets: np.ndarray,
    state_indices: Iterable[int],
    state_weights: np.ndarray,
) -> np.ndarray:
    weighted_states = [int(index) for index in state_indices if float(state_weights[int(index)]) > 0.0]
    return action_indices_for_states(offsets, weighted_states)


def target_matrix(cache: dict[str, Any]) -> np.ndarray:
    return np.column_stack(
        [
            np.asarray(cache["target_ev"], dtype=np.float32),
            np.asarray(cache["target_delta_baseline"], dtype=np.float32),
            np.asarray(cache["target_delta_reference"], dtype=np.float32),
            np.asarray(cache["target_rank_score"], dtype=np.float32),
        ]
    ).astype(np.float32, copy=False)


def normalize_stats(features: np.ndarray, targets: np.ndarray, action_indices: np.ndarray) -> dict[str, np.ndarray]:
    x = np.asarray(features[action_indices], dtype=np.float32)
    y = targets[action_indices].astype(np.float32, copy=False)
    feature_mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale = x.std(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale[feature_scale < 1e-6] = 1.0
    target_mean = y.mean(axis=0, dtype=np.float64).astype(np.float32)
    target_scale = y.std(axis=0, dtype=np.float64).astype(np.float32)
    target_scale[target_scale < 1e-6] = 1.0
    return {
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "target_mean": target_mean,
        "target_scale": target_scale,
    }


def _as_float32_vector(value: Any, *, name: str, length: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {array.shape}")
    return array


def checkpoint_stats(payload: dict[str, Any], *, feature_dim: int) -> dict[str, np.ndarray]:
    return {
        "feature_mean": _as_float32_vector(payload.get("feature_mean"), name="feature_mean", length=feature_dim),
        "feature_scale": _as_float32_vector(payload.get("feature_scale"), name="feature_scale", length=feature_dim),
        "target_mean": _as_float32_vector(payload.get("target_mean"), name="target_mean", length=len(REGRESSION_HEADS)),
        "target_scale": _as_float32_vector(payload.get("target_scale"), name="target_scale", length=len(REGRESSION_HEADS)),
    }


def load_initial_checkpoint_payload(
    torch,
    path: Path,
    *,
    feature_dim: int,
    hidden_layers: tuple[int, ...],
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if str(payload.get("model_kind")) != "hu_turn2_pilot_multihead_mlp":
        raise ValueError(f"unsupported init model kind: {payload.get('model_kind')!r}")
    checkpoint_feature_dim = int(payload.get("feature_dim", -1))
    if checkpoint_feature_dim != int(feature_dim):
        raise ValueError(f"init model feature_dim {checkpoint_feature_dim} does not match cache feature_dim {feature_dim}")
    checkpoint_hidden = tuple(int(value) for value in payload.get("hidden_layer_sizes", ()))
    if checkpoint_hidden != tuple(int(value) for value in hidden_layers):
        raise ValueError(
            "init model hidden_layer_sizes "
            f"{checkpoint_hidden} does not match requested hidden_layer_sizes {tuple(hidden_layers)}"
        )
    if "state_dict" not in payload:
        raise ValueError("init model checkpoint is missing state_dict")
    checkpoint_stats(payload, feature_dim=feature_dim)
    return payload


def iter_state_batches(
    state_indices: np.ndarray,
    offsets: np.ndarray,
    *,
    batch_action_rows: int,
    rng: np.random.Generator,
) -> Iterable[np.ndarray]:
    shuffled = state_indices.copy()
    rng.shuffle(shuffled)
    current: list[int] = []
    action_rows = 0
    for state_index in shuffled:
        length = int(offsets[state_index + 1] - offsets[state_index])
        if current and action_rows + length > batch_action_rows:
            yield np.asarray(current, dtype=np.int64)
            current = []
            action_rows = 0
        current.append(int(state_index))
        action_rows += length
    if current:
        yield np.asarray(current, dtype=np.int64)


def grouped_action_indices(offsets: np.ndarray, state_indices: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    parts: list[np.ndarray] = []
    groups: list[tuple[int, int, int]] = []
    cursor = 0
    for state_index in state_indices:
        start = int(offsets[state_index])
        end = int(offsets[state_index + 1])
        length = end - start
        parts.append(np.arange(start, end, dtype=np.int64))
        groups.append((int(state_index), cursor, cursor + length))
        cursor += length
    if not parts:
        return np.zeros(0, dtype=np.int64), []
    return np.concatenate(parts), groups


def ranking_loss(torch, pred_ev, target_ev, groups: list[tuple[int, int, int]], state_weights: np.ndarray | None = None):
    losses = []
    weights = []
    for state_index, start, end in groups:
        state_weight = 1.0 if state_weights is None else float(state_weights[state_index])
        if state_weight <= 0.0:
            continue
        if end - start <= 1:
            continue
        group_pred = pred_ev[start:end]
        group_target = target_ev[start:end]
        best = int(torch.argmax(group_target).item())
        teacher_gap = (group_target[best] - group_target).clamp(min=0.0, max=2.0)
        pred_gap = group_pred[best] - group_pred
        mask = teacher_gap > 1e-6
        if bool(mask.any()):
            losses.append(torch.relu(teacher_gap[mask] - pred_gap[mask]).square().mean())
            weights.append(state_weight)
    if not losses:
        return pred_ev.sum() * 0.0
    loss_tensor = torch.stack(losses)
    weight_tensor = torch.tensor(weights, dtype=loss_tensor.dtype, device=loss_tensor.device)
    return (loss_tensor * weight_tensor).sum() / weight_tensor.sum().clamp_min(1.0)


def listwise_loss(
    torch,
    pred_ev,
    target_ev,
    groups: list[tuple[int, int, int]],
    *,
    temperature: float = 1.0,
    state_weights: np.ndarray | None = None,
):
    losses = []
    weights = []
    for state_index, start, end in groups:
        state_weight = 1.0 if state_weights is None else float(state_weights[state_index])
        if state_weight <= 0.0:
            continue
        if end - start <= 1:
            continue
        target_probs = torch.softmax(target_ev[start:end] / temperature, dim=0)
        log_probs = torch.log_softmax(pred_ev[start:end] / temperature, dim=0)
        losses.append(-(target_probs * log_probs).sum())
        weights.append(state_weight)
    if not losses:
        return pred_ev.sum() * 0.0
    loss_tensor = torch.stack(losses)
    weight_tensor = torch.tensor(weights, dtype=loss_tensor.dtype, device=loss_tensor.device)
    return (loss_tensor * weight_tensor).sum() / weight_tensor.sum().clamp_min(1.0)


def candidate_pairwise_loss(
    torch,
    pred_ev,
    groups: list[tuple[int, int, int]],
    action_indices: np.ndarray,
    positive_weights: np.ndarray | None,
    negative_weights: np.ndarray | None,
    baseline_action_index: np.ndarray,
    *,
    margin: float,
    topk: int,
):
    if positive_weights is None and negative_weights is None:
        return pred_ev.sum() * 0.0
    losses = []
    weights = []
    safe_topk = max(1, int(topk))
    for state_index, start, end in groups:
        if end - start <= 1:
            continue
        global_rows = action_indices[start:end]
        group_pred = pred_ev[start:end]
        if positive_weights is not None:
            local_positive = [
                local_index
                for local_index, action_row in enumerate(global_rows)
                if float(positive_weights[int(action_row)]) > 0.0
            ]
            for local_index in local_positive:
                competitors = torch.cat((group_pred[:local_index], group_pred[local_index + 1 :]))
                if competitors.numel() == 0:
                    continue
                kth = min(safe_topk, int(competitors.numel()))
                boundary = torch.topk(competitors.detach(), kth).values[-1]
                losses.append(torch.relu(boundary + float(margin) - group_pred[local_index]).square())
                weights.append(float(positive_weights[int(global_rows[local_index])]))
        if negative_weights is not None:
            baseline_local = int(baseline_action_index[state_index])
            if 0 <= baseline_local < end - start:
                baseline_pred = group_pred[baseline_local]
                local_negative = [
                    local_index
                    for local_index, action_row in enumerate(global_rows)
                    if float(negative_weights[int(action_row)]) > 0.0
                ]
                for local_index in local_negative:
                    losses.append(torch.relu(group_pred[local_index] + float(margin) - baseline_pred).square())
                    weights.append(float(negative_weights[int(global_rows[local_index])]))
    if not losses:
        return pred_ev.sum() * 0.0
    loss_tensor = torch.stack(losses)
    weight_tensor = torch.tensor(weights, dtype=loss_tensor.dtype, device=loss_tensor.device)
    return (loss_tensor * weight_tensor).sum() / weight_tensor.sum().clamp_min(1.0)


def gate_loss(
    torch,
    gate_logits,
    groups: list[tuple[int, int, int]],
    gate_labels: np.ndarray,
    *,
    negative_weight: float,
    gate_weights: np.ndarray | None = None,
):
    logits = []
    labels = []
    weights = []
    for state_index, start, end in groups:
        label_id = int(gate_labels[state_index])
        if label_id == 1:
            continue
        logits.append(gate_logits[start:end].mean())
        labels.append(1.0 if label_id == 2 else 0.0)
        base_weight = 1.0 if label_id == 2 else float(negative_weight)
        if gate_weights is not None:
            base_weight *= float(gate_weights[state_index])
        weights.append(base_weight)
    if not logits:
        return gate_logits.sum() * 0.0
    logit_tensor = torch.stack(logits)
    label_tensor = torch.tensor(labels, dtype=gate_logits.dtype, device=gate_logits.device)
    weight_tensor = torch.tensor(weights, dtype=gate_logits.dtype, device=gate_logits.device)
    return (
        torch.nn.functional.binary_cross_entropy_with_logits(
            logit_tensor,
            label_tensor,
            reduction="none",
        )
        * weight_tensor
    ).sum() / weight_tensor.sum().clamp_min(1.0)


def predict_all(torch, net, cache: dict[str, Any], stats: dict[str, np.ndarray], device: str, batch_size: int) -> np.ndarray:
    features = cache["features"]
    mean = torch.from_numpy(stats["feature_mean"]).to(device)
    scale = torch.from_numpy(stats["feature_scale"]).to(device)
    target_mean = stats["target_mean"].astype(np.float32)
    target_scale = stats["target_scale"].astype(np.float32)
    preds: list[np.ndarray] = []
    net.eval()
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            end = min(start + batch_size, features.shape[0])
            x = torch.from_numpy(np.array(features[start:end], dtype=np.float32, copy=True)).to(device)
            x = (x - mean) / scale
            out = net(x).detach().cpu().numpy().astype(np.float32)
            out[:, :4] = out[:, :4] * target_scale + target_mean
            preds.append(out)
    return np.vstack(preds) if preds else np.zeros((0, 5), dtype=np.float32)


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def corr(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def split_eval(
    cache: dict[str, Any],
    predictions: np.ndarray,
    targets: np.ndarray,
    split_name: str,
    state_indices: np.ndarray,
) -> dict[str, Any]:
    offsets = cache["offsets"]
    baseline_idx = cache["baseline_action_index"]
    gate_labels = cache["gate_label_id"]
    ev_errors: list[float] = []
    delta_b_errors: list[float] = []
    delta_r_errors: list[float] = []
    regrets: list[float] = []
    top5_regrets: list[float] = []
    top10_regrets: list[float] = []
    top1 = 0
    top3 = 0
    top5 = 0
    top10 = 0
    pair_correct = 0
    pair_total = 0
    pred_delta_values: list[float] = []
    teacher_delta_values: list[float] = []
    gate_correct = 0
    gate_total = 0
    for state_index in state_indices:
        start = int(offsets[state_index])
        end = int(offsets[state_index + 1])
        pred = predictions[start:end]
        y = targets[start:end]
        ev_errors.extend(np.abs(pred[:, 0] - y[:, 0]).astype(float).tolist())
        delta_b_errors.extend(np.abs(pred[:, 1] - y[:, 1]).astype(float).tolist())
        delta_r_errors.extend(np.abs(pred[:, 2] - y[:, 2]).astype(float).tolist())
        predicted_idx = int(np.argmax(pred[:, 0]))
        best_idx = int(np.argmax(y[:, 0]))
        best_ev = float(y[best_idx, 0])
        regrets.append(best_ev - float(y[predicted_idx, 0]))
        top1 += int(predicted_idx == best_idx)
        top_k = min(3, end - start)
        top_indices = np.argpartition(pred[:, 0], -top_k)[-top_k:]
        top3 += int(best_idx in set(int(i) for i in top_indices))
        top5_k = min(5, end - start)
        top5_indices = np.argpartition(pred[:, 0], -top5_k)[-top5_k:]
        top5_set = set(int(i) for i in top5_indices)
        top5 += int(best_idx in top5_set)
        top5_best_ev = max((float(y[i, 0]) for i in top5_set), default=-math.inf)
        top5_regrets.append(best_ev - top5_best_ev if math.isfinite(top5_best_ev) else 0.0)
        top10_k = min(10, end - start)
        top10_indices = np.argpartition(pred[:, 0], -top10_k)[-top10_k:]
        top10_set = set(int(i) for i in top10_indices)
        top10 += int(best_idx in top10_set)
        top10_best_ev = max((float(y[i, 0]) for i in top10_set), default=-math.inf)
        top10_regrets.append(best_ev - top10_best_ev if math.isfinite(top10_best_ev) else 0.0)
        for i in range(end - start):
            for j in range(i + 1, end - start):
                target_cmp = float(y[i, 0] - y[j, 0])
                if abs(target_cmp) <= 1e-9:
                    continue
                pred_cmp = float(pred[i, 0] - pred[j, 0])
                pair_correct += int((target_cmp > 0) == (pred_cmp > 0))
                pair_total += 1
        base = int(baseline_idx[state_index])
        pred_delta_values.append(float(pred[predicted_idx, 0] - pred[base, 0]))
        teacher_delta_values.append(float(y[predicted_idx, 0] - y[base, 0]))
        label = int(gate_labels[state_index])
        if label != 1:
            gate_logit = float(pred[:, 4].mean())
            gate_pred = 1 if gate_logit >= 0.0 else 0
            gate_true = 1 if label == 2 else 0
            gate_correct += int(gate_pred == gate_true)
            gate_total += 1

    state_count = int(state_indices.size)
    return {
        "split": split_name,
        "states": state_count,
        "actions": int(sum(int(offsets[i + 1] - offsets[i]) for i in state_indices)),
        "ev_mae": float(np.mean(ev_errors)) if ev_errors else 0.0,
        "ev_rmse": float(np.sqrt(np.mean(np.square(ev_errors)))) if ev_errors else 0.0,
        "delta_vs_baseline_mae": float(np.mean(delta_b_errors)) if delta_b_errors else 0.0,
        "delta_vs_reference_mae": float(np.mean(delta_r_errors)) if delta_r_errors else 0.0,
        "avg_regret": float(np.mean(regrets)) if regrets else 0.0,
        "p90_regret": quantile(regrets, 0.90),
        "p95_regret": quantile(regrets, 0.95),
        "p99_regret": quantile(regrets, 0.99),
        "top1_accuracy": float(top1 / state_count) if state_count else 0.0,
        "top3_recall": float(top3 / state_count) if state_count else 0.0,
        "top5_recall": float(top5 / state_count) if state_count else 0.0,
        "top5_avg_regret": float(np.mean(top5_regrets)) if top5_regrets else 0.0,
        "top10_recall": float(top10 / state_count) if state_count else 0.0,
        "top10_avg_regret": float(np.mean(top10_regrets)) if top10_regrets else 0.0,
        "pairwise_ranking_accuracy": float(pair_correct / pair_total) if pair_total else 0.0,
        "calibration_corr_predicted_delta_teacher_delta": corr(pred_delta_values, teacher_delta_values),
        "gate_accuracy_pos_neg": float(gate_correct / gate_total) if gate_total else 0.0,
        "gate_eval_states": gate_total,
    }


def early_stop_metric_value(val_eval: dict[str, Any], metric: str) -> float:
    if not metric.startswith("val_"):
        raise ValueError(f"unsupported early stop metric: {metric}")
    key = metric.removeprefix("val_")
    return float(val_eval[key])


def early_stop_metric_improved(metric: str, current: float, best: float) -> bool:
    maximize = metric.endswith("_recall")
    return current > best if maximize else current < best


def subset_rows(
    cache: dict[str, Any],
    predictions: np.ndarray,
    targets: np.ndarray,
    split_name: str,
    predicate,
    label: str,
) -> dict[str, Any]:
    split_indices = state_indices_for_split(cache["split"], split_name)
    subset = np.asarray(
        [int(index) for index in split_indices if predicate(cache["state_metadata"][int(index)])],
        dtype=np.int64,
    )
    row = split_eval(cache, predictions, targets, label, subset)
    row["group"] = label
    return row


def threshold_sweep(
    cache: dict[str, Any],
    predictions: np.ndarray,
    targets: np.ndarray,
    split_name: str,
) -> list[dict[str, Any]]:
    state_indices = state_indices_for_split(cache["split"], split_name)
    offsets = cache["offsets"]
    baseline_idx = cache["baseline_action_index"]
    metadata = cache["state_metadata"]
    rows: list[dict[str, Any]] = []
    for min_margin in THRESHOLD_T2_VALUES:
        for reference_margin in THRESHOLD_REFERENCE_VALUES:
            gains: list[float] = []
            losses: list[float] = []
            fired = 0
            evaluated = int(state_indices.size)
            first_fired = 0
            second_fired = 0
            first_false_positive = 0
            second_false_positive = 0
            bucket_counts: Counter[str] = Counter()
            bucket_false_positive: Counter[str] = Counter()
            for state_index in state_indices:
                start = int(offsets[state_index])
                end = int(offsets[state_index + 1])
                base = int(baseline_idx[state_index])
                pred_delta = predictions[start:end, 1]
                candidate = int(np.argmax(pred_delta))
                state_reference_margin = float(metadata[int(state_index)].get("baseline_model_margin", 0.0) or 0.0)
                if candidate == base:
                    continue
                if float(pred_delta[candidate]) < min_margin:
                    continue
                if state_reference_margin < reference_margin:
                    continue
                gain = float(targets[start + candidate, 0] - targets[start + base, 0])
                gains.append(gain)
                losses.append(max(0.0, -gain))
                fired += 1
                seat = str(metadata[int(state_index)].get("seat", "unknown"))
                bucket = str(metadata[int(state_index)].get("bucket_group", "unknown"))
                bucket_counts[bucket] += 1
                false_positive = gain < 0.0
                if false_positive:
                    bucket_false_positive[bucket] += 1
                if seat == "first":
                    first_fired += 1
                    first_false_positive += int(false_positive)
                elif seat == "second":
                    second_fired += 1
                    second_false_positive += int(false_positive)
            false_positive_count = sum(1 for value in gains if value < 0.0)
            rows.append(
                {
                    "split": split_name,
                    "hu_turn2_min_margin": min_margin,
                    "hu_turn2_reference_min_margin": reference_margin,
                    "evaluated_states": evaluated,
                    "override_count": fired,
                    "override_rate": fired / evaluated if evaluated else 0.0,
                    "teacher_avg_gain_on_override": float(np.mean(gains)) if gains else 0.0,
                    "false_positive_count": false_positive_count,
                    "false_positive_rate": false_positive_count / fired if fired else 0.0,
                    "p95_loss": quantile(losses, 0.95),
                    "p99_loss": quantile(losses, 0.99),
                    "first_override_count": first_fired,
                    "second_override_count": second_fired,
                    "first_false_positive_count": first_false_positive,
                    "second_false_positive_count": second_false_positive,
                    "bucket_override_counts": json.dumps(dict(bucket_counts), sort_keys=True),
                    "bucket_false_positive_counts": json.dumps(dict(bucket_false_positive), sort_keys=True),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def teacher_mc_label_from_distribution(distribution: Any) -> str:
    if not isinstance(distribution, dict) or not distribution:
        return "unknown teacher MC"
    counts: list[tuple[int, int]] = []
    for key, value in distribution.items():
        try:
            mc = int(key)
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count > 0:
            counts.append((mc, count))
    if not counts:
        return "unknown teacher MC"
    counts.sort()
    if len(counts) == 1:
        return f"MC{counts[0][0]}"
    return "mixed " + "/".join(f"MC{mc}" for mc, _count in counts)


def training_t3_continuation_metadata(cache: dict[str, Any]) -> dict[str, Any]:
    metadata = cache.get("metadata", {})
    cached = metadata.get("t3_continuation_metadata")
    if isinstance(cached, dict) and cached:
        return cached
    state_metadata = cache.get("state_metadata", [])
    mode_counts = Counter(str(row.get("t3_continuation") or "legacy_unspecified") for row in state_metadata)
    policy_counts = Counter(str(row.get("continuation_policy_T3") or "legacy_unspecified") for row in state_metadata)
    primary_mode = next(iter(mode_counts)) if len(mode_counts) == 1 else "mixed"
    primary_policy = next(iter(policy_counts)) if len(policy_counts) == 1 else "mixed"
    return {
        "t3_continuation": primary_mode,
        "continuation_policy_T3": primary_policy,
        "t3_continuation_counts": dict(mode_counts),
        "continuation_policy_T3_counts": dict(policy_counts),
    }


def t3_margin_metadata(t3_metadata: dict[str, Any]) -> tuple[float | None, float | None]:
    mode = str(t3_metadata.get("t3_continuation") or "")
    if mode == "stage7_m5_r10":
        return 5.0, 10.0
    if mode == "stage3_reference_default":
        return 0.0, 0.0
    return None, None


def write_markdown(path: Path, summary: dict[str, Any], threshold_rows: list[dict[str, Any]]) -> None:
    test = summary["eval"]["test"]
    val = summary["eval"]["val"]
    total_states = sum(int(value) for value in summary.get("split_counts", {}).values())
    state_label = f"{total_states // 1000}k" if total_states and total_states % 1000 == 0 else str(total_states)
    title_stage = "Stage8b Safe Override" if summary.get("gate_label_source") == "stage8b_labels_csv" else "Stage8 Broad"
    teacher_mc_label = summary.get("teacher_mc_label") or teacher_mc_label_from_distribution(
        summary.get("teacher_rollout_count_distribution")
    )
    positive_thresholds = [
        row
        for row in threshold_rows
        if row["override_count"] > 0 and row["teacher_avg_gain_on_override"] > 0.0
    ]
    lines = [
        f"# HU T2 {title_stage} {state_label} {teacher_mc_label} Training",
        "",
        "This is a broad teacher-cache training artifact, not a production candidate.",
        "",
        f"- model: `{summary['model_output']}`",
        f"- cache: `{summary['cache_dir']}`",
        f"- teacher rollout counts: `{summary.get('teacher_rollout_count_distribution', {})}`",
        f"- device: `{summary['device']}`",
        f"- gate label source: `{summary.get('gate_label_source', 'cache_gate_label_id')}`",
        f"- loss weight metadata: `{summary.get('loss_weight_metadata', {})}`",
        f"- scoring objective status: `{summary.get('scoring_metadata_status', {}).get('status', 'unknown')}`",
        f"- T3 continuation metadata: `{summary.get('t3_continuation_metadata', {})}`",
        f"- epochs ran: `{summary['epochs_ran']}`",
        f"- best epoch: `{summary['best_epoch']}`",
        f"- early stop metric: `{summary.get('early_stop_metric', 'val_avg_regret')}` = `{summary.get('best_metric_value', 0.0):.4f}`",
        f"- train/val/test states: `{summary['split_counts']['train']}` / `{summary['split_counts']['val']}` / `{summary['split_counts']['test']}`",
        "",
        "## Holdout Metrics",
        "",
        f"- val EV MAE / avg_regret / top3: `{val['ev_mae']:.4f}` / `{val['avg_regret']:.4f}` / `{val['top3_recall']:.4f}`",
        f"- val top5 recall / top5 avg_regret: `{val.get('top5_recall', 0.0):.4f}` / `{val.get('top5_avg_regret', 0.0):.4f}`",
        f"- test EV MAE / avg_regret / top3: `{test['ev_mae']:.4f}` / `{test['avg_regret']:.4f}` / `{test['top3_recall']:.4f}`",
        f"- test top5 recall / top5 avg_regret: `{test.get('top5_recall', 0.0):.4f}` / `{test.get('top5_avg_regret', 0.0):.4f}`",
        f"- test delta baseline MAE: `{test['delta_vs_baseline_mae']:.4f}`",
        f"- test pairwise ranking accuracy: `{test['pairwise_ranking_accuracy']:.4f}`",
        f"- test gate accuracy pos/neg: `{test['gate_accuracy_pos_neg']:.4f}`",
        "",
        "## Threshold Sweep",
        "",
        f"- configs with positive teacher gain and at least one override: `{len(positive_thresholds)}`",
        "- This sweep is teacher-holdout only; do not use it as production evidence.",
        "- `reference_margin_raw` is a T2 baseline/reference score margin and is not comparable to the T3 Stage7 `r10` reference gate.",
        "",
        "## Next Step",
        "",
        summary["recommended_next_step"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise SystemExit("epochs must be positive")
    if args.batch_action_rows <= 0:
        raise SystemExit("batch-action-rows must be positive")
    started_at = time.time()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)

    import torch

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = select_device(torch, args.device)
    hidden_layers = parse_hidden_layers(args.hidden_layer_sizes)

    cache = load_cache(args.cache_dir.resolve())
    t3_metadata = training_t3_continuation_metadata(cache)
    hu_turn3_min_margin, hu_turn3_reference_min_margin = t3_margin_metadata(t3_metadata)
    scoring_status = validate_cache_scoring_for_training(
        cache,
        allow_missing=args.allow_missing_scoring_metadata,
        allow_mismatch=args.allow_scoring_mismatch,
    )
    gate_label_override = None
    if args.stage8b_labels_csv is not None:
        gate_label_override = apply_stage8b_gate_labels(
            cache,
            args.stage8b_labels_csv.resolve(),
            label_column=args.gate_label_column,
            weight_column=args.gate_weight_column,
        )
    loss_state_weights, loss_weight_metadata = build_loss_weight_metadata(
        cache["state_metadata"],
        auxiliary_source_buckets=args.auxiliary_source_bucket,
        auxiliary_regression_weight=args.auxiliary_regression_weight,
        auxiliary_ranking_weight=args.auxiliary_ranking_weight,
        auxiliary_listwise_weight=args.auxiliary_listwise_weight,
        auxiliary_gate_weight=args.auxiliary_gate_weight,
    )
    (
        candidate_action_weights,
        candidate_state_weights,
        candidate_target_overrides,
        candidate_weight_metadata,
    ) = load_candidate_generator_training_adjustments(
        cache,
        [path.resolve() for path in args.candidate_generator_training_jsonl],
    )
    candidate_pairwise_positive_weights = None
    candidate_pairwise_negative_weights = None
    candidate_pairwise_metadata = None
    if args.candidate_pairwise_loss_weight > 0.0:
        (
            candidate_pairwise_positive_weights,
            candidate_pairwise_negative_weights,
            candidate_pairwise_metadata,
        ) = load_candidate_pairwise_training_labels(
            cache,
            [path.resolve() for path in args.candidate_generator_training_jsonl],
        )
    if candidate_state_weights is not None:
        for key in ("regression", "ranking", "listwise"):
            loss_state_weights[key] = np.asarray(loss_state_weights[key], dtype=np.float32) * candidate_state_weights
        loss_weight_metadata["candidate_generator"] = candidate_weight_metadata
    if candidate_pairwise_metadata is not None:
        loss_weight_metadata["candidate_pairwise"] = {
            **candidate_pairwise_metadata,
            "loss_weight": float(args.candidate_pairwise_loss_weight),
            "margin": float(args.candidate_pairwise_margin),
            "topk": int(args.candidate_pairwise_topk),
        }
    targets = apply_candidate_target_overrides(target_matrix(cache), candidate_target_overrides)
    train_states = state_indices_for_split(cache["split"], "train")
    val_states = state_indices_for_split(cache["split"], "val")
    test_states = state_indices_for_split(cache["split"], "test")
    train_actions = action_indices_for_weighted_states(cache["offsets"], train_states, loss_state_weights["regression"])
    if train_actions.size == 0:
        train_actions = action_indices_for_states(cache["offsets"], train_states)
    cache_stats = normalize_stats(cache["features"], targets, train_actions)
    init_payload = None
    init_model_metadata: dict[str, Any] | None = None
    feature_dim = int(cache["metadata"]["feature_dim"])
    if args.init_from_model is not None:
        init_payload = load_initial_checkpoint_payload(
            torch,
            args.init_from_model.resolve(),
            feature_dim=feature_dim,
            hidden_layers=hidden_layers,
        )
        init_model_metadata = {
            "path": str(args.init_from_model.resolve()),
            "hidden_layer_sizes": [int(value) for value in init_payload.get("hidden_layer_sizes", [])],
            "dropout": float(init_payload.get("dropout", 0.0)),
            "stats_source": args.init_stats_source,
            "override_gate_semantics": init_payload.get("override_gate_semantics"),
            "scoring_metadata_status": init_payload.get("scoring_metadata_status"),
        }
    stats = (
        checkpoint_stats(init_payload, feature_dim=feature_dim)
        if init_payload is not None and args.init_stats_source == "init_model"
        else cache_stats
    )

    net = _build_torch_mlp(torch, feature_dim, hidden_layers, args.dropout, output_dim=5).to(device)
    if init_payload is not None:
        net.load_state_dict(init_payload["state_dict"])
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    mean_tensor = torch.from_numpy(stats["feature_mean"]).to(device)
    scale_tensor = torch.from_numpy(stats["feature_scale"]).to(device)
    target_mean_tensor = torch.from_numpy(stats["target_mean"]).to(device)
    target_scale_tensor = torch.from_numpy(stats["target_scale"]).to(device)
    regression_weights = torch.tensor([1.0, 0.7, 0.4, 0.25], dtype=torch.float32, device=device)

    best_state = None
    initial_eval = None
    best_metric_value = -float("inf") if args.early_stop_metric.endswith("_recall") else float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.seed)
    if init_payload is not None:
        initial_predictions = predict_all(torch, net, cache, stats, device, args.batch_action_rows)
        initial_eval = split_eval(cache, initial_predictions, targets, "val", val_states)
        best_metric_value = early_stop_metric_value(initial_eval, args.early_stop_metric)
        best_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}

    for epoch in range(1, args.epochs + 1):
        net.train()
        loss_sum = 0.0
        batch_count = 0
        for batch_states in iter_state_batches(
            train_states,
            cache["offsets"],
            batch_action_rows=args.batch_action_rows,
            rng=rng,
        ):
            action_indices, groups = grouped_action_indices(cache["offsets"], batch_states)
            x = torch.from_numpy(np.array(cache["features"][action_indices], dtype=np.float32, copy=True)).to(device)
            y = torch.from_numpy(targets[action_indices].astype(np.float32, copy=False)).to(device)
            x = (x - mean_tensor) / scale_tensor
            y_norm = (y - target_mean_tensor) / target_scale_tensor
            optimizer.zero_grad(set_to_none=True)
            pred = net(x)
            regression = torch.nn.functional.smooth_l1_loss(
                pred[:, :4],
                y_norm,
                reduction="none",
            )
            state_regression_weights = np.asarray(
                [float(loss_state_weights["regression"][int(state_index)]) for state_index, _start, _end in groups],
                dtype=np.float32,
            )
            action_regression_weights = torch.from_numpy(
                np.repeat(state_regression_weights, [end - start for _state_index, start, end in groups])
            ).to(device=device, dtype=regression.dtype)
            if candidate_action_weights is not None:
                action_boost = torch.from_numpy(
                    np.asarray(candidate_action_weights[action_indices], dtype=np.float32)
                ).to(device=device, dtype=regression.dtype)
                action_regression_weights = action_regression_weights * action_boost
            regression_loss_by_action = (regression * regression_weights).mean(dim=1)
            loss = (regression_loss_by_action * action_regression_weights).sum() / action_regression_weights.sum().clamp_min(1.0)
            if args.ranking_loss_weight > 0.0:
                loss = loss + args.ranking_loss_weight * ranking_loss(
                    torch,
                    pred[:, 0],
                    y_norm[:, 0],
                    groups,
                    loss_state_weights["ranking"],
                )
            if args.listwise_loss_weight > 0.0:
                loss = loss + args.listwise_loss_weight * listwise_loss(
                    torch,
                    pred[:, 0],
                    y_norm[:, 0],
                    groups,
                    state_weights=loss_state_weights["listwise"],
                )
            if args.candidate_pairwise_loss_weight > 0.0:
                loss = loss + args.candidate_pairwise_loss_weight * candidate_pairwise_loss(
                    torch,
                    pred[:, 0],
                    groups,
                    action_indices,
                    candidate_pairwise_positive_weights,
                    candidate_pairwise_negative_weights,
                    cache["baseline_action_index"],
                    margin=args.candidate_pairwise_margin,
                    topk=args.candidate_pairwise_topk,
                )
            if args.gate_loss_weight > 0.0:
                gate_weights = loss_state_weights["gate"]
                if cache.get("gate_label_weight") is not None:
                    gate_weights = gate_weights * np.asarray(cache["gate_label_weight"], dtype=np.float32)
                loss = loss + args.gate_loss_weight * gate_loss(
                    torch,
                    pred[:, 4],
                    groups,
                    cache["gate_label_id"],
                    negative_weight=args.gate_negative_weight,
                    gate_weights=gate_weights,
                )
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
            batch_count += 1

        predictions = predict_all(torch, net, cache, stats, device, args.batch_action_rows)
        val_eval = split_eval(cache, predictions, targets, "val", val_states)
        epoch_row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(batch_count, 1),
            "val_avg_regret": val_eval["avg_regret"],
            "val_ev_mae": val_eval["ev_mae"],
            "val_top3_recall": val_eval["top3_recall"],
            "val_top5_recall": val_eval["top5_recall"],
            "val_top5_avg_regret": val_eval["top5_avg_regret"],
            "early_stop_metric": args.early_stop_metric,
            "early_stop_metric_value": early_stop_metric_value(val_eval, args.early_stop_metric),
        }
        history.append(epoch_row)
        current_metric_value = float(epoch_row["early_stop_metric_value"])
        if early_stop_metric_improved(args.early_stop_metric, current_metric_value, best_metric_value):
            best_metric_value = current_metric_value
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if args.log_every > 0 and (epoch == 1 or epoch % args.log_every == 0):
            print(json.dumps({"event": "epoch", **epoch_row}, separators=(",", ":")), flush=True)
        if args.patience > 0 and stale_epochs >= args.patience:
            break

    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
        best_epoch = history[-1]["epoch"]
    net.load_state_dict(best_state)
    predictions = predict_all(torch, net, cache, stats, device, args.batch_action_rows)
    eval_rows = {
        "train": split_eval(cache, predictions, targets, "train", train_states),
        "val": split_eval(cache, predictions, targets, "val", val_states),
        "test": split_eval(cache, predictions, targets, "test", test_states),
    }

    breakdown_rows: list[dict[str, Any]] = []
    for split_name in ("val", "test"):
        for bucket in sorted({str(row["bucket_group"]) for row in cache["state_metadata"]}):
            breakdown_rows.append(
                subset_rows(
                    cache,
                    predictions,
                    targets,
                    split_name,
                    lambda row, bucket=bucket: str(row["bucket_group"]) == bucket,
                    f"{split_name}:bucket:{bucket}",
                )
            )
    position_rows: list[dict[str, Any]] = []
    for split_name in ("val", "test"):
        for seat in ("first", "second"):
            position_rows.append(
                subset_rows(
                    cache,
                    predictions,
                    targets,
                    split_name,
                    lambda row, seat=seat: str(row["seat"]) == seat,
                    f"{split_name}:position:{seat}",
                )
            )
    label_rows: list[dict[str, Any]] = []
    for split_name in ("val", "test"):
        for label in ("positive", "gray", "negative"):
            label_rows.append(
                subset_rows(
                    cache,
                    predictions,
                    targets,
                    split_name,
                    lambda row, label=label: str(row["pilot_gate_label"]) == label,
                    f"{split_name}:label:{label}",
                )
            )
    margin_rows: list[dict[str, Any]] = []
    for split_name in ("val", "test"):
        for bucket in ("lt_0_05", "0_05_0_10", "0_10_0_25", "0_25_0_50", "0_50_1_00", "ge_1_00"):
            margin_rows.append(
                subset_rows(
                    cache,
                    predictions,
                    targets,
                    split_name,
                    lambda row, bucket=bucket: str(row["margin_bucket"]) == bucket,
                    f"{split_name}:margin:{bucket}",
                )
            )

    threshold_split = "holdout" if args.threshold_split == "holdout" else args.threshold_split
    threshold_rows = threshold_sweep(cache, predictions, targets, threshold_split)

    calibration_rows: list[dict[str, Any]] = []
    for state_index in state_indices_for_split(cache["split"], threshold_split):
        start = int(cache["offsets"][state_index])
        end = int(cache["offsets"][state_index + 1])
        base = int(cache["baseline_action_index"][state_index])
        pred = predictions[start:end]
        y = targets[start:end]
        candidate = int(np.argmax(pred[:, 1]))
        calibration_rows.append(
            {
                "state_index": int(state_index),
                "split": threshold_split,
                "predicted_delta": float(pred[candidate, 1]),
                "teacher_delta": float(y[candidate, 0] - y[base, 0]),
                "teacher_best_delta": float(cache["state_metadata"][int(state_index)]["delta_best_vs_baseline"]),
                "bucket_group": cache["state_metadata"][int(state_index)]["bucket_group"],
                "seat": cache["state_metadata"][int(state_index)]["seat"],
                "pilot_gate_label": cache["state_metadata"][int(state_index)]["pilot_gate_label"],
            }
        )

    split_counts = {
        name: int(np.sum(cache["split"] == split_id))
        for split_id, name in SPLIT_ID_TO_NAME.items()
    }
    positive_thresholds = [
        row
        for row in threshold_rows
        if row["override_count"] > 0 and row["teacher_avg_gain_on_override"] > 0.0
    ]
    recommended = (
        "GO to a larger current-FL-EV broad pass only after the same pipeline is repeated with "
        "a larger holdout and then seat-swap validation. This pilot is enough to validate the "
        "cache/training pipeline, not production adoption."
        if eval_rows["test"]["top3_recall"] > 0.5 and eval_rows["test"]["avg_regret"] >= 0.0
        else "NO-GO to 50k until feature/label/model issues are inspected; pilot metrics are weak."
    )
    summary = {
        "schema": "hu_turn2_stage8_pilot_training_v1",
        "cache_dir": str(args.cache_dir.resolve()),
        "output_dir": str(output_dir),
        "model_output": str(args.model_output),
        "device": device,
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "hidden_layer_sizes": list(hidden_layers),
        "dropout": args.dropout,
        "init_model": init_model_metadata,
        "initial_val_eval": initial_eval,
        "normalization_stats_source": args.init_stats_source if init_payload is not None else "cache",
        "early_stop_metric": args.early_stop_metric,
        "best_metric_value": best_metric_value,
        "epochs_ran": len(history),
        "best_epoch": best_epoch,
        "history": history,
        "split_counts": split_counts,
        "eval": eval_rows,
        "threshold_positive_config_count": len(positive_thresholds),
        "elapsed_seconds": time.time() - started_at,
        "recommended_next_step": recommended,
        "gate_label_source": "stage8b_labels_csv" if args.stage8b_labels_csv is not None else "cache_gate_label_id",
        "gate_label_override": gate_label_override,
        "loss_weight_metadata": loss_weight_metadata,
        "candidate_generator_training": candidate_weight_metadata,
        "scoring_metadata_status": scoring_status,
        "t3_continuation_metadata": t3_metadata,
        "teacher_rollout_count_distribution": cache["metadata"].get("rollout_count_distribution", {}),
        "teacher_mc_label": teacher_mc_label_from_distribution(cache["metadata"].get("rollout_count_distribution", {})),
    }

    torch.save(
        {
            "model_kind": "hu_turn2_pilot_multihead_mlp",
            "feature_dim": int(cache["metadata"]["feature_dim"]),
            "hidden_layer_sizes": list(hidden_layers),
            "dropout": args.dropout,
            "state_dict": best_state,
            "feature_mean": stats["feature_mean"],
            "feature_scale": stats["feature_scale"],
            "target_mean": stats["target_mean"],
            "target_scale": stats["target_scale"],
            "heads": ["ev", "delta_vs_baseline", "delta_vs_reference", "rank_score", "override_gate_logit"],
            "override_gate_semantics": "safe_override_probability" if args.stage8b_labels_csv is not None else "pilot_positive_probability",
            "stage8b_labels_csv": str(args.stage8b_labels_csv.resolve()) if args.stage8b_labels_csv is not None else None,
            "init_from_model": str(args.init_from_model.resolve()) if args.init_from_model is not None else None,
            "init_model": init_model_metadata,
            "normalization_stats_source": args.init_stats_source if init_payload is not None else "cache",
            "early_stop_metric": args.early_stop_metric,
            "best_metric_value": best_metric_value,
            "candidate_generator_training_jsonl": [
                str(path.resolve()) for path in args.candidate_generator_training_jsonl
            ],
            "candidate_generator_training": candidate_weight_metadata,
            "scoring_metadata_status": scoring_status,
            "loss_weight_metadata": loss_weight_metadata,
            "gate_label_column": args.gate_label_column if args.stage8b_labels_csv is not None else None,
            "gate_weight_column": args.gate_weight_column if args.stage8b_labels_csv is not None else None,
            "t3_continuation": t3_metadata.get("t3_continuation"),
            "t3_continuation_policy": t3_metadata.get("continuation_policy_T3"),
            "t3_continuation_metadata": t3_metadata,
            "hu_turn3_min_margin": hu_turn3_min_margin,
            "hu_turn3_reference_min_margin": hu_turn3_reference_min_margin,
            "pilot_only": True,
        },
        args.model_output,
    )

    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "feature_cache_summary.json").write_text(
        json.dumps(cache["metadata"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "split_summary.json").write_text(
        json.dumps(cache["metadata"].get("split_summary", {}), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(output_dir / "training_summary.md", summary, threshold_rows)
    write_csv(output_dir / "training_metrics.csv", history)
    write_csv(output_dir / "holdout_eval.csv", [eval_rows["val"], eval_rows["test"]])
    write_csv(output_dir / "bucket_breakdown.csv", breakdown_rows)
    write_csv(output_dir / "position_breakdown.csv", position_rows)
    write_csv(output_dir / "label_breakdown.csv", label_rows)
    write_csv(output_dir / "margin_bucket_breakdown.csv", margin_rows)
    write_csv(output_dir / "threshold_sweep.csv", threshold_rows)
    write_csv(output_dir / "calibration_plot_data.csv", calibration_rows)
    (output_dir / "recommended_next_step.md").write_text(recommended + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
