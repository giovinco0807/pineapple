"""High-MC audit for HU Turn2 Stage8 selective override failures.

This is an analysis-only tool. It does not start a 50k teacher pass, T1
training, production deployment, or P2 fixation. It selects fired, near-fired,
missed-positive, and suspected-false-positive states from the Stage8 20k
teacher/evaluation artifacts, then can replay selected teacher states with
higher MC using the fixed Stage7_candidate_A m5_r10 continuation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .ai_profiles import (
    DEFAULT_HU_TURN3_STAGE7_MODEL,
    DEFAULT_HU_TURN3_STAGE7_REFERENCE_MODEL,
    DEFAULT_OPENING_MODEL,
    DEFAULT_TURN1_MODEL,
    DEFAULT_TURN2_MODEL,
    DEFAULT_TURN3_MODEL,
    ModelPaths,
    load_model_bundle,
)
from .final_turn_decision_cache import FinalTurnDecisionCache
from .hu_turn2_stage8_runtime import (
    HuTurn2Stage8RuntimeConfig,
    load_hu_turn2_stage8_model,
)
from .hu_turn2_teacher_data import (
    _build_policy_for_profile,
    evaluate_hu_turn2_actions,
)
from .hu_turn3_batch_continuation import (
    HuTurn3ActionCache,
    HuTurn3DecisionCache,
    HuTurn3Stage3ReferenceCache,
    HuTurn3Stage7BatchConfig,
    Stage3StateFeatureCache,
)
from .play_ai import _prediction_thread_context
from .state import Board


DEFAULT_TEACHER = Path("outputs/hu_turn2_stage8_20k_mc512/hu_turn2_stage8_20k_mc512.jsonl")
DEFAULT_CACHE_DIR = Path("D:/ofc-pineapple-storage/regular-ofc-pineapple/feature_cache/hu_t2_stage8_20k_mc512")
DEFAULT_CALIBRATION = Path("outputs/hu_turn2_stage8_20k_mc512_calibration/state_calibration_values.csv")
DEFAULT_DECISION_LOG = Path("outputs/evals/hu_turn2_stage8_20k_seat_swap/hu_turn2_stage8_20k_runtime_decisions.jsonl")
DEFAULT_OUTPUT_DIR = Path("outputs/evals/hu_turn2_stage8_20k_high_mc_audit")
DEFAULT_STAGE8_MODEL = Path("models/hu_turn2_stage8_broad_20k_mc512_reference_override_cached_rank_wide.pt")

DEFAULT_CONFIGS = (
    "2.75/0.05/0.90",
    "2.91/0.10/0.90",
    "3.10/0.10/0.90",
    "3.25/0.10/0.925",
)


@dataclass(frozen=True)
class SelectedState:
    state_index: int
    audit_group: str
    config_id: str
    priority: int
    row: dict[str, str]
    selection_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--calibration-values", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--runtime-decision-log", type=Path, default=DEFAULT_DECISION_LOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    parser.add_argument("--split", choices=("train", "val", "test", "holdout", "all"), default="test")
    parser.add_argument("--mc-samples", type=int, default=4096)
    parser.add_argument("--max-replay-states", type=int, default=0, help="0 writes selection artifacts only.")
    parser.add_argument("--selection-strategy", choices=("priority", "stratified"), default="priority")
    parser.add_argument("--include-fired", type=int, default=15)
    parser.add_argument("--include-suspected-false-positive", type=int, default=15)
    parser.add_argument("--include-missed-positive", type=int, default=15)
    parser.add_argument("--include-near-fired", type=int, default=5)
    parser.add_argument("--max-fired", type=int, default=200)
    parser.add_argument("--max-near-fired", type=int, default=1000)
    parser.add_argument("--max-missed-positive", type=int, default=1000)
    parser.add_argument("--max-suspected-false-positive", type=int, default=1000)
    parser.add_argument("--near-margin-window", type=float, default=0.50)
    parser.add_argument("--near-gate-low", type=float, default=0.75)
    parser.add_argument("--near-gate-high", type=float, default=0.95)
    parser.add_argument("--near-reference-low", type=float, default=0.0)
    parser.add_argument("--near-reference-high", type=float, default=0.30)
    parser.add_argument("--missed-positive-delta", type=float, default=0.50)
    parser.add_argument("--missed-positive-se-multiple", type=float, default=2.0)
    parser.add_argument("--seed-mode", choices=("reuse_original", "offset_by_mc"), default="reuse_original")
    parser.add_argument("--prediction-threads", type=int, default=1)
    parser.add_argument("--continuation-profile", default="current")
    parser.add_argument("--opponent-profile", default="current")
    parser.add_argument("--opening-lookahead-samples", type=int, default=128)
    parser.add_argument("--stage3-feature-encoder-mode", default="rust_direct")
    parser.add_argument("--batched-continuation-batch-size", type=int, default=8192)
    parser.add_argument("--continuation-cache-size", type=int, default=200_000)
    parser.add_argument("--final-turn-cache-size", type=int, default=200_000)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--opening-model", type=Path, default=DEFAULT_OPENING_MODEL)
    parser.add_argument("--turn1-model", type=Path, default=DEFAULT_TURN1_MODEL)
    parser.add_argument("--turn2-model", type=Path, default=DEFAULT_TURN2_MODEL)
    parser.add_argument("--turn3-model", type=Path, default=DEFAULT_TURN3_MODEL)
    parser.add_argument("--hu-turn2-stage8-model", type=Path, default=DEFAULT_STAGE8_MODEL)
    parser.add_argument("--hu-turn3-stage7-model", type=Path, default=DEFAULT_HU_TURN3_STAGE7_MODEL)
    parser.add_argument("--hu-turn3-stage7-reference-model", type=Path, default=DEFAULT_HU_TURN3_STAGE7_REFERENCE_MODEL)
    return parser.parse_args()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def parse_configs(value: str) -> list[HuTurn2Stage8RuntimeConfig]:
    configs: list[HuTurn2Stage8RuntimeConfig] = []
    for part in (item.strip() for item in value.split(",") if item.strip()):
        fields = part.replace("_", "/").split("/")
        if len(fields) != 3:
            raise ValueError(f"config must be m/r/g: {part}")
        configs.append(HuTurn2Stage8RuntimeConfig(float(fields[0]), float(fields[1]), float(fields[2])))
    if not configs:
        raise ValueError("at least one config is required")
    return configs


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def row_in_split(row: dict[str, str], split: str) -> bool:
    if split == "all":
        return True
    if split == "holdout":
        return row.get("split") != "train"
    return row.get("split") == split


def runtime_would_fire(row: dict[str, str], config: HuTurn2Stage8RuntimeConfig) -> bool:
    return (
        safe_int(row.get("candidate_is_baseline"), 1) == 0
        and safe_float(row.get("predicted_delta_vs_baseline")) >= config.min_margin
        and safe_float(row.get("reference_margin_raw")) >= config.reference_min_margin
        and safe_float(row.get("gate_probability")) >= config.gate_threshold
    )


def runtime_no_fire_reason(row: dict[str, str], config: HuTurn2Stage8RuntimeConfig) -> str:
    if safe_int(row.get("candidate_is_baseline"), 1):
        return "same_as_baseline"
    if safe_float(row.get("reference_margin_raw")) < config.reference_min_margin:
        return "below_reference_margin"
    if safe_float(row.get("predicted_delta_vs_baseline")) < config.min_margin:
        return "below_stage8_margin"
    if safe_float(row.get("gate_probability")) < config.gate_threshold:
        return "below_gate_threshold"
    return "would_fire"


def selection_payload(selected: SelectedState) -> dict[str, Any]:
    row = selected.row
    return {
        "state_index": selected.state_index,
        "audit_group": selected.audit_group,
        "config_id": selected.config_id,
        "priority": selected.priority,
        "selection_reason": selected.selection_reason,
        "split": row.get("split"),
        "seat": row.get("seat"),
        "bucket_group": row.get("bucket_group"),
        "run_bucket": row.get("run_bucket"),
        "pilot_gate_label": row.get("pilot_gate_label"),
        "candidate_is_baseline": safe_int(row.get("candidate_is_baseline")),
        "predicted_delta": safe_float(row.get("predicted_delta_vs_baseline")),
        "gate_probability": safe_float(row.get("gate_probability")),
        "reference_margin_raw": safe_float(row.get("reference_margin_raw")),
        "teacher_mc512_delta_candidate_vs_baseline": safe_float(row.get("actual_delta_candidate_vs_baseline")),
        "teacher_mc512_delta_best_vs_baseline": safe_float(row.get("actual_delta_best_vs_baseline")),
        "teacher_best_margin": safe_float(row.get("teacher_best_margin")),
        "SE_delta": safe_float(row.get("SE_delta")),
        "actual_high_regret": parse_bool(row.get("actual_high_regret")),
        "actual_low_margin": parse_bool(row.get("actual_low_margin")),
        "actual_teacher_disagreement": parse_bool(row.get("actual_teacher_disagreement")),
    }


def select_audit_states(
    rows: list[dict[str, str]],
    configs: list[HuTurn2Stage8RuntimeConfig],
    *,
    split: str,
    max_fired: int,
    max_near_fired: int,
    max_missed_positive: int,
    max_suspected_false_positive: int,
    near_margin_window: float,
    near_gate_low: float,
    near_gate_high: float,
    near_reference_low: float,
    near_reference_high: float,
    missed_positive_delta: float,
    missed_positive_se_multiple: float,
) -> list[SelectedState]:
    selected: list[SelectedState] = []
    filtered = [row for row in rows if row_in_split(row, split)]
    for config in configs:
        fired: list[SelectedState] = []
        near: list[SelectedState] = []
        missed: list[SelectedState] = []
        suspected_fp: list[SelectedState] = []
        for row in filtered:
            state_index = safe_int(row.get("state_index"), -1)
            if state_index < 0 or safe_int(row.get("candidate_is_baseline"), 1):
                continue
            predicted_delta = safe_float(row.get("predicted_delta_vs_baseline"))
            gate = safe_float(row.get("gate_probability"))
            reference = safe_float(row.get("reference_margin_raw"))
            actual_delta = safe_float(row.get("actual_delta_candidate_vs_baseline"))
            se = safe_float(row.get("SE_delta"))
            fires = runtime_would_fire(row, config)
            if fires:
                fired.append(SelectedState(state_index, "fired_teacher", config.config_id, 20, row, "teacher_runtime_condition_pass"))
                if actual_delta <= max(0.10, 2.0 * se) or actual_delta < 0.0 or parse_bool(row.get("actual_low_margin")):
                    suspected_fp.append(
                        SelectedState(
                            state_index,
                            "suspected_false_positive",
                            config.config_id,
                            0,
                            row,
                            "fired_with_negative_or_low_margin_mc512_delta",
                        )
                    )
            if (
                not fires
                and config.min_margin - near_margin_window <= predicted_delta <= config.min_margin + near_margin_window
                and near_gate_low <= gate <= near_gate_high
                and near_reference_low <= reference <= near_reference_high
            ):
                near.append(
                    SelectedState(
                        state_index,
                        "near_fired",
                        config.config_id,
                        30,
                        row,
                        runtime_no_fire_reason(row, config),
                    )
                )
            if (
                not fires
                and actual_delta >= missed_positive_delta
                and (se <= 0.0 or actual_delta >= missed_positive_se_multiple * se)
            ):
                missed.append(
                    SelectedState(
                        state_index,
                        "missed_positive",
                        config.config_id,
                        10,
                        row,
                        runtime_no_fire_reason(row, config),
                    )
                )
        fired.sort(key=lambda item: -safe_float(item.row.get("predicted_delta_vs_baseline")))
        near.sort(key=lambda item: abs(safe_float(item.row.get("predicted_delta_vs_baseline")) - config.min_margin))
        missed.sort(key=lambda item: -safe_float(item.row.get("actual_delta_candidate_vs_baseline")))
        suspected_fp.sort(key=lambda item: safe_float(item.row.get("actual_delta_candidate_vs_baseline")))
        selected.extend(suspected_fp[:max_suspected_false_positive])
        selected.extend(fired[:max_fired])
        selected.extend(missed[:max_missed_positive])
        selected.extend(near[:max_near_fired])
    return selected


def _merge_selection_payload(
    by_state: dict[int, dict[str, Any]],
    item: SelectedState,
    *,
    order: int,
    replay_origin_group: str | None = None,
) -> bool:
    payload = selection_payload(item)
    current = by_state.get(item.state_index)
    if current is None:
        payload["audit_groups"] = [item.audit_group]
        payload["config_ids"] = [item.config_id]
        payload["selection_reasons"] = [item.selection_reason]
        payload["selection_order"] = order
        if replay_origin_group is not None:
            payload["replay_origin_group"] = replay_origin_group
        by_state[item.state_index] = payload
        return True
    for key, value in (
        ("audit_groups", item.audit_group),
        ("config_ids", item.config_id),
        ("selection_reasons", item.selection_reason),
    ):
        if value not in current[key]:
            current[key].append(value)
    current["priority"] = min(int(current["priority"]), item.priority)
    current["selection_order"] = min(int(current["selection_order"]), order)
    if replay_origin_group is not None and "replay_origin_group" not in current:
        current["replay_origin_group"] = replay_origin_group
    return False


def dedupe_for_replay(selected: list[SelectedState]) -> list[dict[str, Any]]:
    by_state: dict[int, dict[str, Any]] = {}
    for order, item in enumerate(selected):
        _merge_selection_payload(by_state, item, order=order)
    return sorted(by_state.values(), key=lambda row: (int(row["priority"]), int(row["selection_order"])))


def _stratified_sort_key(group: str, item: SelectedState) -> tuple[float, ...]:
    row = item.row
    if group == "suspected_false_positive":
        return (
            safe_float(row.get("actual_delta_candidate_vs_baseline")),
            -safe_float(row.get("predicted_delta_vs_baseline")),
            safe_float(row.get("teacher_best_margin")),
        )
    if group == "missed_positive":
        return (
            -safe_float(row.get("actual_delta_candidate_vs_baseline")),
            -safe_float(row.get("predicted_delta_vs_baseline")),
            safe_float(row.get("teacher_best_margin")),
        )
    if group == "near_fired":
        return (
            -safe_float(row.get("gate_probability")),
            -safe_float(row.get("predicted_delta_vs_baseline")),
            safe_float(row.get("reference_margin_raw")),
        )
    return (
        -safe_float(row.get("predicted_delta_vs_baseline")),
        safe_float(row.get("actual_delta_candidate_vs_baseline")),
        safe_float(row.get("teacher_best_margin")),
    )


def stratified_for_replay(
    selected: list[SelectedState],
    *,
    include_fired: int,
    include_suspected_false_positive: int,
    include_missed_positive: int,
    include_near_fired: int,
) -> list[dict[str, Any]]:
    targets = (
        ("fired_teacher", include_fired),
        ("suspected_false_positive", include_suspected_false_positive),
        ("missed_positive", include_missed_positive),
        ("near_fired", include_near_fired),
    )
    by_state: dict[int, dict[str, Any]] = {}
    replay_order = 0
    for group, limit in targets:
        if limit <= 0:
            continue
        candidates = [item for item in selected if item.audit_group == group]
        candidates.sort(key=lambda item, group=group: _stratified_sort_key(group, item))
        added = 0
        for item in candidates:
            if item.state_index in by_state:
                _merge_selection_payload(by_state, item, order=replay_order, replay_origin_group=group)
                continue
            if _merge_selection_payload(by_state, item, order=replay_order, replay_origin_group=group):
                replay_order += 1
                added += 1
            if added >= limit:
                break
    # Backfill from priority order if overlap or sparse buckets leave the replay
    # prefix below the requested total.
    requested_total = sum(max(0, limit) for _group, limit in targets)
    if len(by_state) < requested_total:
        for item in sorted(selected, key=lambda state: (state.priority, state.state_index, state.config_id)):
            if item.state_index in by_state:
                _merge_selection_payload(by_state, item, order=replay_order)
                continue
            if _merge_selection_payload(by_state, item, order=replay_order, replay_origin_group="backfill"):
                replay_order += 1
            if len(by_state) >= requested_total:
                break
    return sorted(by_state.values(), key=lambda row: int(row["selection_order"]))


def load_cache_metadata(cache_dir: Path) -> dict[str, Any]:
    path = cache_dir / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"feature cache metadata not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_input_path(path_text: str, repo_root: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else repo_root / path


def load_teacher_rows_for_state_indices(
    *,
    metadata: dict[str, Any],
    repo_root: Path,
    state_indices: set[int],
) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    state_index = 0
    for entry in metadata.get("input_files") or [{"path": metadata.get("input", "")}]:
        input_path = resolve_input_path(str(entry.get("path", "")), repo_root)
        if not input_path.exists():
            raise FileNotFoundError(f"teacher input file not found: {input_path}")
        with input_path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if state_index in state_indices and line.strip():
                    rows[state_index] = json.loads(line)
                state_index += 1
        if state_indices.issubset(rows.keys()):
            break
    return rows


def board_from_json(payload: Any) -> Board:
    if not isinstance(payload, dict):
        raise ValueError("board payload must be an object")
    return Board.from_rows(payload.get("top") or (), payload.get("middle") or (), payload.get("bottom") or ())


def action_original_index(action: Any) -> int | None:
    if isinstance(action, int):
        return int(action)
    if isinstance(action, dict) and action.get("original_index") is not None:
        return safe_int(action.get("original_index"))
    if isinstance(action, dict) and isinstance(action.get("action"), dict):
        return action_original_index(action["action"])
    return None


def action_text(action: dict[str, Any] | None) -> str:
    if not action:
        return ""
    body = action.get("action") if isinstance(action.get("action"), dict) else action
    placements = ",".join(f"{card}->{row}" for card, row in body.get("placements", ()))
    discards = ",".join(str(card) for card in body.get("discards", ()))
    return f"{placements}; discard={discards}"


def action_by_original(actions: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    for action in actions:
        original = action_original_index(action)
        if original is not None:
            output[original] = action
    return output


def baseline_local_index(sample: dict[str, Any]) -> int:
    actions = list(sample.get("actions") or ())
    baseline_original = action_original_index(sample.get("baseline_action"))
    for index, action in enumerate(actions):
        if bool(action.get("is_baseline_action")):
            return index
        if baseline_original is not None and action_original_index(action) == baseline_original:
            return index
    return 0


def candidate_context(sample: dict[str, Any], stage8_model: object) -> dict[str, Any]:
    actions = list(sample.get("actions") or ())
    predictions = np.asarray(stage8_model.predict_sample(sample), dtype=np.float64)
    if predictions.ndim != 2 or predictions.shape[0] != len(actions) or predictions.shape[1] < 5:
        raise ValueError("invalid Stage8 prediction shape")
    candidate = int(np.argmax(predictions[:, 1]))
    model_top1 = int(np.argmax(predictions[:, 0]))
    baseline = baseline_local_index(sample)
    gate_logit = float(np.mean(predictions[:, 4]))
    gate_probability = 1.0 / (1.0 + math.exp(-gate_logit)) if gate_logit >= 0 else math.exp(gate_logit) / (1.0 + math.exp(gate_logit))
    return {
        "candidate_local_index": candidate,
        "baseline_local_index": baseline,
        "model_top1_local_index": model_top1,
        "candidate_original_index": action_original_index(actions[candidate]),
        "baseline_original_index": action_original_index(actions[baseline]),
        "model_top1_original_index": action_original_index(actions[model_top1]),
        "predicted_delta": float(predictions[candidate, 1]),
        "predicted_ev": float(predictions[candidate, 0]),
        "predicted_rank_score": float(predictions[candidate, 3]),
        "gate_probability": gate_probability,
        "predictions_by_original": {
            int(action_original_index(action)): {
                "predicted_ev": float(predictions[index, 0]),
                "predicted_delta": float(predictions[index, 1]),
                "predicted_delta_reference": float(predictions[index, 2]),
                "predicted_rank_score": float(predictions[index, 3]),
            }
            for index, action in enumerate(actions)
            if action_original_index(action) is not None
        },
    }


def replay_seed(sample: dict[str, Any], mc_samples: int, seed_mode: str) -> int:
    seed = safe_int(sample.get("future_rollout_seed"), 0)
    return seed + mc_samples if seed_mode == "offset_by_mc" else seed


def build_batched_config(args: argparse.Namespace) -> HuTurn3Stage7BatchConfig:
    run_hash = hashlib.sha256(
        json.dumps(
            {
                "tool": "hu_turn2_stage8_high_mc_audit",
                "mc_samples": args.mc_samples,
                "stage3_feature_encoder_mode": args.stage3_feature_encoder_mode,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    return HuTurn3Stage7BatchConfig(
        stage7_enabled=True,
        hu_turn3_min_margin=5.0,
        hu_turn3_reference_min_margin=10.0,
        batch_size=args.batched_continuation_batch_size,
        use_cache=True,
        use_stage3_feature_fast_path=True,
        stage3_feature_encoder_mode=args.stage3_feature_encoder_mode,
        stage3_feature_replay_model_path=str(args.hu_turn3_stage7_reference_model),
        stage3_feature_replay_stage7_model_path=str(args.hu_turn3_stage7_model),
        stage3_feature_replay_source=f"hu_turn2_stage8_high_mc_audit_mc{args.mc_samples}",
        stage3_feature_replay_teacher_run_hash=run_hash,
    )


def load_model_bundle_for_replay(args: argparse.Namespace) -> object:
    return load_model_bundle(
        ModelPaths(
            opening=args.opening_model,
            turn1=args.turn1_model,
            turn2=args.turn2_model,
            turn3=args.turn3_model,
            hu_turn3_stage7=args.hu_turn3_stage7_model,
            hu_turn3_stage7_reference=args.hu_turn3_stage7_reference_model,
        ),
        {"current"},
    )


def replay_one_state(
    selection: dict[str, Any],
    sample: dict[str, Any],
    *,
    args: argparse.Namespace,
    bundle: object,
    stage8_model: object,
    batched_config: HuTurn3Stage7BatchConfig,
    batched_cache: HuTurn3DecisionCache,
    batched_reference_cache: HuTurn3Stage3ReferenceCache,
    batched_state_feature_cache: Stage3StateFeatureCache,
    batched_action_cache: HuTurn3ActionCache,
    final_turn_cache: FinalTurnDecisionCache,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    context = candidate_context(sample, stage8_model)
    board = board_from_json(sample.get("board"))
    opponent = board_from_json(sample.get("opponent_board"))
    dealt = tuple(sample.get("dealt") or sample.get("cards_to_place") or ())
    dead = tuple(sample.get("dead_cards") or ())
    if len(dealt) != 3:
        raise ValueError("T2 replay requires exactly 3 dealt cards")
    hero_seat = str(sample.get("seat") or "first")
    opponent_seat = "second" if hero_seat == "first" else "first"
    seed = replay_seed(sample, args.mc_samples, args.seed_mode)
    hero_policy = _build_policy_for_profile(
        args.continuation_profile,
        bundle,
        seed=seed * 4 + 2,
        seat=hero_seat,
        opening_lookahead_samples=args.opening_lookahead_samples,
    )
    opponent_policy = _build_policy_for_profile(
        args.opponent_profile,
        bundle,
        seed=seed * 4 + 3,
        seat=opponent_seat,
        opening_lookahead_samples=args.opening_lookahead_samples,
    )
    started_at = time.perf_counter()
    high = evaluate_hu_turn2_actions(
        board=board,
        dealt_cards=dealt,
        opponent_board=opponent,
        dead_cards=dead,
        hero_seat=hero_seat,
        continuation_policy=hero_policy,
        opponent_policy=opponent_policy,
        baseline_turn2_model=bundle.turn2,
        future_samples=args.mc_samples,
        future_rollout_seed=seed,
        use_batched_continuation=True,
        batched_continuation_config=batched_config,
        batched_continuation_cache=batched_cache,
        batched_reference_cache=batched_reference_cache,
        batched_state_feature_cache=batched_state_feature_cache,
        batched_action_cache=batched_action_cache,
        batched_continuation_batch_size=args.batched_continuation_batch_size,
        final_turn_cache=final_turn_cache,
        use_final_turn_cache=True,
    )
    elapsed = time.perf_counter() - started_at
    if high is None:
        raise ValueError("evaluate_hu_turn2_actions returned no sample")

    initial_by_original = action_by_original(list(sample.get("actions") or ()))
    high_actions = list(high.get("actions") or ())
    high_by_original = action_by_original(high_actions)
    candidate_original = int(context["candidate_original_index"])
    baseline_original = int(context["baseline_original_index"])
    model_top1_original = int(context["model_top1_original_index"])
    high_best = high_actions[0]
    high_second = high_actions[1] if len(high_actions) > 1 else high_best
    high_best_original = int(action_original_index(high_best))
    teacher_mc512_best_original = int(action_original_index((sample.get("actions") or [{}])[0]))
    candidate_high = high_by_original.get(candidate_original)
    baseline_high = high_by_original.get(baseline_original)
    model_top1_high = high_by_original.get(model_top1_original)
    if candidate_high is None or baseline_high is None:
        raise ValueError("candidate or baseline action missing after high-MC replay")
    candidate_ev = safe_float(candidate_high.get("score", candidate_high.get("ev")))
    baseline_ev = safe_float(baseline_high.get("score", baseline_high.get("ev")))
    high_delta = candidate_ev - baseline_ev
    mc512_candidate = initial_by_original.get(candidate_original, {})
    mc512_baseline = initial_by_original.get(baseline_original, {})
    mc512_delta = safe_float(mc512_candidate.get("score", mc512_candidate.get("ev"))) - safe_float(
        mc512_baseline.get("score", mc512_baseline.get("ev"))
    )
    candidate_se = safe_float(candidate_high.get("ev_standard_error", candidate_high.get("standard_error")))
    baseline_se = safe_float(baseline_high.get("ev_standard_error", baseline_high.get("standard_error")))
    delta_se = math.sqrt(candidate_se * candidate_se + baseline_se * baseline_se)
    row = {
        **selection,
        "replay_status": "success",
        "failure_reason": "",
        "mc_samples": args.mc_samples,
        "future_rollout_seed": seed,
        "elapsed_seconds": elapsed,
        "candidate_original_index": candidate_original,
        "baseline_original_index": baseline_original,
        "model_top1_original_index": model_top1_original,
        "teacher_mc512_best_original_index": teacher_mc512_best_original,
        "high_mc_best_original_index": high_best_original,
        "high_mc_second_original_index": action_original_index(high_second),
        "baseline_action": action_text(baseline_high),
        "stage8_candidate_action": action_text(candidate_high),
        "model_top1_action": action_text(model_top1_high),
        "teacher_mc512_best_action": action_text(initial_by_original.get(teacher_mc512_best_original)),
        "high_mc_best_action": action_text(high_best),
        "baseline_high_mc_ev": baseline_ev,
        "candidate_high_mc_ev": candidate_ev,
        "high_mc_best_ev": safe_float(high_best.get("score", high_best.get("ev"))),
        "high_mc_second_ev": safe_float(high_second.get("score", high_second.get("ev"))),
        "high_mc_margin": safe_float(high_best.get("score", high_best.get("ev"))) - safe_float(high_second.get("score", high_second.get("ev"))),
        "high_mc_delta_candidate_vs_baseline": high_delta,
        "high_mc_delta_best_vs_baseline": safe_float(high_best.get("score", high_best.get("ev"))) - baseline_ev,
        "high_mc_delta_best_vs_candidate": safe_float(high_best.get("score", high_best.get("ev"))) - candidate_ev,
        "high_mc_delta_se": delta_se,
        "high_mc_lower95_candidate_vs_baseline": high_delta - 1.96 * delta_se,
        "mc512_delta_candidate_vs_baseline": mc512_delta,
        "sign_mc512_delta": sign(mc512_delta),
        "sign_high_mc_delta": sign(high_delta),
        "sign_stable": sign(mc512_delta) == sign(high_delta),
        "candidate_rank_high_mc": rank_of_original(high_actions, candidate_original),
        "baseline_rank_high_mc": rank_of_original(high_actions, baseline_original),
        "model_top1_rank_high_mc": rank_of_original(high_actions, model_top1_original),
        "teacher_mc512_best_rank_high_mc": rank_of_original(high_actions, teacher_mc512_best_original),
        "model_top1_equals_high_mc_best": model_top1_original == high_best_original,
        "candidate_equals_high_mc_best": candidate_original == high_best_original,
        "teacher_mc512_best_equals_high_mc_best": teacher_mc512_best_original == high_best_original,
        "common_random_future_digest": high.get("common_random_future_digest"),
        "stage7_t3_override_rate_candidate": candidate_high.get("stage7_t3_override_rate"),
        "stage7_t3_eval_count_candidate": candidate_high.get("stage7_t3_eval_count"),
        "stage7_t3_fired_count_candidate": candidate_high.get("stage7_t3_fired_count"),
        "profile_seconds_total": safe_float(high.get("profiling", {}).get("seconds_total")),
        "profile_ms_per_raw_t3": safe_float(high.get("profiling", {}).get("ms_per_raw_t3_decision")),
    }
    action_rows: list[dict[str, Any]] = []
    predictions_by_original = context["predictions_by_original"]
    for rank, action in enumerate(high_actions, start=1):
        original = int(action_original_index(action))
        initial = initial_by_original.get(original, {})
        pred = predictions_by_original.get(original, {})
        action_rows.append(
            {
                "state_index": selection["state_index"],
                "audit_groups": "|".join(selection.get("audit_groups", ())),
                "rank_high_mc": rank,
                "original_index": original,
                "action": action_text(action),
                "high_mc_ev": safe_float(action.get("score", action.get("ev"))),
                "high_mc_se": safe_float(action.get("ev_standard_error", action.get("standard_error"))),
                "mc512_ev": safe_float(initial.get("score", initial.get("ev"))),
                "mc512_se": safe_float(initial.get("ev_standard_error", initial.get("standard_error"))),
                "predicted_ev": pred.get("predicted_ev", ""),
                "predicted_delta": pred.get("predicted_delta", ""),
                "predicted_delta_reference": pred.get("predicted_delta_reference", ""),
                "predicted_rank_score": pred.get("predicted_rank_score", ""),
                "is_baseline": original == baseline_original,
                "is_stage8_candidate": original == candidate_original,
                "is_model_top1": original == model_top1_original,
                "is_teacher_mc512_best": original == teacher_mc512_best_original,
                "is_high_mc_best": original == high_best_original,
            }
        )
    return row, action_rows


def rank_of_original(actions: list[dict[str, Any]], original_index: int) -> int | None:
    for rank, action in enumerate(actions, start=1):
        if action_original_index(action) == original_index:
            return rank
    return None


def runtime_fired_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in iter_jsonl(path) or ():
        if row.get("override_fired"):
            rows.append(
                {
                    "source_kind": "runtime_decision_log",
                    "replay_ready": bool(row.get("dead_cards")),
                    "replay_blocker": "" if row.get("dead_cards") else "missing_dead_cards_in_legacy_runtime_log",
                    "config_id": row.get("config_id"),
                    "seed": row.get("seed"),
                    "hand_id": row.get("hand_id"),
                    "seat": row.get("seat"),
                    "seat_swap": row.get("seat_swap"),
                    "predicted_delta": row.get("predicted_delta"),
                    "gate_probability": row.get("gate_probability"),
                    "reference_margin_raw": row.get("reference_margin_raw"),
                    "candidate_seat_score": row.get("candidate_seat_score"),
                    "hero_board": row.get("hero_board"),
                    "opponent_board": row.get("opponent_board"),
                    "cards_to_place": row.get("cards_to_place"),
                    "baseline_action": row.get("baseline_action"),
                    "stage8_action": row.get("stage8_action"),
                    "final_action": row.get("final_action"),
                }
            )
    return rows


def correlation(x: list[float], y: list[float], *, method: str) -> float:
    if len(x) < 2 or len(y) < 2:
        return 0.0
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if method == "spearman":
        xa = np.argsort(np.argsort(xa)).astype(np.float64)
        ya = np.argsort(np.argsort(ya)).astype(np.float64)
    if np.std(xa) == 0.0 or np.std(ya) == 0.0:
        return 0.0
    return float(np.corrcoef(xa, ya)[0, 1])


def calibration_vs_high_mc_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = [
        ("predicted_delta", "high_mc_delta_candidate_vs_baseline"),
        ("gate_probability", "high_mc_delta_candidate_vs_baseline"),
        ("reference_margin_raw", "high_mc_delta_candidate_vs_baseline"),
        ("mc512_delta_candidate_vs_baseline", "high_mc_delta_candidate_vs_baseline"),
        ("teacher_best_margin", "high_mc_margin"),
    ]
    rows = []
    for x_key, y_key in pairs:
        xs = [safe_float(row.get(x_key), float("nan")) for row in results]
        ys = [safe_float(row.get(y_key), float("nan")) for row in results]
        filtered = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
        x = [item[0] for item in filtered]
        y = [item[1] for item in filtered]
        rows.append(
            {
                "relation": f"{x_key}__{y_key}",
                "n": len(filtered),
                "pearson": correlation(x, y, method="pearson"),
                "spearman": correlation(x, y, method="spearman"),
                "sign_agreement": (
                    sum(1 for a, b in filtered if sign(a) == sign(b)) / max(len(filtered), 1)
                ),
            }
        )
    return rows


def threshold_sweep_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    margins = (1.50, 2.00, 2.25, 2.50, 2.75, 2.91, 3.10, 3.25, 3.50)
    references = (None, 0.00, 0.05, 0.10, 0.20)
    gates = (0.80, 0.85, 0.90, 0.925, 0.95)
    rows: list[dict[str, Any]] = []
    for margin in margins:
        for reference in references:
            for gate in gates:
                fired = []
                missed = []
                for row in results:
                    candidate_is_baseline = int(row.get("candidate_original_index") == row.get("baseline_original_index"))
                    ref_ok = True if reference is None else safe_float(row.get("reference_margin_raw")) >= reference
                    ok = (
                        not candidate_is_baseline
                        and safe_float(row.get("predicted_delta")) >= margin
                        and ref_ok
                        and safe_float(row.get("gate_probability")) >= gate
                    )
                    high_delta = safe_float(row.get("high_mc_delta_candidate_vs_baseline"))
                    if ok:
                        fired.append(high_delta)
                    elif high_delta >= 0.50:
                        missed.append(high_delta)
                losses = [max(0.0, -value) for value in fired]
                rows.append(
                    {
                        "hu_turn2_min_margin": margin,
                        "hu_turn2_reference_min_margin": "none" if reference is None else reference,
                        "hu_turn2_gate_threshold": gate,
                        "audit_states": len(results),
                        "override_count": len(fired),
                        "override_rate_on_audit_set": len(fired) / max(len(results), 1),
                        "high_mc_avg_gain_on_override": float(np.mean(fired)) if fired else 0.0,
                        "false_positive_count": sum(1 for value in fired if value < 0.0),
                        "false_positive_rate": sum(1 for value in fired if value < 0.0) / max(len(fired), 1),
                        "p95_loss": float(np.quantile(losses, 0.95)) if losses else 0.0,
                        "p99_loss": float(np.quantile(losses, 0.99)) if losses else 0.0,
                        "max_loss": max(losses) if losses else 0.0,
                        "missed_positive_count": len(missed),
                    }
                )
    rows.sort(
        key=lambda row: (
            -safe_float(row["high_mc_avg_gain_on_override"]),
            safe_float(row["false_positive_rate"]),
            safe_float(row["p95_loss"]),
        )
    )
    return rows


def diagnosis_for(row: dict[str, Any]) -> str:
    high_delta = safe_float(row.get("high_mc_delta_candidate_vs_baseline"))
    mc512_delta = safe_float(row.get("mc512_delta_candidate_vs_baseline"))
    groups = set(row.get("audit_groups") or ())
    if high_delta >= 0.50 and "missed_positive" in groups:
        return "underfire"
    if sign(mc512_delta) > 0 and sign(high_delta) <= 0:
        return "low_margin_noise"
    if "fired_teacher" in groups and high_delta < 0.0:
        return "false_positive_gate"
    if row.get("candidate_equals_high_mc_best") is False and safe_float(row.get("high_mc_delta_best_vs_candidate")) > 0.25:
        return "model_ranking_error"
    if safe_float(row.get("reference_margin_raw")) < 0.10 and high_delta >= 0.50:
        return "reference_margin_bad_gate"
    return "needs_more_samples"


def group_breakdown(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row.get(key, "unknown")
        if isinstance(value, list):
            values = value
        else:
            values = [value]
        for item in values:
            grouped[str(item)].append(row)
    output = []
    for label, group in sorted(grouped.items()):
        gains = [safe_float(row.get("high_mc_delta_candidate_vs_baseline")) for row in group]
        output.append(
            {
                key: label,
                "records": len(group),
                "avg_high_mc_delta": float(np.mean(gains)) if gains else 0.0,
                "positive_count": sum(1 for value in gains if value > 0.0),
                "negative_count": sum(1 for value in gains if value < 0.0),
                "sign_stable_rate": sum(1 for row in group if row.get("sign_stable")) / max(len(group), 1),
                "candidate_best_rate": sum(1 for row in group if row.get("candidate_equals_high_mc_best")) / max(len(group), 1),
            }
        )
    return output


def _finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = [safe_float(row.get(key), float("nan")) for row in rows]
    return [value for value in values if math.isfinite(value)]


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def audit_metrics_summary(results: list[dict[str, Any]], threshold_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = _finite_values(results, "candidate_rank_high_mc")
    baseline_ranks = _finite_values(results, "baseline_rank_high_mc")
    mc512_delta = _finite_values(results, "mc512_delta_candidate_vs_baseline")
    high_delta = _finite_values(results, "high_mc_delta_candidate_vs_baseline")
    paired = [
        (
            safe_float(row.get("mc512_delta_candidate_vs_baseline"), float("nan")),
            safe_float(row.get("high_mc_delta_candidate_vs_baseline"), float("nan")),
        )
        for row in results
    ]
    paired = [(a, b) for a, b in paired if math.isfinite(a) and math.isfinite(b)]
    best_threshold = max(
        threshold_rows,
        key=lambda row: (
            safe_float(row.get("high_mc_avg_gain_on_override")),
            -safe_float(row.get("false_positive_rate")),
            -safe_float(row.get("p95_loss")),
        ),
        default={},
    )
    return {
        "total_audited_states": len(results),
        "by_selection_type": dict(Counter(str(row.get("replay_origin_group", "unknown")) for row in results)),
        "diagnosis_counts": dict(Counter(str(row.get("diagnosis", "unknown")) for row in results)),
        "mc512_vs_mc4096": {
            "sign_agreement": _rate(sum(1 for a, b in paired if sign(a) == sign(b)), len(paired)),
            "delta_pearson": correlation([a for a, _b in paired], [b for _a, b in paired], method="pearson"),
            "delta_spearman": correlation([a for a, _b in paired], [b for _a, b in paired], method="spearman"),
            "teacher_best_high_mc_best_rate": _rate(
                sum(1 for row in results if row.get("teacher_mc512_best_equals_high_mc_best")),
                len(results),
            ),
            "model_top1_high_mc_best_rate": _rate(
                sum(1 for row in results if row.get("model_top1_equals_high_mc_best")),
                len(results),
            ),
        },
        "stage8_candidate_rank": {
            "mean": float(np.mean(ranks)) if ranks else 0.0,
            "p50": float(np.quantile(ranks, 0.50)) if ranks else 0.0,
            "p90": float(np.quantile(ranks, 0.90)) if ranks else 0.0,
            "rank_le_3_rate": _rate(sum(1 for value in ranks if value <= 3), len(ranks)),
            "rank_gt_5_rate": _rate(sum(1 for value in ranks if value > 5), len(ranks)),
        },
        "baseline_rank": {
            "mean": float(np.mean(baseline_ranks)) if baseline_ranks else 0.0,
            "p50": float(np.quantile(baseline_ranks, 0.50)) if baseline_ranks else 0.0,
            "p90": float(np.quantile(baseline_ranks, 0.90)) if baseline_ranks else 0.0,
            "best_rate": _rate(sum(1 for value in baseline_ranks if value == 1), len(baseline_ranks)),
        },
        "stage8_candidate_delta": {
            "mean": float(np.mean(high_delta)) if high_delta else 0.0,
            "p50": float(np.quantile(high_delta, 0.50)) if high_delta else 0.0,
            "p10": float(np.quantile(high_delta, 0.10)) if high_delta else 0.0,
            "negative_rate": _rate(sum(1 for value in high_delta if value < 0.0), len(high_delta)),
            "positive_ge_0_5_rate": _rate(sum(1 for value in high_delta if value >= 0.5), len(high_delta)),
        },
        "best_high_mc_threshold_config": best_threshold,
    }


def write_audit_metrics_summary(output_dir: Path, results: list[dict[str, Any]], threshold_rows: list[dict[str, Any]]) -> None:
    payload = audit_metrics_summary(results, threshold_rows)
    (output_dir / "audit_metrics_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rank = payload["stage8_candidate_rank"]
    baseline = payload["baseline_rank"]
    delta = payload["stage8_candidate_delta"]
    agreement = payload["mc512_vs_mc4096"]
    lines = [
        "# HU T2 Stage8 High-MC Audit Metrics",
        "",
        f"- total audited states: `{payload['total_audited_states']}`",
        f"- by selection type: `{payload['by_selection_type']}`",
        f"- diagnosis counts: `{payload['diagnosis_counts']}`",
        "",
        "## MC512 vs MC4096",
        "",
        f"- sign agreement: `{agreement['sign_agreement']:.4f}`",
        f"- delta Pearson: `{agreement['delta_pearson']:.4f}`",
        f"- delta Spearman: `{agreement['delta_spearman']:.4f}`",
        f"- teacher best == high-MC best: `{agreement['teacher_best_high_mc_best_rate']:.4f}`",
        f"- model top1 == high-MC best: `{agreement['model_top1_high_mc_best_rate']:.4f}`",
        "",
        "## Stage8 Candidate Rank",
        "",
        f"- average rank: `{rank['mean']:.4f}`",
        f"- p50 / p90 rank: `{rank['p50']:.2f}` / `{rank['p90']:.2f}`",
        f"- rank <= 3 rate: `{rank['rank_le_3_rate']:.4f}`",
        f"- rank > 5 rate: `{rank['rank_gt_5_rate']:.4f}`",
        "",
        "## Baseline Rank",
        "",
        f"- average rank: `{baseline['mean']:.4f}`",
        f"- p50 / p90 rank: `{baseline['p50']:.2f}` / `{baseline['p90']:.2f}`",
        f"- baseline best rate: `{baseline['best_rate']:.4f}`",
        "",
        "## Candidate Delta",
        "",
        f"- mean / p50 / p10: `{delta['mean']:.4f}` / `{delta['p50']:.4f}` / `{delta['p10']:.4f}`",
        f"- negative rate: `{delta['negative_rate']:.4f}`",
        f"- positive >= 0.5 rate: `{delta['positive_ge_0_5_rate']:.4f}`",
        "",
    ]
    (output_dir / "audit_metrics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_selection_summary(
    path: Path,
    selected: list[SelectedState],
    deduped: list[dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
    *,
    selection_strategy: str,
) -> None:
    counts = Counter(item.audit_group for item in selected)
    config_counts = Counter(item.config_id for item in selected)
    replay_origin_counts = Counter(str(row.get("replay_origin_group", "priority")) for row in deduped)
    runtime_ready = sum(1 for row in runtime_rows if row.get("replay_ready"))
    lines = [
        "# HU T2 Stage8 High-MC Audit State Selection",
        "",
        "- scope: `analysis only`",
        "- production: `No-Go`",
        "- 50k teacher: `defer`",
        "- T1: `defer`",
        f"- selection_strategy: `{selection_strategy}`",
        f"- selected rows including config duplicates: `{len(selected)}`",
        f"- unique teacher states for replay: `{len(deduped)}`",
        f"- runtime fired states from legacy log: `{len(runtime_rows)}`",
        f"- runtime fired replay-ready states: `{runtime_ready}`",
        "",
        "## Audit Groups",
        "",
    ]
    for group, count in counts.most_common():
        lines.append(f"- `{group}`: `{count}`")
    lines.extend(["", "## Replay Origin Groups", ""])
    for group, count in replay_origin_counts.most_common():
        lines.append(f"- `{group}`: `{count}`")
    lines.extend(["", "## Configs", ""])
    for config, count in config_counts.most_common():
        lines.append(f"- `{config}`: `{count}`")
    if runtime_rows and runtime_ready < len(runtime_rows):
        lines.extend(
            [
                "",
                "## Runtime Replay Blocker",
                "",
                "Legacy seat-swap runtime logs do not include historical `dead_cards`, so those fired states are selection evidence but not exact high-MC replay inputs. New runtime logs now include `dead_cards`.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_recommended_next_step(path: Path, results: list[dict[str, Any]], failures: list[dict[str, Any]], max_replay_states: int) -> None:
    diagnoses = Counter(row.get("diagnosis", "unknown") for row in results)
    false_positive = diagnoses.get("false_positive_gate", 0)
    underfire = diagnoses.get("underfire", 0)
    enough = len(results) >= 50
    lines = [
        "# Recommended Next Step",
        "",
        "- T2 Stage8 production: `No-Go`",
        "- P2 fixation: `No-Go`",
        "- 50k teacher: `defer`",
        "- T1: `defer`",
        f"- high-MC successes: `{len(results)}`",
        f"- high-MC failures: `{len(failures)}`",
        f"- max_replay_states setting: `{max_replay_states}`",
        "",
        "## Diagnosis Counts",
        "",
    ]
    for label, count in diagnoses.most_common():
        lines.append(f"- `{label}`: `{count}`")
    lines.extend(["", "## Decision", ""])
    if not results:
        lines.append("No high-MC replay rows were run. Run selected MC4096/8192 replay before changing the model or thresholds.")
    elif not enough:
        lines.append("This is a smoke/partial audit only. Continue selected MC4096/8192 replay before 50k or T1.")
    elif underfire >= false_positive and underfire >= 5:
        lines.append("Primary blocker looks like underfire. Re-sweep thresholds on high-MC labels, then run a larger seat-swap validation.")
    elif false_positive:
        lines.append("Primary blocker is false-positive or label-noise risk. Do selected refinement and recalibrate the gate before any broad 50k pass.")
    else:
        lines.append("No single blocker is dominant yet. Expand high-MC audit before changing production runtime.")
    lines.extend(
        [
            "",
            "## Next Command",
            "",
            "```powershell",
            "python -m ofc_regular.audit_hu_turn2_stage8_high_mc `",
            "  --mc-samples 4096 `",
            "  --max-replay-states 50 `",
            "  --selection-strategy stratified `",
            "  --include-fired 15 `",
            "  --include-suspected-false-positive 15 `",
            "  --include-missed-positive 15 `",
            "  --include-near-fired 5 `",
            "  --prediction-threads 1",
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    started_at = time.perf_counter()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = parse_configs(args.configs)
    calibration_rows = read_csv_rows(args.calibration_values)
    selected = select_audit_states(
        calibration_rows,
        configs,
        split=args.split,
        max_fired=args.max_fired,
        max_near_fired=args.max_near_fired,
        max_missed_positive=args.max_missed_positive,
        max_suspected_false_positive=args.max_suspected_false_positive,
        near_margin_window=args.near_margin_window,
        near_gate_low=args.near_gate_low,
        near_gate_high=args.near_gate_high,
        near_reference_low=args.near_reference_low,
        near_reference_high=args.near_reference_high,
        missed_positive_delta=args.missed_positive_delta,
        missed_positive_se_multiple=args.missed_positive_se_multiple,
    )
    if args.selection_strategy == "stratified":
        deduped = stratified_for_replay(
            selected,
            include_fired=args.include_fired,
            include_suspected_false_positive=args.include_suspected_false_positive,
            include_missed_positive=args.include_missed_positive,
            include_near_fired=args.include_near_fired,
        )
    else:
        deduped = dedupe_for_replay(selected)
    runtime_rows = runtime_fired_rows(args.runtime_decision_log)
    write_jsonl(output_dir / "runtime_fired_states.jsonl", runtime_rows)
    write_jsonl(output_dir / "fired_states.jsonl", [selection_payload(item) for item in selected if item.audit_group == "fired_teacher"] + runtime_rows)
    write_jsonl(output_dir / "near_fired_states.jsonl", [selection_payload(item) for item in selected if item.audit_group == "near_fired"])
    write_jsonl(output_dir / "missed_positive_states.jsonl", [selection_payload(item) for item in selected if item.audit_group == "missed_positive"])
    write_jsonl(output_dir / "suspected_false_positive_states.jsonl", [selection_payload(item) for item in selected if item.audit_group == "suspected_false_positive"])
    write_jsonl(output_dir / "selected_teacher_states_deduped.jsonl", deduped)
    write_selection_summary(
        output_dir / "audit_state_selection_summary.md",
        selected,
        deduped,
        runtime_rows,
        selection_strategy=args.selection_strategy,
    )

    results: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if args.max_replay_states > 0 and deduped:
        partial_results_path = output_dir / "high_mc_audit_results.partial.jsonl"
        partial_actions_path = output_dir / "high_mc_action_ev_table.partial.jsonl"
        partial_failures_path = output_dir / "high_mc_audit_failures.partial.jsonl"
        for partial_path in (partial_results_path, partial_actions_path, partial_failures_path):
            if partial_path.exists():
                partial_path.unlink()
        metadata = load_cache_metadata(args.cache_dir)
        targets = {int(row["state_index"]) for row in deduped[: args.max_replay_states]}
        teacher_rows = load_teacher_rows_for_state_indices(metadata=metadata, repo_root=Path.cwd(), state_indices=targets)
        stage8_model = load_hu_turn2_stage8_model(args.hu_turn2_stage8_model, device=args.device)
        bundle = load_model_bundle_for_replay(args)
        batched_config = build_batched_config(args)
        batched_cache = HuTurn3DecisionCache(max_size=args.continuation_cache_size)
        batched_reference_cache = HuTurn3Stage3ReferenceCache(max_size=args.continuation_cache_size)
        batched_state_feature_cache = Stage3StateFeatureCache(max_size=args.continuation_cache_size)
        batched_action_cache = HuTurn3ActionCache(max_size=args.continuation_cache_size)
        final_turn_cache = FinalTurnDecisionCache(max_size=args.final_turn_cache_size)
        with _prediction_thread_context(args.prediction_threads):
            for selection in deduped[: args.max_replay_states]:
                state_index = int(selection["state_index"])
                sample = teacher_rows.get(state_index)
                if sample is None:
                    failures.append({**selection, "replay_status": "failed", "failure_reason": "teacher_sample_missing"})
                    continue
                try:
                    row, per_action = replay_one_state(
                        selection,
                        sample,
                        args=args,
                        bundle=bundle,
                        stage8_model=stage8_model,
                        batched_config=batched_config,
                        batched_cache=batched_cache,
                        batched_reference_cache=batched_reference_cache,
                        batched_state_feature_cache=batched_state_feature_cache,
                        batched_action_cache=batched_action_cache,
                        final_turn_cache=final_turn_cache,
                    )
                    row["diagnosis"] = diagnosis_for(row)
                    results.append(row)
                    action_rows.extend(per_action)
                    with partial_results_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    with partial_actions_path.open("a", encoding="utf-8") as handle:
                        for action_row in per_action:
                            handle.write(json.dumps(action_row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    print(json.dumps({"event": "high_mc_audit_replayed", "state_index": state_index, "mc": args.mc_samples, "diagnosis": row["diagnosis"]}, separators=(",", ":")), flush=True)
                except Exception as exc:
                    failure = {**selection, "replay_status": "failed", "failure_reason": str(exc) or "unknown"}
                    failures.append(failure)
                    with partial_failures_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(failure, ensure_ascii=False, separators=(",", ":")) + "\n")

    write_jsonl(output_dir / "high_mc_audit_results.jsonl", results)
    write_jsonl(output_dir / "high_mc_audit_failures.jsonl", failures)
    write_csv(output_dir / "high_mc_action_ev_table.csv", action_rows)
    write_csv(output_dir / "calibration_vs_high_mc.csv", calibration_vs_high_mc_rows(results))
    threshold_rows = threshold_sweep_rows(results)
    write_csv(output_dir / "threshold_sweep_high_mc.csv", threshold_rows)
    write_csv(output_dir / "diagnosis_breakdown.csv", group_breakdown(results, "diagnosis"))
    write_csv(output_dir / "position_breakdown.csv", group_breakdown(results, "seat"))
    write_csv(output_dir / "bucket_breakdown.csv", group_breakdown(results, "bucket_group"))
    write_audit_metrics_summary(output_dir, results, threshold_rows)
    write_recommended_next_step(output_dir / "recommended_next_step.md", results, failures, args.max_replay_states)

    summary = {
        "selected_rows": len(selected),
        "unique_teacher_states": len(deduped),
        "runtime_fired_states": len(runtime_rows),
        "selection_strategy": args.selection_strategy,
        "high_mc_successes": len(results),
        "high_mc_failures": len(failures),
        "mc_samples": args.mc_samples,
        "max_replay_states": args.max_replay_states,
        "elapsed_seconds": time.perf_counter() - started_at,
        "output_dir": str(output_dir),
        "production": "No-Go",
        "fifty_k_teacher": "defer",
        "t1": "defer",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
