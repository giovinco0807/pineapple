"""Prepare HU Turn2 Stage8b safe-override labels before large training.

Stage8b is a retraining step for the existing Stage8 selective override model.
It keeps the deployable five-head model shape but changes the gate head target
from a broad pilot gate to a safe override target derived from teacher LCB and
hard-negative diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .analyze_hu_turn2_gate_c1_followup import enriched_state_rows, load_action_original_indices
from .analyze_hu_turn2_gate_c1f_expanded_calibration import attach_c1e_fields, load_c1e_index
from .analyze_hu_turn2_pilot_calibration import load_model, write_csv
from .train_hu_turn2_pilot_model import load_cache, predict_all, scoring_metadata_status, target_matrix
from .train_torch_action_value import select_device


DEFAULT_CACHE_DIR = Path("D:/ofc-pineapple-storage/regular-ofc-pineapple/feature_cache/hu_t2_stage8_20k_mc512")
DEFAULT_MODEL = Path("models/hu_turn2_stage8_broad_20k_mc512_reference_override_cached_rank_wide.pt")
DEFAULT_C1E_DIR = Path("outputs/hu_turn2_stage1_pilot_training_c1e_teacher_ev_feature_cache")
DEFAULT_C3_POSTMORTEM_DIR = Path("outputs/evals/hu_turn2_stage8_c3_postmortem")
DEFAULT_TOPK_HARD_NEGATIVES = Path(
    "outputs/evals/hu_turn2_stage8b_topk_hard_negatives/topk_false_positive_hard_negatives.jsonl"
)
DEFAULT_COUNTERFACTUAL_LOSS_TARGETS = Path(
    "outputs/evals/hu_turn2_stage8b_counterfactual_loss_targets/topk_counterfactual_loss_targets.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("outputs/hu_turn2_stage8b_prelarge_training")
LABEL_TO_ID = {"negative": 0, "gray": 1, "positive": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--c1e-dir", type=Path, default=DEFAULT_C1E_DIR)
    parser.add_argument("--c3-postmortem-dir", type=Path, default=DEFAULT_C3_POSTMORTEM_DIR)
    parser.add_argument(
        "--topk-hard-negatives-jsonl",
        type=Path,
        action="append",
        default=None,
        help=(
            "TopK hard-negative JSONL. Can be repeated. "
            "Defaults to the standard TopK hard-negative pack when omitted."
        ),
    )
    parser.add_argument("--counterfactual-loss-targets-jsonl", type=Path, default=DEFAULT_COUNTERFACTUAL_LOSS_TARGETS)
    parser.add_argument("--high-mc-results-jsonl", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--max-high-mc-states", type=int, default=200)
    return parser.parse_args()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def gate_label_from_lcb(row: dict[str, Any], *, lcb_column: str, ucb_z: float) -> str:
    gain = safe_float(row.get("actual_delta_candidate_vs_baseline"))
    stderr = safe_float(row.get("gain_stderr_proxy"))
    lcb = safe_float(row.get(lcb_column), gain - ucb_z * stderr)
    ucb = gain + ucb_z * stderr
    if lcb > 0.0:
        return "positive"
    if ucb < 0.0 or gain <= -0.25:
        return "negative"
    return "gray"


def current_proxy_fires(row: dict[str, Any], *, min_margin: float = 2.5, gate_threshold: float = 0.90) -> bool:
    return (
        safe_int(row.get("candidate_is_baseline")) == 0
        and safe_float(row.get("predicted_delta_vs_baseline")) >= min_margin
        and safe_float(row.get("reference_margin_raw")) >= 0.0
        and safe_float(row.get("gate_probability")) >= gate_threshold
    )


def stage8b_weight(row: dict[str, Any], *, label: str, hard_negative: bool) -> float:
    if hard_negative:
        return 6.0
    if label == "positive":
        return 2.0
    if label == "negative":
        return 1.5
    return 0.15


def load_high_mc_index(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    return {safe_int(row.get("state_index"), -1): row for row in read_jsonl(path) if safe_int(row.get("state_index"), -1) >= 0}


def high_mc_label(row: dict[str, Any], high_mc: dict[str, Any]) -> str:
    delta = safe_float(high_mc.get("high_mc_delta_candidate_vs_baseline"))
    lower95 = safe_float(high_mc.get("high_mc_lower95_candidate_vs_baseline"))
    if lower95 > 0.0:
        return "positive"
    if delta <= -0.05:
        return "negative"
    return "gray"


def attach_stage8b_labels(rows: list[dict[str, Any]], high_mc_rows: dict[int, dict[str, Any]] | None = None) -> None:
    high_mc_rows = high_mc_rows or {}
    for row in rows:
        lcb196 = safe_float(row.get("gain_lcb_1p96"))
        lcb164 = safe_float(row.get("gain_lcb_1p64"))
        gain = safe_float(row.get("actual_delta_candidate_vs_baseline"))
        proxy_fired = current_proxy_fires(row)
        hard_negative = proxy_fired and gain <= 0.0 and safe_int(row.get("candidate_is_baseline")) == 0
        label196 = "negative" if hard_negative else gate_label_from_lcb(row, lcb_column="gain_lcb_1p96", ucb_z=1.96)
        label164 = "negative" if hard_negative else gate_label_from_lcb(row, lcb_column="gain_lcb_1p64", ucb_z=1.64)
        high_mc = high_mc_rows.get(safe_int(row.get("state_index"), -1))
        high_mc_source = ""
        if high_mc:
            high_mc_source = "mc4096"
            label196 = high_mc_label(row, high_mc)
            label164 = label196
            if safe_float(high_mc.get("high_mc_delta_candidate_vs_baseline")) <= 0.0 and proxy_fired:
                hard_negative = True
        row["teacher_delta_lcb_196"] = lcb196
        row["teacher_delta_lcb_164"] = lcb164
        row["high_mc_label_source"] = high_mc_source
        row["high_mc_delta_candidate_vs_baseline"] = safe_float(high_mc.get("high_mc_delta_candidate_vs_baseline")) if high_mc else ""
        row["high_mc_lower95_candidate_vs_baseline"] = (
            safe_float(high_mc.get("high_mc_lower95_candidate_vs_baseline")) if high_mc else ""
        )
        row["high_mc_diagnosis"] = str(high_mc.get("diagnosis", "")) if high_mc else ""
        row["safe_lcb196_label"] = label196
        row["safe_lcb164_label"] = label164
        row["safe_lcb196_gate_label_id"] = LABEL_TO_ID[label196]
        row["safe_lcb164_gate_label_id"] = LABEL_TO_ID[label164]
        row["hard_negative_label"] = int(hard_negative)
        row["gray_label"] = int(label196 == "gray")
        row["current_proxy_m2p5_g0p9_fired"] = int(proxy_fired)
        row["stage8b_gate_weight"] = stage8b_weight(row, label=label196, hard_negative=hard_negative)
        row["stage8b_priority_score"] = priority_score(row)
        row["hard_negative_source"] = "high_mc_runtime_negative" if hard_negative and high_mc_source else ("current_proxy_teacher_negative" if hard_negative else "")


def priority_score(row: dict[str, Any]) -> float:
    if safe_int(row.get("hard_negative_label")):
        return 1000.0 + safe_float(row.get("candidate_loss")) + safe_float(row.get("predicted_delta_vs_baseline"))
    if safe_float(row.get("teacher_delta_lcb_196")) > 0.0 and not safe_int(row.get("current_proxy_m2p5_g0p9_fired")):
        return 500.0 + safe_float(row.get("teacher_delta_lcb_196"))
    if abs(safe_float(row.get("teacher_delta_lcb_196"))) <= 0.25:
        return 250.0 - abs(safe_float(row.get("teacher_delta_lcb_196")))
    return safe_float(row.get("teacher_delta_lcb_196"))


def label_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = (
        "state_index",
        "split",
        "c1e_split",
        "run_bucket",
        "bucket_group",
        "source_bucket",
        "source_group",
        "position",
        "seat",
        "actual_high_regret",
        "actual_low_margin",
        "actual_teacher_disagreement",
        "candidate_is_baseline",
        "candidate_action_local_index",
        "candidate_action_original_index",
        "baseline_action_local_index",
        "baseline_action_original_index",
        "actual_delta_candidate_vs_baseline",
        "candidate_loss",
        "gain_stderr_proxy",
        "teacher_delta_lcb_196",
        "teacher_delta_lcb_164",
        "high_mc_label_source",
        "high_mc_delta_candidate_vs_baseline",
        "high_mc_lower95_candidate_vs_baseline",
        "high_mc_diagnosis",
        "predicted_delta_vs_baseline",
        "gate_probability",
        "predicted_EV_margin_top1_top2",
        "reference_margin_raw",
        "safe_lcb196_label",
        "safe_lcb164_label",
        "safe_lcb196_gate_label_id",
        "safe_lcb164_gate_label_id",
        "hard_negative_label",
        "gray_label",
        "current_proxy_m2p5_g0p9_fired",
        "stage8b_gate_weight",
        "stage8b_priority_score",
        "hard_negative_source",
        "replay_ready",
    )
    return [{key: row.get(key, "") for key in keep} for row in rows]


def breakdown_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    groups: list[tuple[str, str, list[dict[str, Any]]]] = [("overall", "all", rows)]
    for field in ("split", "c1e_split", "bucket_group", "position", "safe_lcb196_label"):
        for value in sorted({str(row.get(field, "unknown")) for row in rows}):
            groups.append((field, value, [row for row in rows if str(row.get(field, "unknown")) == value]))
    for group_field, group_value, subset in groups:
        labels = Counter(str(row.get("safe_lcb196_label")) for row in subset)
        output.append(
            {
                "group_field": group_field,
                "group_value": group_value,
                "rows": len(subset),
                "safe_lcb196_positive": labels.get("positive", 0),
                "safe_lcb196_gray": labels.get("gray", 0),
                "safe_lcb196_negative": labels.get("negative", 0),
                "hard_negative": sum(safe_int(row.get("hard_negative_label")) for row in subset),
                "current_proxy_fired": sum(safe_int(row.get("current_proxy_m2p5_g0p9_fired")) for row in subset),
                "avg_teacher_gain": float(np.mean([safe_float(row.get("actual_delta_candidate_vs_baseline")) for row in subset]))
                if subset
                else 0.0,
            }
        )
    return output


def selected_high_mc_states(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    buckets: list[tuple[str, list[dict[str, Any]], int]] = [
        (
            "hard_negative_teacher",
            [row for row in rows if safe_int(row.get("hard_negative_label"))],
            max(25, limit // 4),
        ),
        (
            "safe_positive_missed_proxy",
            [
                row
                for row in rows
                if safe_float(row.get("teacher_delta_lcb_196")) > 0.0
                and not safe_int(row.get("current_proxy_m2p5_g0p9_fired"))
                and safe_int(row.get("candidate_is_baseline")) == 0
            ],
            max(25, limit // 4),
        ),
        (
            "near_lcb_boundary",
            [
                row
                for row in rows
                if abs(safe_float(row.get("teacher_delta_lcb_196"))) <= 0.25
                and safe_int(row.get("candidate_is_baseline")) == 0
            ],
            max(25, limit // 4),
        ),
        (
            "high_confidence_positive",
            [
                row
                for row in rows
                if safe_float(row.get("teacher_delta_lcb_196")) > 0.0
                and safe_int(row.get("current_proxy_m2p5_g0p9_fired"))
            ],
            max(25, limit // 4),
        ),
    ]
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for group, group_rows, count in buckets:
        sorted_rows = sorted(group_rows, key=lambda row: safe_float(row.get("stage8b_priority_score")), reverse=True)
        for row in sorted_rows:
            state_index = safe_int(row.get("state_index"), -1)
            if state_index in seen:
                continue
            selected.append(selection_payload(row, group, len(selected)))
            seen.add(state_index)
            if sum(1 for item in selected if item["replay_origin_group"] == group) >= count:
                break
            if len(selected) >= limit:
                return selected
    if len(selected) < limit:
        for row in sorted(rows, key=lambda item: safe_float(item.get("stage8b_priority_score")), reverse=True):
            state_index = safe_int(row.get("state_index"), -1)
            if state_index in seen:
                continue
            selected.append(selection_payload(row, "priority_backfill", len(selected)))
            seen.add(state_index)
            if len(selected) >= limit:
                break
    return selected


def selection_payload(row: dict[str, Any], group: str, order: int) -> dict[str, Any]:
    return {
        "state_index": safe_int(row.get("state_index")),
        "selection_order": order,
        "replay_origin_group": group,
        "stage8b_priority_score": safe_float(row.get("stage8b_priority_score")),
        "split": row.get("split"),
        "seat": row.get("seat"),
        "bucket_group": row.get("bucket_group"),
        "candidate_action_local_index": safe_int(row.get("candidate_action_local_index"), -1),
        "candidate_action_original_index": safe_int(row.get("candidate_action_original_index"), -1),
        "baseline_action_local_index": safe_int(row.get("baseline_action_local_index"), -1),
        "baseline_action_original_index": safe_int(row.get("baseline_action_original_index"), -1),
        "actual_delta_candidate_vs_baseline": safe_float(row.get("actual_delta_candidate_vs_baseline")),
        "teacher_delta_lcb_196": safe_float(row.get("teacher_delta_lcb_196")),
        "predicted_delta_vs_baseline": safe_float(row.get("predicted_delta_vs_baseline")),
        "gate_probability": safe_float(row.get("gate_probability")),
        "hard_negative_label": safe_int(row.get("hard_negative_label")),
    }


def c3_runtime_loss_rows(c3_postmortem_dir: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(c3_postmortem_dir / "c3_top_loss_audit_states.jsonl")
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        payload = dict(row)
        payload["selection_order"] = index
        payload["replay_origin_group"] = "c3_runtime_top_loss"
        output.append(payload)
    return output


def topk_hard_negative_rows(paths: Path | Iterable[Path] | None) -> list[dict[str, Any]]:
    if paths is None:
        return []
    if isinstance(paths, Path):
        path_list = [paths]
    else:
        path_list = [Path(path) for path in paths]
    output: list[dict[str, Any]] = []
    for path in path_list:
        if not path.exists():
            continue
        rows = read_jsonl(path)
        for row in rows:
            payload = dict(row)
            payload["selection_order"] = len(output)
            payload["replay_origin_group"] = "topk_mc_false_positive"
            payload["replay_source"] = "stage8b_topk_hard_negative_pack"
            payload["replay_source_path"] = str(path)
            payload["requires_replay_before_training"] = True
            output.append(payload)
    return output


def whole_game_risk_target_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows = read_jsonl(path)
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        payload = dict(row)
        payload["selection_order"] = index
        payload["replay_origin_group"] = "stage8b_topk_whole_game_loss"
        payload["replay_source"] = "stage8b_topk_counterfactual_loss_targets"
        payload["requires_separate_whole_game_risk_head"] = True
        payload["do_not_use_as_local_ev_hard_negative"] = safe_int(payload.get("use_for_local_ev_hard_negative")) == 0
        output.append(payload)
    return output


def write_summary(
    output_dir: Path,
    *,
    rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    c3_runtime_losses: list[dict[str, Any]],
    topk_hard_negatives: list[dict[str, Any]],
    whole_game_risk_targets: list[dict[str, Any]],
    scoring_status: dict[str, Any],
    elapsed: float,
    args: argparse.Namespace,
) -> None:
    counts = Counter(str(row.get("safe_lcb196_label")) for row in rows)
    hard_negatives = sum(safe_int(row.get("hard_negative_label")) for row in rows)
    high_mc_labeled = sum(1 for row in rows if row.get("high_mc_label_source"))
    proxy_fired = sum(safe_int(row.get("current_proxy_m2p5_g0p9_fired")) for row in rows)
    replay_ready = sum(safe_int(row.get("replay_ready")) for row in rows)
    risk_only_count = sum(safe_int(row.get("use_for_whole_game_risk_head")) for row in whole_game_risk_targets)
    local_ev_hard_negative_from_topk = sum(safe_int(row.get("use_for_local_ev_hard_negative")) for row in whole_game_risk_targets)
    scoring_ready = bool(scoring_status.get("training_allowed"))
    ready_status = (
        "Ready to launch with current labels"
        if scoring_ready
        else "No-Go until the feature cache is rerolled/rebuilt under the current scoring objective"
    )
    training_command = (
        "python -m ofc_regular.train_hu_turn2_pilot_model "
        f"--cache-dir {args.cache_dir} "
        f"--stage8b-labels-csv {args.output_dir / 'stage8b_safe_override_labels.csv'} "
        "--gate-label-column safe_lcb196_gate_label_id "
        "--gate-weight-column stage8b_gate_weight "
        "--model-output models/hu_turn2_stage8b_safe_lcb196_20k_reference_override_cached_rank_wide.pt "
        "--output-dir outputs/training/hu_turn2_stage8b_safe_lcb196_20k"
    )
    lines = [
        "# HU Turn2 Stage8b Pre-Large-Training Prep",
        "",
        "This prepares labels and selected replay inputs. It does not start large training, 50k teacher generation, T1, or production.",
        "",
        f"- rows: `{len(rows)}`",
        f"- replay-ready rows: `{replay_ready}`",
        f"- safe_lcb196 positive / gray / negative: `{counts.get('positive', 0)}` / `{counts.get('gray', 0)}` / `{counts.get('negative', 0)}`",
        f"- hard negatives: `{hard_negatives}`",
        f"- high-MC label overrides: `{high_mc_labeled}`",
        f"- current proxy m2.5/g0.9 fired: `{proxy_fired}`",
        f"- selected teacher high-MC states: `{len(selected)}`",
        f"- C3 runtime top-loss replay states: `{len(c3_runtime_losses)}`",
        f"- TopK runtime hard-negative replay states: `{len(topk_hard_negatives)}`",
        f"- TopK whole-game risk-only targets: `{risk_only_count}`",
        f"- TopK local-EV hard negatives from counterfactual audit: `{local_ev_hard_negative_from_topk}`",
        f"- scoring objective status: `{scoring_status.get('status')}`",
        f"- cache FL EV 14: `{scoring_status.get('cache_fl_ev_14')}`",
        f"- current FL EV 14: `{scoring_status.get('expected_fl_ev_14')}`",
        f"- elapsed seconds: `{elapsed:.2f}`",
        "",
        "## Large Training Command",
        "",
        "```powershell",
        training_command,
        "```",
        "",
        "## Decision",
        "",
        f"- Stage8b safe-LCB large training: `{ready_status}`",
        "- TopK hard negatives: `Replay/feature-generate before using as supervised labels`",
        "- TopK whole-game risk-only rows: `Do not mix into local EV/safe-LCB gate labels; train a separate risk/counterfactual head first`",
        "- 50k teacher: `No-Go`",
        "- T1: `No-Go`",
        "- production / P2 fixed: `No-Go`",
    ]
    (output_dir / "stage8b_prep_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    readiness = {
        "schema": "hu_turn2_stage8b_prelarge_training_v1",
        "rows": len(rows),
        "replay_ready_rows": replay_ready,
        "safe_lcb196_counts": dict(counts),
        "hard_negative_rows": hard_negatives,
        "high_mc_label_overrides": high_mc_labeled,
        "current_proxy_m2p5_g0p9_fired": proxy_fired,
        "selected_high_mc_states": len(selected),
        "c3_runtime_top_loss_replay_states": len(c3_runtime_losses),
        "topk_hard_negative_replay_states": len(topk_hard_negatives),
        "topk_hard_negatives_require_replay_before_training": bool(topk_hard_negatives),
        "whole_game_risk_targets": len(whole_game_risk_targets),
        "whole_game_risk_only_targets": risk_only_count,
        "topk_local_ev_hard_negatives_from_counterfactual_audit": local_ev_hard_negative_from_topk,
        "whole_game_risk_targets_require_separate_head": bool(whole_game_risk_targets),
        "scoring_metadata_status": scoring_status,
        "large_training_ready": scoring_ready,
        "safe_lcb_training_ready": scoring_ready,
        "risk_head_training_ready": False,
        "large_training_command": training_command,
        "fifty_k_teacher": "No-Go",
        "t1": "No-Go",
        "production": "No-Go",
        "elapsed_seconds": elapsed,
    }
    (output_dir / "stage8b_training_ready.json").write_text(json.dumps(readiness, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch

    device = select_device(torch, args.device)
    cache = load_cache(args.cache_dir)
    scoring_status = scoring_metadata_status(cache["metadata"])
    net, stats, _payload = load_model(torch, args.model, device)
    predictions = predict_all(torch, net, cache, stats, device, args.batch_size)
    targets = target_matrix(cache)
    original_indices = load_action_original_indices(args.cache_dir, int(cache["metadata"]["action_count"]))
    rows = enriched_state_rows(cache, predictions, targets, original_indices)
    attach_c1e_fields(rows, load_c1e_index(args.c1e_dir))
    high_mc_rows = load_high_mc_index(args.high_mc_results_jsonl)
    attach_stage8b_labels(rows, high_mc_rows)

    selected = selected_high_mc_states(rows, limit=args.max_high_mc_states)
    c3_losses = c3_runtime_loss_rows(args.c3_postmortem_dir)
    topk_hard_negative_paths = args.topk_hard_negatives_jsonl or [DEFAULT_TOPK_HARD_NEGATIVES]
    topk_hard_negatives = topk_hard_negative_rows(topk_hard_negative_paths)
    whole_game_risk_targets = whole_game_risk_target_rows(args.counterfactual_loss_targets_jsonl)

    write_csv(args.output_dir / "stage8b_safe_override_labels.csv", label_rows(rows))
    write_csv(args.output_dir / "stage8b_label_breakdown.csv", breakdown_rows(rows))
    hard_negative_rows = [row for row in label_rows(rows) if safe_int(row.get("hard_negative_label"))]
    write_csv(args.output_dir / "stage8b_hard_negative_candidates.csv", hard_negative_rows)
    write_jsonl(args.output_dir / "stage8b_selected_high_mc_states.jsonl", selected)
    write_jsonl(args.output_dir / "stage8b_c3_runtime_top_loss_replay_states.jsonl", c3_losses)
    write_jsonl(args.output_dir / "stage8b_topk_hard_negative_replay_states.jsonl", topk_hard_negatives)
    write_jsonl(args.output_dir / "stage8b_whole_game_risk_targets.jsonl", whole_game_risk_targets)
    write_summary(
        args.output_dir,
        rows=rows,
        selected=selected,
        c3_runtime_losses=c3_losses,
        topk_hard_negatives=topk_hard_negatives,
        whole_game_risk_targets=whole_game_risk_targets,
        scoring_status=scoring_status,
        elapsed=time.perf_counter() - started,
        args=args,
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "selected_high_mc_states": len(selected),
                "topk_hard_negative_replay_states": len(topk_hard_negatives),
                "whole_game_risk_targets": len(whole_game_risk_targets),
                "output_dir": str(args.output_dir),
                "scoring_metadata_status": scoring_status,
                "large_training_ready": bool(scoring_status.get("training_allowed")),
                "risk_head_training_ready": False,
                "fifty_k_teacher": "No-Go",
                "t1": "No-Go",
                "production": "No-Go",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
