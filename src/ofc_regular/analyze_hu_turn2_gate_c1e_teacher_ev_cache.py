"""Gate C1e teacher-EV feature-cache readiness audit.

Gate C1e is still calibration-only. It verifies whether a 20k-30k teacher-EV
feature cache exists and is replay-ready for the C1f expanded calibration gate.
It does not start production training, T1 training, 50k teacher generation, or
C2-small heldout evaluation.
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

from .analyze_hu_turn2_gate_c1_followup import (
    action_text,
    enriched_state_rows,
    load_action_original_indices,
)
from .analyze_hu_turn2_gate_c1_large_calibration import sample_source_group
from .analyze_hu_turn2_gate_c1d_threshold_repair import (
    c1d_strategy_specs,
    threshold_passes_strategy,
)
from .analyze_hu_turn2_pilot_calibration import load_model, rows_for_split, write_csv
from .build_hu_turn2_pilot_feature_cache import resolve_input_files
from .train_hu_turn2_pilot_model import load_cache, predict_all, target_matrix
from .train_torch_action_value import select_device


DEFAULT_TEACHER_INPUT = Path("outputs/hu_turn2_stage8_20k_mc512/hu_turn2_stage8_20k_mc512.jsonl")
DEFAULT_BUCKET_SIDECAR_DIR = Path("outputs/hu_turn2_stage8_20k_mc512")
DEFAULT_CACHE_DIR = Path(
    "D:/ofc-pineapple-storage/regular-ofc-pineapple/feature_cache/hu_t2_stage8_20k_mc512"
)
DEFAULT_MODEL = Path("models/hu_turn2_stage8_pilot_2000_mc512_reference_override_cached_rank_wide.pt")
DEFAULT_C1D_DIR = Path("outputs/hu_turn2_stage1_pilot_training_c1d_threshold_repair")
DEFAULT_OUTPUT_DIR = Path("outputs/hu_turn2_stage1_pilot_training_c1e_teacher_ev_feature_cache")
REPLAY_REQUIRED_SAMPLE_FIELDS = (
    "sample_id",
    "state_id",
    "seat",
    "source",
    "source_bucket",
    "board",
    "opponent_board",
    "dead_cards",
    "dealt",
    "actions",
    "baseline_action",
    "reference_action",
    "best_action",
    "future_rollout_seed",
    "rollout_count",
)
TARGET_UNBIASED_ROWS = 20_000
TARGET_ENRICHED_ROWS = 10_000
TARGET_MIN_ROWS = 20_000
TARGET_MAX_ROWS = 30_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-input", type=Path, default=DEFAULT_TEACHER_INPUT)
    parser.add_argument("--bucket-sidecar-dir", type=Path, default=DEFAULT_BUCKET_SIDECAR_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--c1d-dir", type=Path, default=DEFAULT_C1D_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--target-unbiased-rows", type=int, default=TARGET_UNBIASED_ROWS)
    parser.add_argument("--target-enriched-rows", type=int, default=TARGET_ENRICHED_ROWS)
    parser.add_argument("--target-min-rows", type=int, default=TARGET_MIN_ROWS)
    parser.add_argument("--target-max-rows", type=int, default=TARGET_MAX_ROWS)
    return parser.parse_args()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def action_original_index(action: Any) -> int | None:
    if isinstance(action, int):
        return int(action)
    if isinstance(action, dict) and action.get("original_index") is not None:
        return int(action["original_index"])
    if isinstance(action, dict) and isinstance(action.get("action"), dict):
        return action_original_index(action["action"])
    return None


def action_by_original_index(actions: list[Any]) -> dict[int, dict[str, Any]]:
    lookup: dict[int, dict[str, Any]] = {}
    for action in actions:
        original = action_original_index(action)
        if original is not None and isinstance(action, dict):
            lookup[int(original)] = action
    return lookup


def action_value(actions: list[Any], original_index: int) -> dict[str, Any] | None:
    return action_by_original_index(actions).get(int(original_index))


def compact_replay_packet(sample: dict[str, Any], row: dict[str, Any], *, teacher_path: Path, line_number: int) -> str:
    packet = {
        "teacher_jsonl": str(teacher_path),
        "line_number": line_number,
        "sample_id": sample.get("sample_id"),
        "state_id": sample.get("state_id"),
        "seat": sample.get("seat"),
        "source": sample.get("source"),
        "source_bucket": sample.get("source_bucket"),
        "board": sample.get("board"),
        "opponent_board": sample.get("opponent_board"),
        "dead_cards": sample.get("dead_cards"),
        "visible_dead_cards": sample.get("visible_dead_cards"),
        "dealt": sample.get("dealt"),
        "future_rollout_seed": sample.get("future_rollout_seed"),
        "rollout_count": sample.get("rollout_count"),
        "baseline_original_index": row.get("baseline_action_original_index"),
        "candidate_original_index": row.get("candidate_action_original_index"),
        "teacher_original_index": row.get("teacher_best_action_original_index"),
    }
    return json.dumps(packet, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def missing_replay_fields(sample: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for field in REPLAY_REQUIRED_SAMPLE_FIELDS:
        value = sample.get(field)
        if value is None or value == "" or value == [] or value == {}:
            missing.append(field)
    actions = sample.get("actions")
    if not isinstance(actions, list) or not actions:
        if "actions" not in missing:
            missing.append("actions")
    else:
        for index, action in enumerate(actions):
            if action_original_index(action) is None:
                missing.append(f"actions[{index}].original_index")
                break
    return missing


def c1e_split_for_row(row: dict[str, Any]) -> str:
    return "unbiased" if sample_source_group(row) == "policy_on_distribution" else "enriched"


def source_position_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        counts[(row["c1e_split"], row["source_group"], str(row.get("position", "unknown")))] += 1
    total = len(rows)
    return [
        {
            "c1e_split": split_name,
            "source_group": source,
            "position": seat,
            "rows": count,
            "share_total": count / max(total, 1),
        }
        for (split_name, source, seat), count in sorted(counts.items())
    ]


def build_event_row(
    *,
    row: dict[str, Any],
    sample: dict[str, Any],
    teacher_path: Path,
    line_number: int,
    cache_dir: Path,
    feature_dim: int,
) -> dict[str, Any]:
    actions = sample.get("actions") or []
    baseline_original = safe_int(row["baseline_action_original_index"])
    candidate_original = safe_int(row["candidate_action_original_index"])
    teacher_original = safe_int(row["teacher_best_action_original_index"])
    baseline_action = action_value(actions, baseline_original)
    candidate_action = action_value(actions, candidate_original)
    teacher_action = action_value(actions, teacher_original)
    missing = missing_replay_fields(sample)
    if baseline_action is None:
        missing.append("baseline_action_original_index_lookup")
    if candidate_action is None:
        missing.append("candidate_action_original_index_lookup")
    if teacher_action is None:
        missing.append("teacher_best_action_original_index_lookup")
    feature_start = safe_int(row["feature_start_offset"])
    feature_end = safe_int(row["feature_end_offset"])
    sample_id = sample.get("sample_id", row.get("sample_id"))
    return {
        "state_index": row["state_index"],
        "split": row["split"],
        "c1e_split": row["c1e_split"],
        "sample_id": sample_id,
        "source": sample.get("source", ""),
        "source_bucket": sample.get("source_bucket", ""),
        "source_group": row["source_group"],
        "position": sample.get("seat", row.get("seat", "")),
        "state_id": sample.get("state_id", ""),
        "replay_id": sample.get("state_id") or sample_id or row["state_index"],
        "run_bucket": row.get("run_bucket", ""),
        "bucket_group": row.get("bucket_group", ""),
        "predicted_bucket": row.get("predicted_bucket") or row.get("bucket_group") or row.get("run_bucket", ""),
        "source_bucket_requested": row.get("source_bucket_requested", ""),
        "source_bucket_actual": row.get("source_bucket_actual", ""),
        "actual_high_regret": row.get("actual_high_regret", ""),
        "actual_low_margin": row.get("actual_low_margin", ""),
        "actual_teacher_disagreement": row.get("actual_teacher_disagreement", ""),
        "gate_label": row.get("pilot_gate_label", ""),
        "features": f"mmap:{cache_dir / 'features.float32.mmap'}[{feature_start}:{feature_end}]",
        "feature_cache_dir": str(cache_dir),
        "feature_start_offset": feature_start,
        "feature_end_offset": feature_end,
        "feature_dim": feature_dim,
        "legal_actions": safe_int(row["action_count"]),
        "baseline_action": action_text(baseline_action),
        "candidate_action": action_text(candidate_action),
        "teacher_action": action_text(teacher_action),
        "baseline_action_original_index": baseline_original,
        "candidate_action_original_index": candidate_original,
        "teacher_action_original_index": teacher_original,
        "baseline_action_local_index": row["baseline_action_local_index"],
        "candidate_action_local_index": row["candidate_action_local_index"],
        "teacher_action_local_index": row["teacher_best_action_local_index"],
        "baseline_ev_mc512": row["baseline_action_ev"],
        "candidate_ev_mc512": row["candidate_action_ev"],
        "teacher_ev_mc512": row["teacher_best_action_ev"],
        "teacher_gain_mc512": row["actual_delta_candidate_vs_baseline"],
        "mc_n": sample.get("rollout_count", 0),
        "predicted_delta_vs_baseline": row["predicted_delta_vs_baseline"],
        "reference_margin_raw": row["reference_margin_raw"],
        "gate_score": row["gate_probability"],
        "original_index": candidate_original,
        "local_action_index": row["candidate_action_local_index"],
        "global_action_id": "",
        "replay_packet_path": f"{teacher_path}:{line_number}",
        "replay_packet_json": compact_replay_packet(sample, row, teacher_path=teacher_path, line_number=line_number),
        "replay_ready": int(not missing),
        "missing_replay_fields": ";".join(sorted(set(missing))),
        "false_positive_flag": int(safe_float(row["actual_delta_candidate_vs_baseline"]) < 0.0),
    }


def add_feature_offsets(rows: list[dict[str, Any]], cache: dict[str, Any]) -> None:
    offsets = cache["offsets"]
    for row in rows:
        index = safe_int(row["state_index"])
        row["feature_start_offset"] = int(offsets[index])
        row["feature_end_offset"] = int(offsets[index + 1])
        row["c1e_split"] = c1e_split_for_row(row)


def build_event_rows(
    *,
    teacher_input: Path,
    bucket_sidecar_dir: Path,
    cache_dir: Path,
    rows: list[dict[str, Any]],
    feature_dim: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_index = {safe_int(row["state_index"]): row for row in rows}
    event_rows: list[dict[str, Any]] = []
    readiness_rows: list[dict[str, Any]] = []
    state_index = 0
    for path, _run_bucket in resolve_input_files(teacher_input, bucket_sidecar_dir):
        for line_number, sample in enumerate(iter_jsonl(path), start=1):
            row = by_index.get(state_index)
            if row is None:
                state_index += 1
                continue
            event = build_event_row(
                row=row,
                sample=sample,
                teacher_path=path,
                line_number=line_number,
                cache_dir=cache_dir,
                feature_dim=feature_dim,
            )
            event_rows.append(event)
            readiness_rows.append(
                {
                    "state_index": event["state_index"],
                    "c1e_split": event["c1e_split"],
                    "sample_id": event["sample_id"],
                    "source_group": event["source_group"],
                    "position": event["position"],
                    "replay_ready": event["replay_ready"],
                    "missing_replay_fields": event["missing_replay_fields"],
                    "replay_packet_path": event["replay_packet_path"],
                    "baseline_action_recoverable": int(bool(event["baseline_action"])),
                    "candidate_action_recoverable": int(bool(event["candidate_action"])),
                    "teacher_action_recoverable": int(bool(event["teacher_action"])),
                }
            )
            state_index += 1
    return event_rows, readiness_rows


def candidate_fire_estimate_rows(rows: list[dict[str, Any]], c1d_dir: Path, *, target_unbiased: int, target_enriched: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    strategies = c1d_strategy_specs()
    c1d_metrics = {
        (row.get("candidate_id"), row.get("split")): row
        for row in read_csv(c1d_dir / "c1d_candidate_grid_metrics.csv")
    }
    for strategy in strategies:
        candidate_id = str(strategy["candidate_id"])
        for split_name in ("val", "test", "holdout", "all"):
            subset = rows if split_name == "all" else rows_for_split(rows, split_name)
            fired = [row for row in subset if threshold_passes_strategy(row, strategy)]
            unbiased_total = sum(1 for row in subset if row["c1e_split"] == "unbiased")
            enriched_total = sum(1 for row in subset if row["c1e_split"] == "enriched")
            unbiased_fired = sum(1 for row in fired if row["c1e_split"] == "unbiased")
            enriched_fired = sum(1 for row in fired if row["c1e_split"] == "enriched")
            projected_unbiased = unbiased_fired * (target_unbiased / max(unbiased_total, 1))
            projected_enriched = enriched_fired * (target_enriched / max(enriched_total, 1))
            gains = [safe_float(row["actual_delta_candidate_vs_baseline"]) for row in fired]
            fp_count = sum(1 for gain in gains if gain < 0.0)
            c1d_row = c1d_metrics.get((candidate_id, split_name), {})
            output.append(
                {
                    "candidate_id": candidate_id,
                    "split": split_name,
                    "current_rows": len(subset),
                    "current_fires": len(fired),
                    "current_unbiased_rows": unbiased_total,
                    "current_unbiased_fires": unbiased_fired,
                    "current_enriched_rows": enriched_total,
                    "current_enriched_fires": enriched_fired,
                    "projected_c1e_fires_unbiased20k": projected_unbiased,
                    "projected_c1e_fires_enriched10k": projected_enriched,
                    "projected_c1e_fires_total": projected_unbiased + projected_enriched,
                    "current_false_positive_count": fp_count,
                    "current_false_positive_rate": fp_count / max(len(fired), 1),
                    "current_teacher_avg_gain": float(np.mean(gains)) if gains else 0.0,
                    "c1d_status": c1d_row.get("c2_small_status", ""),
                    "c1d_blockers": c1d_row.get("c2_small_blockers", ""),
                }
            )
    return output


def gate_status(
    *,
    total_rows: int,
    unbiased_rows: int,
    enriched_rows: int,
    replay_ready_rows: int,
    missing_replay_rows: int,
    missing_source_position_rows: int,
    unrecoverable_action_rows: int,
    target_min: int,
    target_max: int,
    target_unbiased: int,
    target_enriched: int,
) -> tuple[str, list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    if total_rows < target_min:
        blockers.append("total_rows_lt_20k")
    if total_rows > target_max:
        blockers.append("total_rows_gt_30k")
    if unbiased_rows < target_unbiased:
        warnings.append("unbiased_rows_lt_20k")
    if enriched_rows < target_enriched:
        warnings.append("enriched_rows_lt_10k")
    if replay_ready_rows != total_rows:
        blockers.append("replay_ready_not_100pct")
    if missing_replay_rows:
        blockers.append("missing_replay_fields")
    if missing_source_position_rows:
        blockers.append("missing_source_or_position")
    if unrecoverable_action_rows:
        blockers.append("unrecoverable_baseline_candidate_or_teacher_action")
    return ("hard_pass" if not blockers else "blocked", blockers, warnings)


def write_summary(
    output_dir: Path,
    *,
    total_rows: int,
    unbiased_rows: int,
    enriched_rows: int,
    readiness_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    fire_rows: list[dict[str, Any]],
    status: str,
    blockers: list[str],
    warnings: list[str],
    elapsed: float,
) -> None:
    replay_ready = sum(safe_int(row["replay_ready"]) for row in readiness_rows)
    missing = len(readiness_rows) - replay_ready
    strong_pass = status == "hard_pass" and not warnings
    best_estimates = sorted(
        [row for row in fire_rows if row["split"] == "all"],
        key=lambda row: safe_float(row["projected_c1e_fires_total"]),
        reverse=True,
    )[:6]
    lines = [
        "# Gate C1e Teacher-EV Feature Cache Summary",
        "",
        "Gate C1e is calibration-only. It does not authorize 50k teacher, T1, C2-small, or production training.",
        "",
        f"- execution status: `{status}`",
        f"- blockers: `{';'.join(blockers) if blockers else 'none'}`",
        f"- warnings: `{';'.join(warnings) if warnings else 'none'}`",
        f"- strong pass: `{strong_pass}`",
        f"- total rows: `{total_rows}`",
        f"- unbiased rows: `{unbiased_rows}`",
        f"- enriched rows: `{enriched_rows}`",
        f"- replay-ready rows: `{replay_ready}/{total_rows}`",
        f"- missing replay rows: `{missing}`",
        f"- elapsed seconds: `{elapsed:.2f}`",
        "",
        "## Source / Position",
        "",
        "| c1e_split | source | position | rows |",
        "|---|---|---|---:|",
    ]
    for row in source_rows:
        lines.append(
            f"| {row['c1e_split']} | {row['source_group']} | {row['position']} | {row['rows']} |"
        )
    lines.extend(
        [
            "",
            "## Bucket Schema",
            "",
            "- `predicted_bucket` is the normalized alias for `bucket_group`; it is derived from `run_bucket` for predicted-pool shards.",
            "- `run_bucket` preserves the physical input bucket name.",
            "- `source_bucket_requested` / `source_bucket_actual` preserve sampler intent and actual accepted source when available.",
            "- actual bucket labels are `actual_high_regret`, `actual_low_margin`, and `actual_teacher_disagreement`.",
            "- `gate_label` is the explicit alias for `pilot_gate_label`.",
            "",
            "## T2 Margin Scale",
            "",
            "`reference_margin_raw` is a T2 baseline/reference score margin on the T2 model scale. It is not comparable to the T3 Stage7 `hu_turn3_reference_min_margin=10.0` gate and should not be treated as a strong hard gate unless validated separately.",
            "",
            "## Candidate Fire Estimate",
            "",
            "| candidate | split | current fires | projected C1e fires | current FP rate |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in best_estimates:
        lines.append(
            "| {candidate_id} | {split} | {current_fires} | {projected:.1f} | {fp:.3f} |".format(
                candidate_id=row["candidate_id"],
                split=row["split"],
                current_fires=row["current_fires"],
                projected=safe_float(row["projected_c1e_fires_total"]),
                fp=safe_float(row["current_false_positive_rate"]),
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- production training: `No-Go`",
            "- T1 training: `No-Go`",
            "- 50k teacher: `No-Go`",
            "- C2-small heldout: `No-Go`",
            "- C1f expanded calibration: `" + ("Go" if status == "hard_pass" else "No-Go") + "`",
        ]
    )
    (output_dir / "c1e_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    go_lines = [
        "# Gate C1e Go/No-Go For C1f",
        "",
        f"- C1e hard pass: `{status == 'hard_pass'}`",
        f"- C1e strong pass: `{strong_pass}`",
        f"- blockers: `{';'.join(blockers) if blockers else 'none'}`",
        f"- warnings: `{';'.join(warnings) if warnings else 'none'}`",
        "- C1f expanded calibration: `" + ("Go" if status == "hard_pass" else "No-Go") + "`",
        "- C2-small heldout: `No-Go`",
        "- 50k teacher: `No-Go`",
        "- T1 training: `No-Go`",
        "- production training: `No-Go`",
        "",
        "C1f requires a 20k-30k teacher-EV cache with 100% replay-ready rows. Unbiased/enriched target misses are reported as warnings so the expanded calibration can still measure source bias explicitly.",
    ]
    (output_dir / "go_nogo_for_c1f.md").write_text("\n".join(go_lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    started_at = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch

    device = select_device(torch, args.device)
    cache = load_cache(args.cache_dir)
    net, stats, _payload = load_model(torch, args.model, device)
    predictions = predict_all(torch, net, cache, stats, device, args.batch_size)
    targets = target_matrix(cache)
    action_original_indices = load_action_original_indices(args.cache_dir, int(cache["metadata"]["action_count"]))
    rows = enriched_state_rows(cache, predictions, targets, action_original_indices)
    add_feature_offsets(rows, cache)

    feature_dim = int(cache["metadata"].get("feature_dim", 0))
    event_rows, readiness_rows = build_event_rows(
        teacher_input=args.teacher_input,
        bucket_sidecar_dir=args.bucket_sidecar_dir,
        cache_dir=args.cache_dir,
        rows=rows,
        feature_dim=feature_dim,
    )
    if len(event_rows) != len(rows):
        missing_indices = set(range(len(rows))) - {safe_int(row["state_index"]) for row in event_rows}
        for index in sorted(missing_indices):
            readiness_rows.append(
                {
                    "state_index": index,
                    "c1e_split": rows[index].get("c1e_split", ""),
                    "sample_id": rows[index].get("sample_id", ""),
                    "source_group": rows[index].get("source_group", ""),
                    "position": rows[index].get("seat", ""),
                    "replay_ready": 0,
                    "missing_replay_fields": "teacher_jsonl_row_missing",
                    "replay_packet_path": "",
                    "baseline_action_recoverable": 0,
                    "candidate_action_recoverable": 0,
                    "teacher_action_recoverable": 0,
                }
            )

    unbiased_rows = [row for row in event_rows if row["c1e_split"] == "unbiased"]
    enriched_rows_out = [row for row in event_rows if row["c1e_split"] == "enriched"]
    merged_rows = sorted(event_rows, key=lambda row: safe_int(row["state_index"]))
    source_rows = source_position_rows(merged_rows)
    fire_rows = candidate_fire_estimate_rows(
        rows,
        args.c1d_dir,
        target_unbiased=args.target_unbiased_rows,
        target_enriched=args.target_enriched_rows,
    )
    replay_ready = sum(safe_int(row["replay_ready"]) for row in readiness_rows)
    missing_replay = len(readiness_rows) - replay_ready
    missing_source_position = sum(
        1
        for row in merged_rows
        if not str(row.get("source", "")).strip() or not str(row.get("position", "")).strip()
    )
    unrecoverable_actions = sum(
        1
        for row in readiness_rows
        if not (
            safe_int(row.get("baseline_action_recoverable"))
            and safe_int(row.get("candidate_action_recoverable"))
            and safe_int(row.get("teacher_action_recoverable"))
        )
    )
    status, blockers, warnings = gate_status(
        total_rows=len(merged_rows),
        unbiased_rows=len(unbiased_rows),
        enriched_rows=len(enriched_rows_out),
        replay_ready_rows=replay_ready,
        missing_replay_rows=missing_replay,
        missing_source_position_rows=missing_source_position,
        unrecoverable_action_rows=unrecoverable_actions,
        target_min=args.target_min_rows,
        target_max=args.target_max_rows,
        target_unbiased=args.target_unbiased_rows,
        target_enriched=args.target_enriched_rows,
    )

    write_csv(args.output_dir / "teacher_ev_feature_cache_unbiased.csv", unbiased_rows)
    write_csv(args.output_dir / "teacher_ev_feature_cache_enriched.csv", enriched_rows_out)
    write_csv(args.output_dir / "teacher_ev_feature_cache_merged.csv", merged_rows)
    write_csv(args.output_dir / "replay_readiness.csv", readiness_rows)
    write_csv(args.output_dir / "source_position_distribution.csv", source_rows)
    write_csv(args.output_dir / "candidate_fire_estimate.csv", fire_rows)
    write_summary(
        args.output_dir,
        total_rows=len(merged_rows),
        unbiased_rows=len(unbiased_rows),
        enriched_rows=len(enriched_rows_out),
        readiness_rows=readiness_rows,
        source_rows=source_rows,
        fire_rows=fire_rows,
        status=status,
        blockers=blockers,
        warnings=warnings,
        elapsed=time.perf_counter() - started_at,
    )
    print(
        json.dumps(
            {
                "gate": "C1e",
                "status": status,
                "blockers": blockers,
                "warnings": warnings,
                "rows": len(merged_rows),
                "unbiased_rows": len(unbiased_rows),
                "enriched_rows": len(enriched_rows_out),
                "replay_ready_rows": replay_ready,
                "missing_source_position_rows": missing_source_position,
                "unrecoverable_action_rows": unrecoverable_actions,
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
