"""Benchmark T2 hybrid runtime with T3 union exact refinement.

This is a local harness for the serving path:

    T2 action-value shortlist -> sync candidates -> sampled T3 deal
    -> T3 union candidate pool -> Rust exact rerank.

The benchmark intentionally writes full per-position output so real hands and
placements can be inspected after a run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.tutor.exact_late import action_key
from ai.tutor.hybrid_t1t2 import (
    HybridConfig,
    _runtime_config_mode_payload,
    _turn_model_cascade_switch_ensemble_defaults,
    _turn_model_conditional_ensemble_defaults,
    _turn_model_conditional_switch_ensemble_defaults,
    _turn_model_ensemble_defaults,
    evaluate_hybrid_position,
    load_t2_sync_selector,
    make_action_value_evaluator,
    parse_turn_model_cascade_switch_ensemble_specs,
    parse_turn_model_conditional_ensemble_specs,
    parse_turn_model_conditional_switch_ensemble_specs,
    parse_turn_model_ensemble_specs,
)
from ai.tutor.t3_runtime import T3UnionCandidatePool

DEFAULT_INPUT = Path("ai/data/hybrid_t1t2_active_20260531/local_t1t2_500_mc300.jsonl")
DEFAULT_RUNTIME_CONFIG = Path("ai/config/t1t2_t3_union_runtime_20260607.json")
DEFAULT_OUTPUT_DIR = Path("D:/ofc-pineapple-data/t3_evloss_fresh_20260607/t2_t3_union_benchmark_20260607")

CONFIG_OVERRIDES = (
    "shortlist_k",
    "insurance_k",
    "sync_exact_k",
    "t2_sync_exact_k",
    "max_pool",
    "time_budget_ms",
    "t2_initial_samples_per_candidate",
    "t2_extra_samples_per_round",
    "t2_max_samples_per_candidate",
    "t2_close_candidate_limit",
    "t2_baseline_min_samples",
    "t2_model_rank_min_samples_k",
    "t2_model_rank_min_samples",
    "t2_model_rank_min_samples_min_remaining_ms",
    "t2_post_refine_sync_exact_k",
    "t2_post_refine_min_remaining_ms",
    "t2_post_refine_policy",
    "t2_post_refine_extra_model_score_min",
    "t2_sync_model_insurance_k",
    "t2_sync_model_insurance_top1_kk_min",
    "t2_tactical_insurance_k",
    "t2_selection_policy",
    "t2_selection_model_weight",
    "t2_selection_structured_top_model_weight",
    "t2_model_top1_rescue_fl_min",
    "t2_model_top1_rescue_bust_max",
    "t2_model_top1_rescue_selected_model_rank_min",
    "t2_model_top1_rescue_refined_delta_max",
    "t2_model_top1_rescue2_fl_min",
    "t2_model_top1_rescue2_fl_max",
    "t2_model_top1_rescue2_bust_max",
    "t2_model_top1_rescue2_selected_model_rank_min",
    "t2_model_top1_rescue2_refined_delta_max",
    "t2_model_top1_bust_rescue_selected_model_rank_min",
    "t2_model_top1_bust_rescue_refined_delta_max",
    "t2_model_top1_bust_rescue_model_delta_min",
    "t2_model_top1_bust_rescue_bust_delta_min",
    "t2_model_top1_bust_rescue_top_bust_max",
    "t2_model_top1_bust_rescue_top_fl_min",
    "t2_model_top1_bust_rescue_current_fl_min",
    "t2_model_top1_bust_rescue_fl_delta_min",
    "t2_middle_fill_bottom_shift_rescue_selected_model_rank_max",
    "t2_middle_fill_bottom_shift_rescue_challenger_model_rank_max",
    "t2_middle_fill_bottom_shift_rescue_model_gap_max",
    "t2_middle_fill_bottom_shift_rescue_bust_delta_min",
    "t2_middle_fill_bottom_shift_rescue_fl_delta_min",
    "t2_middle_fill_bottom_shift_rescue_challenger_bust_max",
    "t2_middle_fill_bottom_shift_rescue_challenger_fl_min",
    "t2_middle_fill_bottom_shift_rescue_selected_bust_min",
    "t2_model_rank_rescue_k",
    "t2_model_rank_rescue_selected_model_rank_min",
    "t2_model_rank_rescue_refined_delta_max",
    "t2_model_rank_rescue_model_delta_min",
    "t2_model_rank_rescue_min_refined_score",
    "t3_pool_k",
)

FLOAT_CONFIG_OVERRIDES = {
    "t2_sync_model_insurance_top1_kk_min",
    "t2_model_top1_rescue_fl_min",
    "t2_model_top1_rescue_bust_max",
    "t2_model_top1_rescue_refined_delta_max",
    "t2_model_top1_rescue2_fl_min",
    "t2_model_top1_rescue2_fl_max",
    "t2_model_top1_rescue2_bust_max",
    "t2_model_top1_rescue2_refined_delta_max",
    "t2_model_top1_bust_rescue_refined_delta_max",
    "t2_model_top1_bust_rescue_model_delta_min",
    "t2_model_top1_bust_rescue_bust_delta_min",
    "t2_model_top1_bust_rescue_top_bust_max",
    "t2_model_top1_bust_rescue_top_fl_min",
    "t2_model_top1_bust_rescue_current_fl_min",
    "t2_model_top1_bust_rescue_fl_delta_min",
    "t2_middle_fill_bottom_shift_rescue_model_gap_max",
    "t2_middle_fill_bottom_shift_rescue_bust_delta_min",
    "t2_middle_fill_bottom_shift_rescue_fl_delta_min",
    "t2_middle_fill_bottom_shift_rescue_challenger_bust_max",
    "t2_middle_fill_bottom_shift_rescue_challenger_fl_min",
    "t2_middle_fill_bottom_shift_rescue_selected_bust_min",
    "t2_model_rank_rescue_refined_delta_max",
    "t2_model_rank_rescue_model_delta_min",
    "t2_model_rank_rescue_min_refined_score",
    "t2_post_refine_extra_model_score_min",
    "t2_selection_model_weight",
    "t2_selection_structured_top_model_weight",
}

STRING_CONFIG_OVERRIDES = {
    "t2_post_refine_policy",
    "t2_selection_policy",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def resolve_path(path: str | Path) -> Path:
    out = Path(path)
    if out.is_absolute():
        return out
    return repo_root() / out


def merged_runtime_models(runtime_payload: dict[str, Any], mode_payload: dict[str, Any]) -> dict[str, Any]:
    models: dict[str, Any] = {}
    base_models = runtime_payload.get("models")
    if isinstance(base_models, dict):
        models.update(base_models)
    mode_models = mode_payload.get("models")
    if isinstance(mode_models, dict):
        models.update(mode_models)
    return models


def iter_jsonl(path: Path, limit: int = 0) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig") as f:
        emitted = 0
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            emitted += 1
            yield line_no, row
            if limit > 0 and emitted >= limit:
                break


def extract_t2_payload(record: dict[str, Any]) -> dict[str, Any] | None:
    """Return a usable T2 position from raw or wrapped benchmark records."""
    if int(record.get("turn", -1)) == 2 and record.get("board") and record.get("dealt"):
        return record
    source = record.get("source_record")
    if isinstance(source, dict) and int(source.get("turn", -1)) == 2:
        return source
    payload = record.get("payload")
    if isinstance(payload, dict) and int(payload.get("turn", -1)) == 2:
        return payload
    return None


def candidate_action(candidate: dict[str, Any]) -> dict[str, Any] | None:
    action = candidate.get("action")
    if isinstance(action, dict):
        return action
    if "placements" in candidate or "discard" in candidate:
        return {
            "placements": candidate.get("placements") or [],
            "discard": candidate.get("discard"),
        }
    return None


def candidate_teacher_score(candidate: dict[str, Any]) -> float | None:
    for key in ("teacher_score", "source_score", "score", "ev", "EV", "avg_score"):
        value = candidate.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    mc = candidate.get("mc")
    if isinstance(mc, dict):
        for key in ("avg_score", "ev", "EV", "score"):
            value = mc.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    metrics = candidate.get("metrics")
    if isinstance(metrics, dict):
        for key in ("score", "ev", "EV"):
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    return None


def teacher_scores_from_record(record: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for candidate in record.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        action = candidate_action(candidate)
        score = candidate_teacher_score(candidate)
        if action is None or score is None:
            continue
        out[action_key(action)] = float(score)
    return out


def best_result_action_key(result: dict[str, Any]) -> str | None:
    best = result.get("best")
    if not isinstance(best, dict):
        return None
    action = best.get("action")
    if isinstance(action, dict):
        return action_key(action)
    return None


def model_top1_action_key(result: dict[str, Any]) -> str | None:
    candidates = result.get("candidates") or []
    for candidate in candidates:
        if isinstance(candidate, dict) and int(candidate.get("model_rank", 999999)) == 1:
            action = candidate.get("action")
            if isinstance(action, dict):
                return action_key(action)
    return None


def compare_teacher(record: dict[str, Any], result: dict[str, Any]) -> dict[str, Any] | None:
    scores = teacher_scores_from_record(record)
    if not scores:
        return None
    values = list(scores.values())
    if len(values) > 1 and max(values) - min(values) <= 1.0e-12:
        return {
            "valid": False,
            "invalid_reason": "degenerate_teacher_scores",
            "teacher_best_action_key": None,
            "teacher_best_score": None,
            "chosen_action_key": best_result_action_key(result),
            "chosen_teacher_score": None,
            "chosen_ev_loss": None,
            "chosen_hit": False,
            "model_top1_action_key": model_top1_action_key(result),
            "model_top1_teacher_score": None,
            "model_top1_ev_loss": None,
            "teacher_candidate_count": len(scores),
            "teacher_score_min": float(min(values)),
            "teacher_score_max": float(max(values)),
        }
    teacher_best_key, teacher_best_score = max(scores.items(), key=lambda item: item[1])
    chosen_key = best_result_action_key(result)
    model_key = model_top1_action_key(result)
    chosen_score = scores.get(chosen_key or "")
    model_score = scores.get(model_key or "")
    chosen_loss = None if chosen_score is None else max(0.0, teacher_best_score - chosen_score)
    model_loss = None if model_score is None else max(0.0, teacher_best_score - model_score)
    return {
        "valid": True,
        "invalid_reason": None,
        "teacher_best_action_key": teacher_best_key,
        "teacher_best_score": float(teacher_best_score),
        "chosen_action_key": chosen_key,
        "chosen_teacher_score": chosen_score,
        "chosen_ev_loss": chosen_loss,
        "chosen_hit": bool(chosen_loss is not None and chosen_loss <= 1.0e-9),
        "model_top1_action_key": model_key,
        "model_top1_teacher_score": model_score,
        "model_top1_ev_loss": model_loss,
        "teacher_candidate_count": len(scores),
    }


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = round((len(ordered) - 1) * pct)
    return ordered[int(max(0, min(len(ordered) - 1, rank)))]


def mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def compact_row(
    *,
    source_line: int,
    position: dict[str, Any],
    result: dict[str, Any],
    teacher: dict[str, Any] | None,
    runtime_mode: str,
    model_path: str,
) -> dict[str, Any]:
    t3_summary = result.get("t3_union_summary") or {}
    best = result.get("best") if isinstance(result.get("best"), dict) else {}
    return {
        "source_line": int(source_line),
        "turn": int(position.get("turn", -1)),
        "position": position.get("position") or ("btn" if bool(position.get("is_btn", False)) else "bb"),
        "runtime_mode": runtime_mode,
        "model_path": model_path,
        "dealt": position.get("dealt"),
        "board": position.get("board"),
        "opponent_board": position.get("opponent_board"),
        "elapsed_ms": float(result.get("elapsed_ms", 0.0) or 0.0),
        "legal_actions": int(result.get("legal_actions", 0) or 0),
        "candidate_pool_size": int(result.get("candidate_pool_size", 0) or 0),
        "sync_refinement_candidate_count": int(result.get("sync_refinement_candidate_count", 0) or 0),
        "exact_evaluated": int(result.get("exact_evaluated", 0) or 0),
        "refinement_error": result.get("refinement_error"),
        "refinement_backend": result.get("t2_exact_backend") or result.get("refinement_backend"),
        "t3_union_samples": int(t3_summary.get("samples", 0) or 0),
        "t3_union_pool_size_mean": t3_summary.get("pool_size_mean"),
        "t3_union_pool_size_max": t3_summary.get("pool_size_max"),
        "t3_union_exact_evaluated_mean": t3_summary.get("exact_evaluated_mean"),
        "t3_union_exact_evaluated_max": t3_summary.get("exact_evaluated_max"),
        "model_top1_action_idx": result.get("model_top1_action_idx"),
        "best_action_idx": result.get("best_action_idx"),
        "model_top1_overridden": bool(result.get("model_top1_overridden", False)),
        "t2_model_top1_rescue_applied": bool(result.get("t2_model_top1_rescue_applied", False)),
        "t2_model_top1_rescue_details": result.get("t2_model_top1_rescue_details"),
        "t2_model_top1_bust_rescue_applied": bool(
            result.get("t2_model_top1_bust_rescue_applied", False)
        ),
        "t2_model_top1_bust_rescue_details": result.get("t2_model_top1_bust_rescue_details"),
        "t2_middle_fill_bottom_shift_rescue_applied": bool(
            result.get("t2_middle_fill_bottom_shift_rescue_applied", False)
        ),
        "t2_middle_fill_bottom_shift_rescue_details": result.get(
            "t2_middle_fill_bottom_shift_rescue_details"
        ),
        "t2_model_rank_rescue_applied": bool(result.get("t2_model_rank_rescue_applied", False)),
        "t2_model_rank_rescue_details": result.get("t2_model_rank_rescue_details"),
        "best_refined_score": result.get("best_refined_score"),
        "best_model_score": best.get("model_score"),
        "best_action": best.get("action"),
        "teacher": teacher,
    }


def summarize(rows: list[dict[str, Any]], *, time_budget_ms: int) -> dict[str, Any]:
    elapsed = [float(row["elapsed_ms"]) for row in rows]
    pool_sizes = [float(row["candidate_pool_size"]) for row in rows]
    t3_pool_means = [
        float(row["t3_union_pool_size_mean"])
        for row in rows
        if isinstance(row.get("t3_union_pool_size_mean"), (int, float))
    ]
    t3_pool_maxes = [
        float(row["t3_union_pool_size_max"])
        for row in rows
        if isinstance(row.get("t3_union_pool_size_max"), (int, float))
    ]
    teacher_dict_rows = [row for row in rows if isinstance(row.get("teacher"), dict)]
    teacher_rows = [row for row in teacher_dict_rows if (row.get("teacher") or {}).get("valid", True)]
    invalid_teacher_rows = [row for row in teacher_dict_rows if not (row.get("teacher") or {}).get("valid", True)]
    chosen_losses = [
        float(row["teacher"]["chosen_ev_loss"])
        for row in teacher_rows
        if isinstance(row["teacher"].get("chosen_ev_loss"), (int, float))
    ]
    model_losses = [
        float(row["teacher"]["model_top1_ev_loss"])
        for row in teacher_rows
        if isinstance(row["teacher"].get("model_top1_ev_loss"), (int, float))
    ]
    errors = [row for row in rows if row.get("refinement_error")]
    return {
        "count": len(rows),
        "time_budget_ms": int(time_budget_ms),
        "latency_ms": {
            "mean": mean(elapsed),
            "p50": percentile(elapsed, 0.50),
            "p95": percentile(elapsed, 0.95),
            "max": max(elapsed) if elapsed else None,
            "under_budget_count": sum(1 for value in elapsed if value <= time_budget_ms),
            "under_budget_rate": (
                sum(1 for value in elapsed if value <= time_budget_ms) / len(elapsed) if elapsed else None
            ),
        },
        "candidate_pool_size": {
            "mean": mean(pool_sizes),
            "p95": percentile(pool_sizes, 0.95),
            "max": max(pool_sizes) if pool_sizes else None,
        },
        "t3_union_pool_size": {
            "mean_of_means": mean(t3_pool_means),
            "max": max(t3_pool_maxes) if t3_pool_maxes else None,
        },
        "model_top1_overridden_count": sum(1 for row in rows if row.get("model_top1_overridden")),
        "model_top1_overridden_rate": (
            sum(1 for row in rows if row.get("model_top1_overridden")) / len(rows) if rows else None
        ),
        "t2_model_top1_rescue": {
            "details_count": sum(1 for row in rows if isinstance(row.get("t2_model_top1_rescue_details"), dict)),
            "applied_count": sum(1 for row in rows if row.get("t2_model_top1_rescue_applied")),
            "applied_rate": (
                sum(1 for row in rows if row.get("t2_model_top1_rescue_applied")) / len(rows)
                if rows
                else None
            ),
        },
        "t2_model_top1_bust_rescue": {
            "details_count": sum(
                1 for row in rows if isinstance(row.get("t2_model_top1_bust_rescue_details"), dict)
            ),
            "applied_count": sum(1 for row in rows if row.get("t2_model_top1_bust_rescue_applied")),
            "applied_rate": (
                sum(1 for row in rows if row.get("t2_model_top1_bust_rescue_applied")) / len(rows)
                if rows
                else None
            ),
        },
        "t2_middle_fill_bottom_shift_rescue": {
            "details_count": sum(
                1
                for row in rows
                if isinstance(row.get("t2_middle_fill_bottom_shift_rescue_details"), dict)
            ),
            "applied_count": sum(
                1 for row in rows if row.get("t2_middle_fill_bottom_shift_rescue_applied")
            ),
            "applied_rate": (
                sum(1 for row in rows if row.get("t2_middle_fill_bottom_shift_rescue_applied")) / len(rows)
                if rows
                else None
            ),
        },
        "t2_model_rank_rescue": {
            "details_count": sum(1 for row in rows if isinstance(row.get("t2_model_rank_rescue_details"), dict)),
            "applied_count": sum(1 for row in rows if row.get("t2_model_rank_rescue_applied")),
            "applied_rate": (
                sum(1 for row in rows if row.get("t2_model_rank_rescue_applied")) / len(rows)
                if rows
                else None
            ),
        },
        "teacher_compared_count": len(teacher_rows),
        "teacher_invalid_count": len(invalid_teacher_rows),
        "teacher_invalid_reasons": {
            reason: sum(
                1
                for row in invalid_teacher_rows
                if (row.get("teacher") or {}).get("invalid_reason") == reason
            )
            for reason in sorted(
                {
                    str((row.get("teacher") or {}).get("invalid_reason") or "unknown")
                    for row in invalid_teacher_rows
                }
            )
        },
        "chosen_vs_teacher": {
            "hit_count": sum(1 for row in teacher_rows if (row.get("teacher") or {}).get("chosen_hit")),
            "hit_rate": (
                sum(1 for row in teacher_rows if (row.get("teacher") or {}).get("chosen_hit")) / len(teacher_rows)
                if teacher_rows
                else None
            ),
            "ev_loss_mean": mean(chosen_losses),
            "ev_loss_p95": percentile(chosen_losses, 0.95),
            "ev_loss_max": max(chosen_losses) if chosen_losses else None,
            "loss_ge_0_1": sum(1 for value in chosen_losses if value >= 0.1),
            "loss_ge_0_5": sum(1 for value in chosen_losses if value >= 0.5),
            "loss_ge_1_0": sum(1 for value in chosen_losses if value >= 1.0),
        },
        "model_top1_vs_teacher": {
            "ev_loss_mean": mean(model_losses),
            "ev_loss_p95": percentile(model_losses, 0.95),
            "ev_loss_max": max(model_losses) if model_losses else None,
        },
        "errors": {
            "count": len(errors),
            "examples": [
                {
                    "source_line": row.get("source_line"),
                    "refinement_error": row.get("refinement_error"),
                }
                for row in errors[:5]
            ],
        },
    }


def summary_markdown(summary: dict[str, Any], *, input_path: Path, output_dir: Path, mode_name: str) -> str:
    latency = summary["latency_ms"]
    chosen = summary["chosen_vs_teacher"]
    pool = summary["candidate_pool_size"]
    t3 = summary["t3_union_pool_size"]
    t2_rescue = summary.get("t2_model_top1_rescue") or {}
    t2_bust_rescue = summary.get("t2_model_top1_bust_rescue") or {}
    t2_shift_rescue = summary.get("t2_middle_fill_bottom_shift_rescue") or {}
    t2_rank_rescue = summary.get("t2_model_rank_rescue") or {}
    lines = [
        "# T2 T3-union runtime benchmark",
        "",
        f"- input: `{input_path}`",
        f"- output_dir: `{output_dir}`",
        f"- runtime_mode: `{mode_name}`",
        f"- count: {summary['count']}",
        f"- latency_ms mean/p50/p95/max: {latency['mean']:.1f} / {latency['p50']:.1f} / {latency['p95']:.1f} / {latency['max']:.1f}" if latency["mean"] is not None else "- latency_ms: n/a",
        f"- under_budget: {latency['under_budget_count']}/{summary['count']} ({latency['under_budget_rate']:.1%})" if latency["under_budget_rate"] is not None else "- under_budget: n/a",
        f"- candidate_pool mean/p95/max: {pool['mean']:.1f} / {pool['p95']:.1f} / {pool['max']:.0f}" if pool["mean"] is not None else "- candidate_pool: n/a",
        f"- T3 union pool mean_of_means/max: {t3['mean_of_means']:.1f} / {t3['max']:.0f}" if t3["mean_of_means"] is not None else "- T3 union pool: n/a",
        f"- model_top1_overridden: {summary['model_top1_overridden_count']}/{summary['count']} ({summary['model_top1_overridden_rate']:.1%})" if summary["model_top1_overridden_rate"] is not None else "- model_top1_overridden: n/a",
        f"- t2_model_top1_rescue applied/details: {t2_rescue.get('applied_count', 0)}/{t2_rescue.get('details_count', 0)}",
        f"- t2_model_top1_bust_rescue applied/details: {t2_bust_rescue.get('applied_count', 0)}/{t2_bust_rescue.get('details_count', 0)}",
        f"- t2_middle_fill_bottom_shift_rescue applied/details: {t2_shift_rescue.get('applied_count', 0)}/{t2_shift_rescue.get('details_count', 0)}",
        f"- t2_model_rank_rescue applied/details: {t2_rank_rescue.get('applied_count', 0)}/{t2_rank_rescue.get('details_count', 0)}",
        f"- teacher_compared: {summary['teacher_compared_count']}",
        f"- teacher_invalid: {summary.get('teacher_invalid_count', 0)} {summary.get('teacher_invalid_reasons', {})}",
        f"- chosen teacher hit/loss mean/p95/max: {chosen['hit_count']}/{summary['teacher_compared_count']} / {chosen['ev_loss_mean']:.4f} / {chosen['ev_loss_p95']:.4f} / {chosen['ev_loss_max']:.4f}" if chosen["ev_loss_mean"] is not None else "- chosen teacher comparison: n/a",
        f"- chosen loss counts >=0.1/>=0.5/>=1.0: {chosen['loss_ge_0_1']} / {chosen['loss_ge_0_5']} / {chosen['loss_ge_1_0']}",
        f"- errors: {summary['errors']['count']}",
        "",
    ]
    return "\n".join(lines)


def build_config(mode_payload: dict[str, Any], args: argparse.Namespace) -> HybridConfig:
    defaults = HybridConfig()
    values: dict[str, Any] = {}
    for field in fields(HybridConfig):
        values[field.name] = mode_payload.get(field.name, getattr(defaults, field.name))
    values["t2_refinement"] = mode_payload.get("t2_refinement", "exact_partial")
    values["t2_exact_backend"] = mode_payload.get("t2_exact_backend", "t3_union")
    values["enable_sync_refinement"] = True
    for name in CONFIG_OVERRIDES:
        value = getattr(args, name, None)
        if value is not None:
            values[name] = value
    if args.t3_pool_config:
        values["t3_pool_config"] = args.t3_pool_config
    if args.t2_sync_selection_policy:
        values["t2_sync_selection_policy"] = args.t2_sync_selection_policy
    if args.t2_sync_selector:
        values["t2_sync_selector"] = load_t2_sync_selector(args.t2_sync_selector)
        if not args.t2_sync_selection_policy:
            values["t2_sync_selection_policy"] = "selector"
    if args.t2_final_selector:
        values["t2_final_selector"] = load_t2_sync_selector(args.t2_final_selector)
        values["t2_selection_policy"] = "selector"
    return HybridConfig(**values)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    input_path = resolve_path(args.input)
    runtime_config_path = resolve_path(args.runtime_config)
    runtime_payload = json.loads(runtime_config_path.read_text(encoding="utf-8-sig"))
    mode_name, mode_payload = _runtime_config_mode_payload(runtime_payload, args.runtime_mode)
    models_payload = merged_runtime_models(runtime_payload, mode_payload)
    model_path = args.model or models_payload.get("action_value")
    if not model_path:
        raise ValueError("No action-value model configured; pass --model or set models.action_value")
    turn_model_ensemble_specs = _turn_model_ensemble_defaults(
        models_payload.get("turn_model_ensembles") or models_payload.get("action_value_ensembles_by_turn")
    )
    turn_model_conditional_ensemble_specs = _turn_model_conditional_ensemble_defaults(
        models_payload.get("turn_model_conditional_ensembles")
        or models_payload.get("conditional_action_value_ensembles_by_turn")
    )
    turn_model_conditional_switch_ensemble_specs = _turn_model_conditional_switch_ensemble_defaults(
        models_payload.get("turn_model_conditional_switch_ensembles")
        or models_payload.get("conditional_switch_action_value_ensembles_by_turn")
    )
    turn_model_cascade_switch_ensemble_specs = _turn_model_cascade_switch_ensemble_defaults(
        models_payload.get("turn_model_cascade_switch_ensembles")
        or models_payload.get("cascade_switch_action_value_ensembles_by_turn")
    )
    config = build_config(mode_payload, args)
    t2_final_selector_path = models_payload.get("t2_final_selector") if isinstance(models_payload, dict) else None
    if t2_final_selector_path and config.t2_final_selector is None:
        config = replace(config, t2_final_selector=load_t2_sync_selector(t2_final_selector_path))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    compact_path = output_dir / "rows.jsonl"
    misses_path = output_dir / "teacher_misses.jsonl"
    summary_path = output_dir / "summary.json"
    markdown_path = output_dir / "summary.md"

    evaluator = make_action_value_evaluator(
        model_path=model_path,
        model_ensembles_by_turn=parse_turn_model_ensemble_specs(turn_model_ensemble_specs),
        model_conditional_ensembles_by_turn=parse_turn_model_conditional_ensemble_specs(
            turn_model_conditional_ensemble_specs
        ),
        model_conditional_switch_ensembles_by_turn=parse_turn_model_conditional_switch_ensemble_specs(
            turn_model_conditional_switch_ensemble_specs
        ),
        model_cascade_switch_ensembles_by_turn=parse_turn_model_cascade_switch_ensemble_specs(
            turn_model_cascade_switch_ensemble_specs
        ),
        device=args.device,
    )
    t3_config_path = resolve_path(config.t3_pool_config)
    t3_pooler = T3UnionCandidatePool(config_path=t3_config_path, device=args.t3_device)
    rust_solver = resolve_path(args.rust_solver) if args.rust_solver else None

    rows: list[dict[str, Any]] = []
    processed = 0
    started = time.perf_counter()
    with results_path.open("w", encoding="utf-8") as full_out, compact_path.open(
        "w", encoding="utf-8"
    ) as compact_out, misses_path.open("w", encoding="utf-8") as miss_out:
        for source_line, record in iter_jsonl(input_path, limit=0):
            payload = extract_t2_payload(record)
            if payload is None:
                continue
            if args.position and str(payload.get("position") or "").lower() != args.position.lower():
                continue
            if args.limit > 0 and processed >= args.limit:
                break
            processed += 1
            result = evaluate_hybrid_position(
                payload,
                evaluator=evaluator,
                config=config,
                rust_solver_path=rust_solver,
                t3_pooler=t3_pooler,
            )
            teacher = compare_teacher(payload, result)
            compact = compact_row(
                source_line=source_line,
                position=payload,
                result=result,
                teacher=teacher,
                runtime_mode=mode_name,
                model_path=str(model_path),
            )
            rows.append(compact)
            full_out.write(
                json.dumps(
                    {
                        "source_line": source_line,
                        "payload": payload,
                        "result": result,
                        "teacher": teacher,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            compact_out.write(json.dumps(compact, ensure_ascii=False) + "\n")
            if teacher and isinstance(teacher.get("chosen_ev_loss"), (int, float)) and teacher["chosen_ev_loss"] > 0.0:
                miss_out.write(json.dumps(compact, ensure_ascii=False) + "\n")
            if args.progress_every > 0 and processed % args.progress_every == 0:
                elapsed = time.perf_counter() - started
                print(f"processed={processed} elapsed_s={elapsed:.1f} last_ms={compact['elapsed_ms']:.1f}", flush=True)

    summary = summarize(rows, time_budget_ms=int(config.time_budget_ms))
    summary.update(
        {
            "input": str(input_path),
            "runtime_config": str(runtime_config_path),
            "runtime_mode": mode_name,
            "model_path": str(model_path),
            "turn_model_ensembles": len(turn_model_ensemble_specs),
        "turn_model_conditional_ensembles": len(turn_model_conditional_ensemble_specs),
        "turn_model_conditional_switch_ensembles": len(turn_model_conditional_switch_ensemble_specs),
        "turn_model_cascade_switch_ensembles": len(turn_model_cascade_switch_ensemble_specs),
        "t3_pool_config": str(t3_config_path),
            "rust_solver": str(rust_solver) if rust_solver else None,
            "output_dir": str(output_dir),
            "wall_elapsed_s": time.perf_counter() - started,
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(
        summary_markdown(summary, input_path=input_path, output_dir=output_dir, mode_name=mode_name),
        encoding="utf-8",
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input JSONL with T2 positions.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for benchmark outputs.")
    parser.add_argument("--runtime-config", default=str(DEFAULT_RUNTIME_CONFIG), help="Hybrid runtime config JSON.")
    parser.add_argument("--runtime-mode", default="", help="Runtime mode in the config; empty uses default_mode.")
    parser.add_argument("--model", default="", help="Override action-value model checkpoint.")
    parser.add_argument("--device", default="cpu", help="Device for the T2 action-value model.")
    parser.add_argument("--t3-device", default="auto", help="Device for T3 union candidate-source models.")
    parser.add_argument(
        "--rust-solver",
        default=str(Path("ai/rust_solver/target/release/t3_exact_solver.exe")),
        help="Rust T3 exact solver path.",
    )
    parser.add_argument("--limit", type=int, default=100, help="Number of T2 positions to benchmark.")
    parser.add_argument("--position", default="", choices=("", "bb", "btn"), help="Optional position filter.")
    parser.add_argument("--progress-every", type=int, default=10, help="Print progress every N positions.")
    for name in CONFIG_OVERRIDES:
        if name in FLOAT_CONFIG_OVERRIDES:
            value_type = float
        elif name in STRING_CONFIG_OVERRIDES:
            value_type = str
        else:
            value_type = int
        parser.add_argument(f"--{name.replace('_', '-')}", type=value_type, default=None)
    parser.add_argument("--t3-pool-config", default="", help="Override T3 union pool config.")
    parser.add_argument(
        "--t2-sync-selection-policy",
        default="",
        choices=("", "rank", "selector"),
        help="Override T2 sync refinement candidate ordering.",
    )
    parser.add_argument("--t2-sync-selector", default="", help="Path to a T2 sync selector JSON.")
    parser.add_argument("--t2-final-selector", default="", help="Path to a T2 final selector JSON.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = run_benchmark(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
