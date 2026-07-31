"""Evaluate T1/T2 hybrid refinement against JSONL teacher records.

The teacher JSONL format is produced by active teacher generation and stores
the original position plus per-action MC/exact labels.  This script runs the
runtime hybrid evaluator, then compares both raw model Top1 and post-refinement
Top1 against the teacher best action.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.tutor.hybrid_t1t2 import (
    HybridConfig,
    _provided_cli_dests,
    apply_hybrid_runtime_config,
    evaluate_hybrid_position,
    load_t1_final_selector,
    load_t1_override_gate,
    load_t1_runtime_gate,
    load_t1_sync_selector,
    make_action_value_evaluator,
    parse_turn_model_blend_specs,
    parse_turn_model_specs,
)


ROWS = {"top": "top", "middle": "middle", "mid": "middle", "bottom": "bottom", "bot": "bottom"}


def _norm_card(card: str | None) -> str:
    if not card:
        return ""
    card = str(card)
    if card in {"Xj", "JK"} or card.startswith("X"):
        return "X"
    return card


def _norm_row(row: str | None) -> str:
    return ROWS.get(str(row), str(row))


def canonical_action(action: dict[str, Any] | None) -> tuple[tuple[tuple[str, str], ...], str]:
    action = action or {}
    placements = tuple(
        sorted(
            (
                _norm_card(card),
                _norm_row(row),
            )
            for card, row in (action.get("placements") or [])
        )
    )
    return placements, _norm_card(action.get("discard"))


def candidate_score(candidate: dict[str, Any]) -> float:
    metrics = candidate.get("metrics") or {}
    if "score" in metrics:
        return float(metrics["score"])
    mc = candidate.get("mc") or {}
    if "avg_score" in mc:
        return float(mc["avg_score"])
    if "score" in candidate:
        return float(candidate["score"])
    return float("-inf")


def candidate_samples(candidate: dict[str, Any]) -> int:
    metrics = candidate.get("metrics") or {}
    if "samples" in metrics:
        return int(metrics["samples"] or 0)
    mc = candidate.get("mc") or {}
    if "simulations" in mc:
        return int(mc["simulations"] or 0)
    return 0


def candidate_action(candidate: dict[str, Any]) -> dict[str, Any]:
    if "action" in candidate and isinstance(candidate["action"], dict):
        return candidate["action"]
    return {
        "placements": candidate.get("placements") or [],
        "discard": candidate.get("discard"),
    }


def teacher_index(record: dict[str, Any]) -> dict[str, Any]:
    candidates = list(record.get("candidates") or [])
    if not candidates:
        return {
            "best_action": None,
            "best_score": float("nan"),
            "second_score": float("nan"),
            "margin": float("nan"),
            "score_by_action": {},
            "samples": 0,
        }
    scored = [(candidate_score(candidate), idx, candidate) for idx, candidate in enumerate(candidates)]
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, _best_idx, best_candidate = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else float("-inf")
    score_by_action = {
        canonical_action(candidate_action(candidate)): candidate_score(candidate)
        for candidate in candidates
    }
    rank_by_action = {
        canonical_action(candidate_action(candidate)): rank
        for rank, (_score, _idx, candidate) in enumerate(scored, start=1)
    }
    return {
        "best_action": canonical_action(candidate_action(best_candidate)),
        "best_score": float(best_score),
        "second_score": float(second_score),
        "margin": float(best_score - second_score),
        "score_by_action": score_by_action,
        "rank_by_action": rank_by_action,
        "samples": max(candidate_samples(candidate) for candidate in candidates),
    }


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = int((len(values) * p + 0.999999) - 1)
    return values[max(0, min(idx, len(values) - 1))]


class Stats:
    def __init__(self) -> None:
        self.decisions = 0
        self.model_hits = 0
        self.final_hits = 0
        self.model_zero_regret_hits = 0
        self.final_zero_regret_hits = 0
        self.model_regret_sum = 0.0
        self.final_regret_sum = 0.0
        self.model_regret_max = 0.0
        self.final_regret_max = 0.0
        self.unmatched_model = 0
        self.unmatched_final = 0
        self.pool_hits = 0
        self.sync_hits = 0
        self.pool_zero_regret_hits = 0
        self.sync_zero_regret_hits = 0
        self.refined_positions = 0
        self.exact_evaluated = 0
        self.overrides = 0
        self.override_improved = 0
        self.override_worse = 0
        self.override_same = 0
        self.time_budget_violations = 0
        self.elapsed_ms: list[float] = []

    def add(
        self,
        *,
        teacher: dict[str, Any],
        result: dict[str, Any],
        model_action: tuple[tuple[tuple[str, str], ...], str] | None,
        final_action: tuple[tuple[tuple[str, str], ...], str] | None,
    ) -> None:
        self.decisions += 1
        best_action = teacher["best_action"]
        best_score = float(teacher["best_score"])
        scores: dict[Any, float] = teacher["score_by_action"]

        model_score = scores.get(model_action)
        final_score = scores.get(final_action)
        if model_score is None:
            self.unmatched_model += 1
            model_regret = 0.0
        else:
            model_regret = max(0.0, best_score - float(model_score))
        if final_score is None:
            self.unmatched_final += 1
            final_regret = 0.0
        else:
            final_regret = max(0.0, best_score - float(final_score))

        if model_action == best_action:
            self.model_hits += 1
        if final_action == best_action:
            self.final_hits += 1
        if model_score is not None and model_regret <= 1e-9:
            self.model_zero_regret_hits += 1
        if final_score is not None and final_regret <= 1e-9:
            self.final_zero_regret_hits += 1
        pool_actions = {
            canonical_action(item.get("action"))
            for item in (result.get("candidates") or [])
            if item.get("action") is not None
        }
        sync_action_indices = {int(idx) for idx in (result.get("sync_exact_action_indices") or [])}
        sync_actions = {
            canonical_action(item.get("action"))
            for item in (result.get("candidates") or [])
            if item.get("action") is not None and int(item.get("action_idx", -1)) in sync_action_indices
        }
        if best_action in pool_actions:
            self.pool_hits += 1
        if best_action in sync_actions:
            self.sync_hits += 1
        pool_scores = [float(scores[action]) for action in pool_actions if action in scores]
        sync_scores = [float(scores[action]) for action in sync_actions if action in scores]
        if pool_scores and max(0.0, best_score - max(pool_scores)) <= 1e-9:
            self.pool_zero_regret_hits += 1
        if sync_scores and max(0.0, best_score - max(sync_scores)) <= 1e-9:
            self.sync_zero_regret_hits += 1

        self.model_regret_sum += model_regret
        self.final_regret_sum += final_regret
        self.model_regret_max = max(self.model_regret_max, model_regret)
        self.final_regret_max = max(self.final_regret_max, final_regret)

        exact_evaluated = int(result.get("exact_evaluated", 0) or 0)
        if exact_evaluated > 0:
            self.refined_positions += 1
            self.exact_evaluated += exact_evaluated
        if bool(result.get("model_top1_overridden")):
            self.overrides += 1
            if final_regret < model_regret:
                self.override_improved += 1
            elif final_regret > model_regret:
                self.override_worse += 1
            else:
                self.override_same += 1
        elapsed = float(result.get("elapsed_ms", 0.0) or 0.0)
        self.elapsed_ms.append(elapsed)
        if elapsed > float(result.get("time_budget_ms", 0) or 0) + 50.0:
            self.time_budget_violations += 1

    def as_dict(self) -> dict[str, Any]:
        denom = max(self.decisions, 1)
        return {
            "decisions": self.decisions,
            "model_top1_recall": self.model_hits / denom,
            "final_top1_recall": self.final_hits / denom,
            "model_zero_regret_recall": self.model_zero_regret_hits / denom,
            "final_zero_regret_recall": self.final_zero_regret_hits / denom,
            "top1_recall_delta": (self.final_hits - self.model_hits) / denom,
            "model_avg_regret": self.model_regret_sum / denom,
            "final_avg_regret": self.final_regret_sum / denom,
            "avg_regret_delta": (self.final_regret_sum - self.model_regret_sum) / denom,
            "model_max_regret": self.model_regret_max,
            "final_max_regret": self.final_regret_max,
            "unmatched_model": self.unmatched_model,
            "unmatched_final": self.unmatched_final,
            "pool_recall": self.pool_hits / denom,
            "sync_recall": self.sync_hits / denom,
            "pool_zero_regret_recall": self.pool_zero_regret_hits / denom,
            "sync_zero_regret_recall": self.sync_zero_regret_hits / denom,
            "refined_positions": self.refined_positions,
            "exact_evaluated": self.exact_evaluated,
            "avg_exact_evaluated": self.exact_evaluated / denom,
            "overrides": self.overrides,
            "override_improved": self.override_improved,
            "override_worse": self.override_worse,
            "override_same": self.override_same,
            "elapsed_ms": {
                "avg": statistics.fmean(self.elapsed_ms) if self.elapsed_ms else 0.0,
                "p50": percentile(self.elapsed_ms, 0.50),
                "p95": percentile(self.elapsed_ms, 0.95),
                "max": max(self.elapsed_ms) if self.elapsed_ms else 0.0,
            },
            "time_budget_violations": self.time_budget_violations,
        }


def _iter_jsonl(path: Path, limit: int = 0, skip: int = 0) -> Iterable[tuple[int, dict[str, Any]]]:
    yielded = 0
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            if line_no <= skip:
                continue
            if line.strip():
                if limit > 0 and yielded >= limit:
                    break
                yielded += 1
                yield line_no, json.loads(line)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Hybrid Refinement Teacher Evaluation",
        "",
        f"- input: `{report['input']}`",
        f"- model: `{report['model']}`",
        f"- time budget: {report['config']['time_budget_ms']} ms",
        "",
        "| scope | decisions | model top1 | final top1 | final zero-regret | pool | sync | sync zero-regret | delta | model regret | final regret | p95 ms | overrides | improved | worse |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def row(name: str, stats: dict[str, Any]) -> str:
        return (
            f"| {name} | {stats['decisions']} | {stats['model_top1_recall']:.1%} | "
            f"{stats['final_top1_recall']:.1%} | {stats['final_zero_regret_recall']:.1%} | "
            f"{stats['pool_recall']:.1%} | {stats['sync_recall']:.1%} | "
            f"{stats['sync_zero_regret_recall']:.1%} | {stats['top1_recall_delta']:+.1%} | "
            f"{stats['model_avg_regret']:.3f} | {stats['final_avg_regret']:.3f} | "
            f"{stats['elapsed_ms']['p95']:.1f} | {stats['overrides']} | "
            f"{stats['override_improved']} | {stats['override_worse']} |"
        )

    lines.append(row("overall", report["overall"]))
    for turn, stats in sorted(report["by_turn"].items(), key=lambda item: int(item[0])):
        lines.append(row(f"T{turn}", stats))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def selector_rows_for_result(line_no: int, result: dict[str, Any], teacher: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = list(result.get("candidates") or [])
    sync_action_indices = {int(idx) for idx in (result.get("sync_exact_action_indices") or [])}
    refined_sorted = sorted(
        [item for item in candidates if item.get("refined_score") is not None],
        key=lambda item: (
            float(item.get("refined_score", float("-inf"))),
            float(item.get("model_score", float("-inf"))),
            -int(item.get("model_rank", 999999)),
        ),
        reverse=True,
    )
    refined_rank_by_idx = {
        int(item.get("action_idx", -1)): rank for rank, item in enumerate(refined_sorted, start=1)
    }
    rows: list[dict[str, Any]] = []
    for output_rank, item in enumerate(candidates, start=1):
        action = canonical_action(item.get("action"))
        action_idx = int(item.get("action_idx", -1))
        teacher_score = teacher["score_by_action"].get(action)
        teacher_rank = teacher.get("rank_by_action", {}).get(action)
        predicted_types = item.get("predicted_fl_types") or {}
        rows.append(
            {
                "line": int(line_no),
                "turn": int(result.get("turn", -1)),
                "action_idx": action_idx,
                "action": item.get("action"),
                "board": item.get("board"),
                "opponent_board": item.get("opponent_board"),
                "known_discards": item.get("known_discards"),
                "output_rank": output_rank,
                "model_rank": item.get("model_rank"),
                "refinement_rank": item.get("refinement_rank", item.get("model_rank")),
                "aux_model_rank": item.get("aux_model_rank"),
                "refined_rank": refined_rank_by_idx.get(action_idx),
                "is_sync": action_idx in sync_action_indices,
                "is_refined": item.get("refined_score") is not None,
                "is_model_top1": action_idx == result.get("model_top1_action_idx"),
                "is_runtime_best": action_idx == result.get("best_action_idx"),
                "is_teacher_best": action == teacher["best_action"],
                "teacher_score": teacher_score,
                "teacher_rank": teacher_rank,
                "teacher_best_score": teacher["best_score"],
                "teacher_margin": teacher["margin"],
                "teacher_samples": teacher["samples"],
                "model_score": item.get("model_score"),
                "model_raw_score": item.get("model_raw_score"),
                "aux_model_score": item.get("aux_model_score"),
                "aux_model_raw_score": item.get("aux_model_raw_score"),
                "sync_selector_score": item.get("sync_selector_score"),
                "final_selector_score": item.get("final_selector_score"),
                "arbitration_selector_score": item.get("arbitration_selector_score"),
                "refined_score": item.get("refined_score"),
                "predicted_bust": item.get("predicted_bust"),
                "predicted_fl": item.get("predicted_fl"),
                "predicted_qq": predicted_types.get("qq"),
                "predicted_kk": predicted_types.get("kk"),
                "predicted_aa": predicted_types.get("aa"),
                "predicted_trips": predicted_types.get("trips"),
                "refinement_source": item.get("refinement_source"),
                "refinement_skip_reason": item.get("refinement_skip_reason"),
                "insurance_reason": item.get("insurance_reason"),
                "samples": item.get("samples"),
                "elapsed_ms": item.get("elapsed_ms"),
                "runtime_best_action_idx": result.get("best_action_idx"),
                "model_top1_action_idx": result.get("model_top1_action_idx"),
                "exact_evaluated": result.get("exact_evaluated"),
                "position_elapsed_ms": result.get("elapsed_ms"),
            }
        )
    return rows


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = HybridConfig(
        shortlist_k=args.shortlist_k,
        insurance_k=args.insurance_k,
        sync_exact_k=args.sync_exact_k,
        t2_sync_exact_k=args.t2_sync_exact_k,
        t1_adaptive_sync_exact_k=args.t1_adaptive_sync_exact_k,
        t1_adaptive_model_rank_k=args.t1_adaptive_model_rank_k,
        t1_sync_model_insurance_k=args.t1_sync_model_insurance_k,
        t1_sync_tactical_insurance_k=args.t1_sync_tactical_insurance_k,
        t1_aux_shortlist_k=args.t1_aux_shortlist_k,
        max_pool=args.max_pool,
        time_budget_ms=args.time_budget_ms,
        enable_sync_refinement=not args.disable_refinement,
        t1_refinement=args.t1_refinement,
        t1_mc_sims=args.t1_mc_sims,
        t1_recursive_beam=args.t1_recursive_beam,
        t1_recursive_child_sims=args.t1_recursive_child_sims,
        t1_refinement_first_batch_k=args.t1_refinement_first_batch_k,
        t1_refinement_tail_min_remaining_ms=args.t1_refinement_tail_min_remaining_ms,
        t1_refinement_timeout_headroom_ms=args.t1_refinement_timeout_headroom_ms,
        t1_extra_refine_top_k=args.t1_extra_refine_top_k,
        t1_extra_refine_margin=args.t1_extra_refine_margin,
        t1_extra_refine_sims=args.t1_extra_refine_sims,
        t1_sync_selection_policy=args.t1_sync_selection_policy,
        t1_sync_selector=load_t1_sync_selector(args.t1_sync_selector),
        t1_selection_policy=args.t1_selection_policy,
        t1_selection_model_weight=args.t1_selection_model_weight,
        t1_final_selector=load_t1_final_selector(args.t1_final_selector),
        t1_arbitration_selector=load_t1_final_selector(args.t1_arbitration_selector),
        t1_arbitration_challenger_selector=load_t1_final_selector(args.t1_arbitration_challenger_selector),
        t1_arbitration_policy=args.t1_arbitration_policy,
        t1_arbitration_dual_gate_policy=args.t1_arbitration_dual_gate_policy,
        t1_final_challenger_selector=load_t1_final_selector(args.t1_final_challenger_selector),
        t1_final_challenger_gate=load_t1_runtime_gate(args.t1_final_challenger_gate),
        t1_final_challenger_gate_policy=args.t1_final_challenger_gate_policy,
        t1_refined_challenger_bust_max=args.t1_refined_challenger_bust_max,
        t1_refined_challenger_refined_delta_max=args.t1_refined_challenger_refined_delta_max,
        t1_blend_challenger_weight=args.t1_blend_challenger_weight,
        t1_blend_challenger_bust_min=args.t1_blend_challenger_bust_min,
        t1_blend_challenger_premium_fl_delta_min=args.t1_blend_challenger_premium_fl_delta_min,
        t1_low_risk_blend_challenger_weight=args.t1_low_risk_blend_challenger_weight,
        t1_low_risk_blend_challenger_fl_max=args.t1_low_risk_blend_challenger_fl_max,
        t1_low_risk_blend_challenger_bust_max=args.t1_low_risk_blend_challenger_bust_max,
        t1_model_rescue_qq_delta_min=args.t1_model_rescue_qq_delta_min,
        t1_model_rescue_premium_delta_min=args.t1_model_rescue_premium_delta_min,
        t1_model_rescue_candidate_fl_delta_min=args.t1_model_rescue_candidate_fl_delta_min,
        t1_model_rescue_model_score_min=args.t1_model_rescue_model_score_min,
        t1_model_rescue_model_bust_max=args.t1_model_rescue_model_bust_max,
        t1_model_rescue_refined_delta_max=args.t1_model_rescue_refined_delta_max,
        t1_final_bottom_sparse_rescue_margin=args.t1_final_bottom_sparse_rescue_margin,
        t1_no_refine_fallback_policy=args.t1_no_refine_fallback_policy,
        t1_override_margin=args.t1_override_margin,
        t1_override_gate=load_t1_override_gate(args.t1_override_gate),
        t1_override_gate_threshold=args.t1_override_gate_threshold,
        t2_extra_margin=args.t2_extra_margin,
        t2_baseline_min_samples=args.t2_baseline_min_samples,
        t2_adaptive_shortlist_k=args.t2_adaptive_shortlist_k,
        t2_adaptive_sync_exact_k=args.t2_adaptive_sync_exact_k,
        t2_adaptive_max_model_score=args.t2_adaptive_max_model_score,
        t2_refinement=args.t2_refinement,
        t2_mc_sims=args.t2_mc_sims,
        t2_initial_samples_per_candidate=args.t2_initial_samples_per_candidate,
        t2_extra_samples_per_round=args.t2_extra_samples_per_round,
        t2_max_samples_per_candidate=args.t2_max_samples_per_candidate,
        t2_close_candidate_limit=args.t2_close_candidate_limit,
    )
    turn_models = parse_turn_model_specs(args.turn_model)
    turn_model_blends = parse_turn_model_blend_specs(args.turn_model_blend)
    evaluator = make_action_value_evaluator(
        model_path=args.model,
        model_paths_by_turn=turn_models,
        model_blends_by_turn=turn_model_blends,
        device=args.device or None,
    )
    shortlist_evaluator = (
        make_action_value_evaluator(model_path=args.t1_aux_shortlist_model, device=args.device or None)
        if args.t1_aux_shortlist_model and args.t1_aux_shortlist_k > 0
        else None
    )
    turns = {int(part) for part in args.turns.split(",") if part.strip()}

    overall = Stats()
    by_turn: dict[int, Stats] = defaultdict(Stats)
    teacher_modes = Counter()
    misses_path = output_dir / "misses.jsonl"
    results_path = output_dir / "results.jsonl"
    selector_rows_path = output_dir / "selector_rows.jsonl"

    with misses_path.open("w", encoding="utf-8") as misses_f, results_path.open("w", encoding="utf-8") as results_f:
        selector_f = selector_rows_path.open("w", encoding="utf-8") if args.write_selector_rows else None
        for line_no, record in _iter_jsonl(Path(args.input), args.limit, args.skip):
            turn = int(record.get("turn", -1))
            if turn not in turns:
                continue
            teacher = teacher_index(record)
            if teacher["best_action"] is None:
                continue

            result = evaluate_hybrid_position(
                record,
                evaluator=evaluator,
                shortlist_evaluator=shortlist_evaluator,
                config=config,
                rust_solver_path=args.rust_solver or None,
            )
            model_candidate = next(
                (
                    item
                    for item in result.get("candidates", [])
                    if int(item.get("action_idx", -1)) == int(result.get("model_top1_action_idx", -2))
                ),
                None,
            )
            final_candidate = result.get("best")
            model_action = canonical_action(model_candidate.get("action") if model_candidate else None)
            final_action = canonical_action(final_candidate.get("action") if final_candidate else None)
            pool_actions = {
                canonical_action(item.get("action"))
                for item in (result.get("candidates") or [])
                if item.get("action") is not None
            }
            sync_action_indices = {int(idx) for idx in (result.get("sync_exact_action_indices") or [])}
            sync_actions = {
                canonical_action(item.get("action"))
                for item in (result.get("candidates") or [])
                if item.get("action") is not None and int(item.get("action_idx", -1)) in sync_action_indices
            }

            overall.add(teacher=teacher, result=result, model_action=model_action, final_action=final_action)
            by_turn[turn].add(teacher=teacher, result=result, model_action=model_action, final_action=final_action)
            teacher_modes[str(record.get("eval_mode") or record.get("teacher_eval_mode") or "unknown")] += 1

            result_row = {
                "line": line_no,
                "turn": turn,
                "teacher_eval_mode": record.get("eval_mode") or record.get("teacher_eval_mode"),
                "teacher_samples": teacher["samples"],
                "teacher_margin": teacher["margin"],
                "teacher_best_score": teacher["best_score"],
                "model_top1_action_idx": result.get("model_top1_action_idx"),
                "best_action_idx": result.get("best_action_idx"),
                "model_top1_refined_score": result.get("model_top1_refined_score"),
                "best_refined_score": result.get("best_refined_score"),
                "refined_override_delta": result.get("refined_override_delta"),
                "t1_override_features": result.get("t1_override_features"),
                "t1_override_gate_probability": result.get("t1_override_gate_probability"),
                "t1_override_gate_accepted": result.get("t1_override_gate_accepted"),
                "t1_override_reject_reason": result.get("t1_override_reject_reason"),
                "t1_arbitration_details": result.get("t1_arbitration_details"),
                "t1_arbitration_applied": result.get("t1_arbitration_applied"),
                "t1_final_challenger_details": result.get("t1_final_challenger_details"),
                "t1_final_challenger_applied": result.get("t1_final_challenger_applied"),
                "t1_refined_challenger_details": result.get("t1_refined_challenger_details"),
                "t1_refined_challenger_applied": result.get("t1_refined_challenger_applied"),
                "t1_blend_challenger_details": result.get("t1_blend_challenger_details"),
                "t1_blend_challenger_applied": result.get("t1_blend_challenger_applied"),
                "t1_low_risk_blend_challenger_details": result.get(
                    "t1_low_risk_blend_challenger_details"
                ),
                "t1_low_risk_blend_challenger_applied": result.get(
                    "t1_low_risk_blend_challenger_applied"
                ),
                "t1_model_rescue_details": result.get("t1_model_rescue_details"),
                "t1_model_rescue_applied": result.get("t1_model_rescue_applied"),
                "t1_final_bottom_sparse_rescue_applied": result.get("t1_final_bottom_sparse_rescue_applied"),
                "t1_final_bottom_sparse_rescue_action_idx": result.get("t1_final_bottom_sparse_rescue_action_idx"),
                "t1_final_bottom_sparse_rescue_delta": result.get("t1_final_bottom_sparse_rescue_delta"),
                "t1_no_refine_fallback_applied": result.get("t1_no_refine_fallback_applied"),
                "t1_no_refine_fallback_action_idx": result.get("t1_no_refine_fallback_action_idx"),
                "model_top1_hit": model_action == teacher["best_action"],
                "final_top1_hit": final_action == teacher["best_action"],
                "teacher_best_in_pool": teacher["best_action"] in pool_actions,
                "teacher_best_in_sync": teacher["best_action"] in sync_actions,
                "model_teacher_score": teacher["score_by_action"].get(model_action),
                "final_teacher_score": teacher["score_by_action"].get(final_action),
                "exact_evaluated": result.get("exact_evaluated"),
                "refinement_samples_by_action": result.get("refinement_samples_by_action"),
                "sync_refinement_candidate_count": result.get("sync_refinement_candidate_count"),
                "model_top1_overridden": result.get("model_top1_overridden"),
                "elapsed_ms": result.get("elapsed_ms"),
                "refinement_error": result.get("refinement_error"),
            }
            results_f.write(json.dumps(result_row, ensure_ascii=False) + "\n")
            if selector_f is not None:
                for selector_row in selector_rows_for_result(line_no, result, teacher):
                    selector_f.write(json.dumps(selector_row, ensure_ascii=False) + "\n")
            if not result_row["final_top1_hit"]:
                misses_f.write(
                    json.dumps(
                        {
                            **result_row,
                            "source": record.get("source"),
                            "source_line": record.get("source_line"),
                            "board": record.get("board"),
                            "opponent_board": record.get("opponent_board"),
                            "dealt": record.get("dealt"),
                            "known_discards": record.get("known_discards"),
                            "teacher_best_action": teacher["best_action"],
                            "model_action": model_action,
                            "final_action": final_action,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        if selector_f is not None:
            selector_f.close()

    report = {
        "input": str(args.input),
        "model": str(args.model),
        "output": str(output_dir),
        "teacher_modes": dict(teacher_modes),
        "config": {
            "runtime_config": args.runtime_config,
            "runtime_mode": args.runtime_mode,
            "shortlist_k": args.shortlist_k,
            "insurance_k": args.insurance_k,
            "sync_exact_k": args.sync_exact_k,
            "t2_sync_exact_k": args.t2_sync_exact_k,
            "t1_adaptive_sync_exact_k": args.t1_adaptive_sync_exact_k,
            "t1_adaptive_model_rank_k": args.t1_adaptive_model_rank_k,
            "t1_sync_model_insurance_k": args.t1_sync_model_insurance_k,
            "t1_sync_tactical_insurance_k": args.t1_sync_tactical_insurance_k,
            "t1_aux_shortlist_model": args.t1_aux_shortlist_model,
            "t1_aux_shortlist_k": args.t1_aux_shortlist_k,
            "max_pool": args.max_pool,
            "time_budget_ms": args.time_budget_ms,
            "skip": args.skip,
            "t1_refinement": args.t1_refinement,
            "t1_mc_sims": args.t1_mc_sims,
            "t1_recursive_beam": args.t1_recursive_beam,
            "t1_recursive_child_sims": args.t1_recursive_child_sims,
            "t1_refinement_first_batch_k": args.t1_refinement_first_batch_k,
            "t1_refinement_tail_min_remaining_ms": args.t1_refinement_tail_min_remaining_ms,
            "t1_refinement_timeout_headroom_ms": args.t1_refinement_timeout_headroom_ms,
            "t1_extra_refine_top_k": args.t1_extra_refine_top_k,
            "t1_extra_refine_margin": args.t1_extra_refine_margin,
            "t1_extra_refine_sims": args.t1_extra_refine_sims,
            "t1_sync_selection_policy": args.t1_sync_selection_policy,
            "t1_sync_selector": args.t1_sync_selector,
            "t1_selection_policy": args.t1_selection_policy,
            "t1_selection_model_weight": args.t1_selection_model_weight,
            "t1_final_selector": args.t1_final_selector,
            "t1_arbitration_selector": args.t1_arbitration_selector,
            "t1_arbitration_challenger_selector": args.t1_arbitration_challenger_selector,
            "t1_arbitration_policy": args.t1_arbitration_policy,
        "t1_arbitration_dual_gate_policy": args.t1_arbitration_dual_gate_policy,
        "t1_final_challenger_selector": args.t1_final_challenger_selector,
        "t1_final_challenger_gate": args.t1_final_challenger_gate,
        "t1_final_challenger_gate_policy": args.t1_final_challenger_gate_policy,
        "t1_refined_challenger_bust_max": args.t1_refined_challenger_bust_max,
        "t1_refined_challenger_refined_delta_max": args.t1_refined_challenger_refined_delta_max,
        "t1_blend_challenger_weight": args.t1_blend_challenger_weight,
        "t1_blend_challenger_bust_min": args.t1_blend_challenger_bust_min,
        "t1_blend_challenger_premium_fl_delta_min": args.t1_blend_challenger_premium_fl_delta_min,
        "t1_low_risk_blend_challenger_weight": args.t1_low_risk_blend_challenger_weight,
        "t1_low_risk_blend_challenger_fl_max": args.t1_low_risk_blend_challenger_fl_max,
        "t1_low_risk_blend_challenger_bust_max": args.t1_low_risk_blend_challenger_bust_max,
        "t1_model_rescue_qq_delta_min": args.t1_model_rescue_qq_delta_min,
        "t1_model_rescue_premium_delta_min": args.t1_model_rescue_premium_delta_min,
        "t1_model_rescue_candidate_fl_delta_min": args.t1_model_rescue_candidate_fl_delta_min,
        "t1_model_rescue_model_score_min": args.t1_model_rescue_model_score_min,
        "t1_model_rescue_model_bust_max": args.t1_model_rescue_model_bust_max,
        "t1_model_rescue_refined_delta_max": args.t1_model_rescue_refined_delta_max,
        "t1_final_bottom_sparse_rescue_margin": args.t1_final_bottom_sparse_rescue_margin,
        "t1_no_refine_fallback_policy": args.t1_no_refine_fallback_policy,
            "t1_override_margin": args.t1_override_margin,
            "t1_override_gate": args.t1_override_gate,
            "t1_override_gate_threshold": args.t1_override_gate_threshold,
            "t2_extra_margin": args.t2_extra_margin,
            "t2_baseline_min_samples": args.t2_baseline_min_samples,
            "t2_adaptive_shortlist_k": args.t2_adaptive_shortlist_k,
            "t2_adaptive_sync_exact_k": args.t2_adaptive_sync_exact_k,
            "t2_adaptive_max_model_score": args.t2_adaptive_max_model_score,
            "t2_refinement": args.t2_refinement,
            "t2_mc_sims": args.t2_mc_sims,
            "t2_initial_samples_per_candidate": args.t2_initial_samples_per_candidate,
            "t2_extra_samples_per_round": args.t2_extra_samples_per_round,
            "t2_max_samples_per_candidate": args.t2_max_samples_per_candidate,
            "t2_close_candidate_limit": args.t2_close_candidate_limit,
            "refinement_enabled": not args.disable_refinement,
            "turn_models": turn_models,
            "turn_model_blends": {
                str(turn): {
                    "model_a": spec[0],
                    "model_b": spec[1],
                    "model_b_weight": spec[2],
                }
                for turn, spec in sorted(turn_model_blends.items())
            },
        },
        "overall": overall.as_dict(),
        "by_turn": {str(turn): stats.as_dict() for turn, stats in sorted(by_turn.items())},
        "results_path": str(results_path),
        "misses_path": str(misses_path),
        "selector_rows_path": str(selector_rows_path) if args.write_selector_rows else "",
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(output_dir / "summary.md", report)
    return report


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate hybrid runtime refinement against teacher JSONL records")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument(
        "--runtime-config",
        default="",
        help="Optional JSON runtime config. Values are used as defaults unless the same CLI option is provided.",
    )
    parser.add_argument(
        "--runtime-mode",
        default="",
        help="Optional mode inside --runtime-config. Defaults to the config's default_mode.",
    )
    parser.add_argument(
        "--turn-model",
        action="append",
        default=[],
        help="Optional per-turn model override, e.g. 2=path/to/action_value_best.pt",
    )
    parser.add_argument(
        "--turn-model-blend",
        action="append",
        default=[],
        help="Optional per-turn two-model blend, e.g. 1=old.pt,new.pt,0.9. Overrides --turn-model for that turn.",
    )
    parser.add_argument("--device", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--turns", default="1,2")
    parser.add_argument("--shortlist-k", type=int, default=15)
    parser.add_argument("--insurance-k", type=int, default=5)
    parser.add_argument("--sync-exact-k", type=int, default=3)
    parser.add_argument("--t2-sync-exact-k", type=int, default=0)
    parser.add_argument("--t2-adaptive-shortlist-k", type=int, default=0)
    parser.add_argument("--t2-adaptive-sync-exact-k", type=int, default=0)
    parser.add_argument("--t2-adaptive-max-model-score", type=float, default=None)
    parser.add_argument("--t1-adaptive-sync-exact-k", type=int, default=0)
    parser.add_argument("--t1-adaptive-model-rank-k", type=int, default=0)
    parser.add_argument("--t1-sync-model-insurance-k", type=int, default=0)
    parser.add_argument("--t1-sync-tactical-insurance-k", type=int, default=0)
    parser.add_argument(
        "--t1-aux-shortlist-model",
        default="",
        help="Optional T1-only auxiliary model used only to add candidates to the shortlist.",
    )
    parser.add_argument("--t1-aux-shortlist-k", type=int, default=0)
    parser.add_argument("--max-pool", type=int, default=20)
    parser.add_argument("--time-budget-ms", type=int, default=5000)
    parser.add_argument("--t1-refinement", default="none", choices=["none", "mc_board", "recursive_mc"])
    parser.add_argument("--t1-mc-sims", type=int, default=32)
    parser.add_argument("--t1-recursive-beam", type=int, default=5)
    parser.add_argument("--t1-recursive-child-sims", type=int, default=2)
    parser.add_argument("--t1-refinement-first-batch-k", type=int, default=0)
    parser.add_argument("--t1-refinement-tail-min-remaining-ms", type=int, default=0)
    parser.add_argument("--t1-refinement-timeout-headroom-ms", type=int, default=0)
    parser.add_argument("--t1-extra-refine-top-k", type=int, default=0)
    parser.add_argument("--t1-extra-refine-margin", type=float, default=0.0)
    parser.add_argument("--t1-extra-refine-sims", type=int, default=0)
    parser.add_argument("--t1-sync-selection-policy", default="rank", choices=["rank", "selector"])
    parser.add_argument("--t1-sync-selector", default="")
    parser.add_argument("--t1-selection-policy", default="refined_score", choices=["refined_score", "refined_plus_model", "selector"])
    parser.add_argument("--t1-selection-model-weight", type=float, default=0.0)
    parser.add_argument("--t1-final-selector", default="")
    parser.add_argument("--t1-arbitration-selector", default="")
    parser.add_argument("--t1-arbitration-challenger-selector", default="")
    parser.add_argument(
        "--t1-arbitration-policy",
        default="none",
        choices=[
            "none",
            "current_refined_rank_ge2_and_qq_nonnegative",
            "pairwise_refined_delta_ge_m086_current_bust_ge_0026",
            "refined_score_model_rank_ge6_conflict_ge_m086",
            "mixed_kk_delta_le_0043_model_delta_le_1261",
            "mixed_kk_delta_le_0043_model_delta_le_1111_conflict_ge_m0483",
            "remain85_current_margin_le_0664",
        ],
    )
    parser.add_argument(
        "--t1-arbitration-dual-gate-policy",
        default="none",
        choices=["none", "new_if_kk_and_premium_delta_nonpositive"],
    )
    parser.add_argument("--t1-final-challenger-selector", default="")
    parser.add_argument("--t1-final-challenger-gate", default="")
    parser.add_argument(
        "--t1-final-challenger-gate-policy",
        default="none",
        choices=[
            "none",
            "guard2_bust_delta_ge_m066_rank_gap_delta_ge_m1",
            "guard2_bust_rank_refined_delta_ge_m0778",
            "guard2_bust_delta_ge_m066",
            "logistic_gate",
            "logistic_gate_refined_override_delta_ge_0677",
        ],
    )
    parser.add_argument("--t1-refined-challenger-bust-max", type=float, default=-1.0)
    parser.add_argument("--t1-refined-challenger-refined-delta-max", type=float, default=-1.0)
    parser.add_argument("--t1-blend-challenger-weight", type=float, default=-1.0)
    parser.add_argument("--t1-blend-challenger-bust-min", type=float, default=-1.0)
    parser.add_argument("--t1-blend-challenger-premium-fl-delta-min", type=float, default=-1.0)
    parser.add_argument("--t1-low-risk-blend-challenger-weight", type=float, default=-1.0)
    parser.add_argument("--t1-low-risk-blend-challenger-fl-max", type=float, default=-1.0)
    parser.add_argument("--t1-low-risk-blend-challenger-bust-max", type=float, default=-1.0)
    parser.add_argument("--t1-model-rescue-qq-delta-min", type=float, default=-1.0)
    parser.add_argument("--t1-model-rescue-premium-delta-min", type=float, default=-1.0)
    parser.add_argument("--t1-model-rescue-candidate-fl-delta-min", type=float, default=-1.0)
    parser.add_argument("--t1-model-rescue-model-score-min", type=float, default=-1.0)
    parser.add_argument("--t1-model-rescue-model-bust-max", type=float, default=-1.0)
    parser.add_argument("--t1-model-rescue-refined-delta-max", type=float, default=-1.0)
    parser.add_argument("--t1-final-bottom-sparse-rescue-margin", type=float, default=-1.0)
    parser.add_argument(
        "--t1-no-refine-fallback-policy",
        default="model",
        choices=["model", "fl_safe", "low_bust", "model_bust"],
    )
    parser.add_argument("--t1-override-margin", type=float, default=0.0)
    parser.add_argument("--t1-override-gate", default="")
    parser.add_argument("--t1-override-gate-threshold", type=float, default=0.5)
    parser.add_argument("--t2-extra-margin", type=float, default=1.0)
    parser.add_argument("--t2-baseline-min-samples", type=int, default=3)
    parser.add_argument("--t2-refinement", default="mc_board", choices=["exact_partial", "mc_board"])
    parser.add_argument("--t2-mc-sims", type=int, default=300)
    parser.add_argument("--t2-initial-samples-per-candidate", type=int, default=3)
    parser.add_argument("--t2-extra-samples-per-round", type=int, default=2)
    parser.add_argument("--t2-max-samples-per-candidate", type=int, default=24)
    parser.add_argument("--t2-close-candidate-limit", type=int, default=4)
    parser.add_argument("--disable-refinement", action="store_true")
    parser.add_argument("--rust-solver", default="")
    parser.add_argument("--write-selector-rows", action="store_true")
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(raw_argv)
    provided_dests = _provided_cli_dests(raw_argv)
    runtime_config_result = apply_hybrid_runtime_config(args, provided_dests=provided_dests)
    if runtime_config_result is not None:
        runtime_mode, _runtime_mode_config = runtime_config_result
        args.runtime_mode = runtime_mode
    if not args.model:
        parser.error("--model is required unless supplied by --runtime-config")
    report = evaluate(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
