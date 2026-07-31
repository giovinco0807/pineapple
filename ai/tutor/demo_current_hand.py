"""Play and print one local hand with the current tutor runtime.

This is a diagnostic/demo runner.  It does not train or modify models.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.action_space import Action
from ai.engine.encoding import ALL_CARDS, Board
from ai.engine.game_engine import GameEngine, Hand
from ai.prob_engine_wrapper import evaluate_mc_t0, evaluate_t0_ladder
from ai.tutor.exact_late import evaluate_late_position
from ai.tutor.hybrid_t1t2 import (
    HybridConfig,
    _provided_cli_dests,
    action_to_dict,
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


ROWS = ("top", "middle", "bottom")


def board_dict(board: Board) -> dict[str, list[str]]:
    return {"top": list(board.top), "middle": list(board.middle), "bottom": list(board.bottom)}


def board_text(board: Board) -> str:
    return (
        f"T:{' '.join(board.top) or '-'} | "
        f"M:{' '.join(board.middle) or '-'} | "
        f"B:{' '.join(board.bottom) or '-'}"
    )


def _normalize_candidate_card(card: Any, dealt: list[str], used_jokers: set[str]) -> str:
    raw = str(card)
    if raw not in {"Xj", "JK", "X"}:
        return raw
    for joker in ("X1", "X2"):
        if joker in dealt and joker not in used_jokers:
            used_jokers.add(joker)
            return joker
    return "X1"


def action_from_candidate(candidate: dict[str, Any], dealt: list[str], *, turn: int) -> Action:
    used_jokers: set[str] = set()
    placements = [
        (_normalize_candidate_card(card, dealt, used_jokers), str(row))
        for card, row in (candidate.get("placements") or [])
    ]
    discard = candidate.get("discard")
    if discard in {"Xj", "JK", "X"}:
        discard = _normalize_candidate_card(discard, dealt, used_jokers)
    if turn == 0 or discard not in dealt:
        discard = None
    return Action(placements=placements, discard=discard)


def action_text(action: Action) -> str:
    placed = "; ".join(f"{card}->{row}" for card, row in action.placements)
    discard = action.discard if action.discard else "-"
    return f"{placed}; discard {discard}"


def exclude_cards(hand: Hand, seat: int) -> list[str]:
    exclude: list[str] = []
    exclude.extend(hand.boards[1 - seat].all_cards())
    exclude.extend(hand.discards[seat])
    return exclude


def payload_from_hand(hand: Hand, seat: int) -> dict[str, Any]:
    obs = hand.get_observation(seat)
    return {
        "turn": int(obs.turn),
        "board": board_dict(obs.board_self),
        "opponent_board": board_dict(obs.board_opponent),
        "dealt": list(obs.dealt_cards),
        "known_discards": list(obs.known_discards_self),
        "exclude": exclude_cards(hand, seat),
        "is_btn": bool(obs.is_btn),
        "is_fl": bool(obs.is_fl),
        "opp_is_fl": bool(obs.opp_is_fl),
    }


def namespace_from_runtime(runtime_config: Path, runtime_mode: str, device: str) -> argparse.Namespace:
    args = argparse.Namespace(
        runtime_config=str(runtime_config),
        runtime_mode=runtime_mode,
        model="",
        t1_aux_shortlist_model="",
        t1_aux_shortlist_k=0,
        turn_model=[],
        turn_model_blend=[],
        device=device,
        shortlist_k=15,
        insurance_k=5,
        sync_exact_k=3,
        t2_sync_exact_k=0,
        t1_adaptive_sync_exact_k=0,
        t1_adaptive_model_rank_k=0,
        t1_sync_model_insurance_k=0,
        t1_sync_tactical_insurance_k=0,
        max_pool=20,
        time_budget_ms=5000,
        t1_refinement="none",
        t1_mc_sims=32,
        t1_recursive_beam=5,
        t1_recursive_child_sims=2,
        t1_refinement_first_batch_k=0,
        t1_refinement_tail_min_remaining_ms=0,
        t1_refinement_timeout_headroom_ms=0,
        t1_extra_refine_top_k=0,
        t1_extra_refine_margin=0.0,
        t1_extra_refine_sims=0,
        t1_sync_selection_policy="rank",
        t1_sync_selector="",
        t1_selection_policy="refined_score",
        t1_selection_model_weight=0.0,
        t1_final_selector="",
        t1_arbitration_selector="",
        t1_arbitration_challenger_selector="",
        t1_arbitration_policy="none",
        t1_arbitration_dual_gate_policy="none",
        t1_final_challenger_selector="",
        t1_final_challenger_gate="",
        t1_final_challenger_gate_policy="none",
        t1_refined_challenger_bust_max=-1.0,
        t1_refined_challenger_refined_delta_max=-1.0,
        t1_blend_challenger_weight=-1.0,
        t1_blend_challenger_bust_min=-1.0,
        t1_blend_challenger_premium_fl_delta_min=-1.0,
        t1_low_risk_blend_challenger_weight=-1.0,
        t1_low_risk_blend_challenger_fl_max=-1.0,
        t1_low_risk_blend_challenger_bust_max=-1.0,
        t1_model_rescue_qq_delta_min=-1.0,
        t1_model_rescue_premium_delta_min=-1.0,
        t1_model_rescue_candidate_fl_delta_min=-1.0,
        t1_model_rescue_model_score_min=-1.0,
        t1_model_rescue_model_bust_max=-1.0,
        t1_model_rescue_refined_delta_max=-1.0,
        t1_final_bottom_sparse_rescue_margin=-1.0,
        t1_no_refine_fallback_policy="model",
        t1_override_margin=0.0,
        t1_override_gate="",
        t1_override_gate_threshold=0.5,
        t2_initial_samples_per_candidate=3,
        t2_extra_samples_per_round=2,
        t2_max_samples_per_candidate=24,
        t2_close_candidate_limit=4,
        t2_baseline_min_samples=3,
        t2_adaptive_shortlist_k=0,
        t2_adaptive_sync_exact_k=0,
        t2_adaptive_max_model_score=None,
        t2_refinement="mc_board",
        t2_mc_sims=300,
        disable_refinement=False,
    )
    runtime_result = apply_hybrid_runtime_config(args, provided_dests=_provided_cli_dests([]))
    if runtime_result is not None:
        args.runtime_mode = runtime_result[0]
    return args


def config_from_args(args: argparse.Namespace) -> HybridConfig:
    return HybridConfig(
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
        t2_initial_samples_per_candidate=args.t2_initial_samples_per_candidate,
        t2_extra_samples_per_round=args.t2_extra_samples_per_round,
        t2_max_samples_per_candidate=args.t2_max_samples_per_candidate,
        t2_close_candidate_limit=args.t2_close_candidate_limit,
        t2_baseline_min_samples=args.t2_baseline_min_samples,
        t2_adaptive_shortlist_k=args.t2_adaptive_shortlist_k,
        t2_adaptive_sync_exact_k=args.t2_adaptive_sync_exact_k,
        t2_adaptive_max_model_score=args.t2_adaptive_max_model_score,
        t2_refinement=args.t2_refinement,
        t2_mc_sims=args.t2_mc_sims,
        enable_sync_refinement=not args.disable_refinement,
    )


class CurrentRuntimePlayer:
    def __init__(
        self,
        args: argparse.Namespace,
        t0_sims: int,
        t2_sims: int,
        engine_path: str | None,
        *,
        t0_mode: str = "mc",
        t0_stage_sims: str = "4,12,32,96",
        t0_stage_limits: str = "80,32,12,5",
    ):
        self.args = args
        args.t2_mc_sims = int(t2_sims)
        self.config = config_from_args(args)
        self.t0_sims = int(t0_sims)
        self.t2_sims = int(t2_sims)
        self.engine_path = engine_path
        self.t0_mode = t0_mode
        self.t0_stage_sims = t0_stage_sims
        self.t0_stage_limits = t0_stage_limits
        self.evaluator = make_action_value_evaluator(
            model_path=args.model,
            model_paths_by_turn=parse_turn_model_specs(args.turn_model),
            model_blends_by_turn=parse_turn_model_blend_specs(args.turn_model_blend),
            device=args.device or None,
        )
        self.shortlist_evaluator = (
            make_action_value_evaluator(model_path=args.t1_aux_shortlist_model, device=args.device or None)
            if args.t1_aux_shortlist_model and args.t1_aux_shortlist_k > 0
            else None
        )

    def choose(self, hand: Hand, seat: int) -> tuple[Action, dict[str, Any]]:
        payload = payload_from_hand(hand, seat)
        board = hand.boards[seat]
        turn = int(hand.turn)
        dealt = list(hand.dealt_cards[seat])
        exclude = exclude_cards(hand, seat)
        started = time.perf_counter()
        if turn == 0:
            if self.t0_mode == "ladder":
                result = evaluate_t0_ladder(
                    dealt=dealt,
                    exclude=exclude,
                    stage_sims=self.t0_stage_sims,
                    stage_limits=self.t0_stage_limits,
                    engine_path=self.engine_path,
                )
            else:
                result = evaluate_mc_t0(
                    dealt=dealt,
                    exclude=exclude,
                    sims=self.t0_sims,
                    engine_path=self.engine_path,
                )
            best = result["candidates"][0]
            action = action_from_candidate(best, dealt, turn=turn)
            meta = {
                "source": f"t0_ladder_{self.t0_stage_sims}" if self.t0_mode == "ladder" else f"t0_mc{self.t0_sims}",
                "best": best,
                "candidates": result.get("candidates", []),
            }
        elif turn in {1, 2}:
            result = evaluate_hybrid_position(
                payload,
                evaluator=self.evaluator,
                shortlist_evaluator=self.shortlist_evaluator,
                config=self.config,
                rust_solver_path=self.engine_path,
            )
            best = result["best"]
            action = action_from_candidate(best["action"], dealt, turn=turn)
            meta = {"source": f"hybrid_t{turn}", **result}
        elif turn == 3:
            result = evaluate_late_position(
                board=board,
                dealt=dealt,
                turn=turn,
                opponent_board=hand.boards[1 - seat],
                exclude=exclude,
                top_n=20,
                prefer_rust=True,
                rust_solver_path=None,
                rust_timeout_s=max(1.0, self.config.time_budget_ms / 1000.0),
            )
            best = result["best"]
            action = action_from_candidate(best["action"], dealt, turn=turn)
            meta = {"source": f"{result.get('source', 'exact')}_t3", **result}
        else:
            result = evaluate_late_position(
                board=board,
                dealt=dealt,
                turn=turn,
                opponent_board=hand.boards[1 - seat],
                exclude=exclude,
                top_n=20,
                prefer_rust=False,
            )
            best = result["best"]
            action = action_from_candidate(best["action"], dealt, turn=turn)
            meta = {"source": f"{result.get('source', 'exact')}_t4", **result}
        meta["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        return action, meta


def metric_text(meta: dict[str, Any]) -> str:
    best = meta.get("best") or {}
    if "model_score" in best:
        parts = [
            f"model_rank {best.get('model_rank')}",
            f"model_score {float(best.get('model_score', 0.0)):.3f}",
        ]
        if best.get("refined_score") is not None:
            parts.append(f"refined {float(best['refined_score']):.3f}")
        parts.append(f"bust {100.0 * float(best.get('predicted_bust', 0.0)):.1f}%")
        parts.append(f"FL {100.0 * float(best.get('predicted_fl', 0.0)):.1f}%")
        parts.append(f"pool {meta.get('candidate_pool_size')}")
        parts.append(f"refined_n {meta.get('exact_evaluated')}")
        return ", ".join(parts)
    if "mc" in best:
        mc = best["mc"]
        return (
            f"score {float(mc.get('avg_score', 0.0)):.3f}, "
            f"bust {100.0 * float(mc.get('bust_rate', 0.0)):.1f}%, "
            f"FL {100.0 * float(mc.get('fl_rate', 0.0)):.1f}%, "
            f"sims {mc.get('simulations')}"
        )
    if "metrics" in best:
        metrics = best["metrics"] or {}
        return (
            f"score {float(metrics.get('score', metrics.get('ev', 0.0))):.3f}, "
            f"raw {float(metrics.get('raw_score', 0.0)):.3f}, "
            f"bust {100.0 * float(metrics.get('bust_rate', 0.0)):.1f}%, "
            f"FL {100.0 * float(metrics.get('fl_rate', 0.0)):.1f}%, "
            f"samples {metrics.get('samples')}"
        )
    return (
        f"EV {float(best.get('ev', 0.0)):.3f}, "
        f"bust {100.0 * float(best.get('bust_prob', 0.0)):.1f}%, "
        f"FL {100.0 * float(best.get('fl_rate', 0.0)):.1f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Show one hand played by the current local tutor runtime")
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument(
        "--runtime-config",
        default="ai/data/hybrid_t1t2_active_20260531/t1_runtime_current_local_20260604.json",
    )
    parser.add_argument("--runtime-mode", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--t0-sims", type=int, default=64)
    parser.add_argument("--t0-mode", choices=["mc", "ladder"], default="mc")
    parser.add_argument("--t0-stage-sims", default="4,12,32,96")
    parser.add_argument("--t0-stage-limits", default="80,32,12,5")
    parser.add_argument("--t2-sims", type=int, default=300)
    parser.add_argument("--engine-path", default="ai/rust_solver/target/release/prob_engine.exe")
    opts = parser.parse_args()

    runtime_args = namespace_from_runtime(Path(opts.runtime_config), opts.runtime_mode, opts.device)
    player = CurrentRuntimePlayer(
        runtime_args,
        t0_sims=opts.t0_sims,
        t2_sims=opts.t2_sims,
        engine_path=opts.engine_path,
        t0_mode=opts.t0_mode,
        t0_stage_sims=opts.t0_stage_sims,
        t0_stage_limits=opts.t0_stage_limits,
    )
    rng = random.Random(opts.seed)
    deck = list(ALL_CARDS)
    rng.shuffle(deck)
    hand = Hand(deck=deck, btn=0)
    logs: list[dict[str, Any]] = []

    while not hand.is_hand_complete():
        for seat in [0, 1]:
            before = hand.boards[seat].copy()
            dealt = list(hand.dealt_cards[seat])
            action, meta = player.choose(hand, seat)
            hand.apply_action(seat, action)
            logs.append(
                {
                    "seat": seat,
                    "turn": int(hand.turn),
                    "dealt": dealt,
                    "before": before,
                    "action": action,
                    "after": hand.boards[seat].copy(),
                    "source": meta.get("source"),
                    "elapsed_ms": meta.get("elapsed_ms"),
                    "metrics": metric_text(meta),
                }
            )
        if not hand.is_hand_complete():
            hand.deal_next_turn()

    result = GameEngine.compute_result(hand)
    summary = {
        "seed": opts.seed,
        "runtime_config": str(opts.runtime_config),
        "runtime_mode": runtime_args.runtime_mode,
        "t0_sims": opts.t0_sims,
        "t0_mode": opts.t0_mode,
        "logs": logs,
        "result": result,
    }
    print(format_summary(summary))


def format_summary(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(
        f"seed={summary['seed']} runtime_mode={summary['runtime_mode']} "
        f"t0_mode={summary['t0_mode']} t0_sims={summary['t0_sims']}"
    )
    lines.append("")
    for item in summary["logs"]:
        prefix = f"P{item['seat']} T{item['turn']}"
        lines.append(f"{prefix} dealt: {' '.join(item['dealt'])}")
        lines.append(f"{prefix} before: {board_text(item['before'])}")
        lines.append(f"{prefix} choose: {action_text(item['action'])}")
        lines.append(f"{prefix} after:  {board_text(item['after'])}")
        lines.append(
            f"{prefix} eval: {item['source']}, {item['metrics']}, "
            f"{float(item['elapsed_ms']):.0f} ms"
        )
        lines.append("")
    result = summary["result"]
    lines.append("Final")
    for seat, board in enumerate(result.boards):
        lines.append(f"P{seat}: {board_text(board)}")
        lines.append(
            f"    busted={result.busted[seat]} royalties={result.royalties[seat]['total']} "
            f"FL={result.fl_entry[seat]} names={json.dumps(result.hand_names[seat], ensure_ascii=False)}"
        )
    lines.append(f"raw_score: P0 {result.raw_score[0]} / P1 {result.raw_score[1]}")
    lines.append(f"line_results(P0): {result.line_results} scoop={result.scoop}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
