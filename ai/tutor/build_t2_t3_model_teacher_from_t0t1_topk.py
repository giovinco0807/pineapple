"""Build T2 all-action teacher rows from T0/T1 model TopK branches.

For each generated route, the target player's T0 and T1 decisions are expanded
with model TopK branches. At the target player's T2 decision, every legal T2
action is evaluated by sampling T3 deals and using a T3 action-value model to
choose the best T3 continuation for each draw.

This is a model-label generator, not a full exact oracle. It is intended for
fast local iteration and hard-case mining before promoting selected rows to
stronger capped/exact labels.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Literal

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.action_space import Action, get_turn_actions
from ai.models.action_value_reranker import ActionValueReranker
from ai.tutor.benchmark_t2_t3_value_model import evaluate_t2_candidate
from ai.tutor.build_branch_expanded_t3_targets import (
    DEFAULT_T0_MODEL,
    DEFAULT_T1_MODEL,
    DEFAULT_T2_MODEL,
    PlayerState,
    Route,
    action_to_dict,
    apply_to_player,
    board_to_dict,
    choose_ranked_actions,
    deal,
    exclude_for,
    load_model,
    make_deck,
    opponent_position,
    player_for,
    root_route,
    trace_entry,
    with_player,
)
from ai.tutor.run_t2_exact_oracle import action_key


Position = Literal["bb", "btn"]
DEFAULT_T3_BB_MODEL = (
    "D:/ofc-pineapple-data/t3_jokerfix_extra_20260605/models/"
    "t3-20k-plus-gen5k-hardneginit-bb/action_value_best.pt"
)
DEFAULT_T3_BTN_MODEL = (
    "D:/ofc-pineapple-data/t3_jokerfix_extra_20260605/models/"
    "t3-20k-plus-gen5k-hardneginit-btn/action_value_best.pt"
)


def parse_positions(value: str) -> list[Position]:
    raw = value.lower().strip()
    if raw == "both":
        return ["bb", "btn"]
    if raw not in {"bb", "btn"}:
        raise argparse.ArgumentTypeError("--position must be bb, btn, or both")
    return [raw]  # type: ignore[list-item]


def route_with_deck(route: Route, deck: tuple[str, ...]) -> Route:
    return Route(
        bb=route.bb,
        btn=route.btn,
        deck=deck,
        target_position=route.target_position,
        root_index=route.root_index,
        target_t0_rank=route.target_t0_rank,
        target_t1_rank=route.target_t1_rank,
        target_t2_rank=route.target_t2_rank,
        branch_id=route.branch_id,
        trace=route.trace,
    )


def branch_id_for(route: Route, *, t0_rank: int | None = None, t1_rank: int | None = None) -> str:
    values = (
        route.target_t0_rank if t0_rank is None else t0_rank,
        route.target_t1_rank if t1_rank is None else t1_rank,
    )
    return "-".join(str(v) for v in values if v >= 0)


def expand_position(
    *,
    route: Route,
    position: Position,
    dealt: list[str],
    turn: int,
    top_k: int,
    model: ActionValueReranker,
    device: torch.device,
    batch_size: int,
    branch_role: str,
) -> list[Route]:
    player = player_for(route, position)
    opponent = player_for(route, opponent_position(position))
    rows = choose_ranked_actions(
        model=model,
        device=device,
        board=player.board,
        opponent_board=opponent.board,
        dealt=dealt,
        known_discards=player.known_discards,
        turn=turn,
        position=position,
        top_k=top_k,
        batch_size=batch_size,
    )
    out: list[Route] = []
    for row in rows:
        next_player = apply_to_player(player, row["action"])
        base = with_player(route, position, next_player)
        t0_rank = route.target_t0_rank
        t1_rank = route.target_t1_rank
        if position == route.target_position and branch_role == "target_branch":
            if turn == 0:
                t0_rank = int(row["rank"])
            elif turn == 1:
                t1_rank = int(row["rank"])
        out.append(
            Route(
                bb=base.bb,
                btn=base.btn,
                deck=base.deck,
                target_position=base.target_position,
                root_index=base.root_index,
                target_t0_rank=t0_rank,
                target_t1_rank=t1_rank,
                target_t2_rank=base.target_t2_rank,
                branch_id=branch_id_for(base, t0_rank=t0_rank, t1_rank=t1_rank),
                trace=base.trace
                + (
                    trace_entry(
                        seat=position,
                        turn=turn,
                        dealt=dealt,
                        row=row,
                        branch_role=branch_role,
                    ),
                ),
            )
        )
    return out


def advance_round(
    *,
    routes: list[Route],
    turn: int,
    bb_dealt: list[str],
    btn_dealt: list[str],
    after_round_deck: tuple[str, ...],
    models: dict[int, ActionValueReranker],
    device: torch.device,
    args: argparse.Namespace,
) -> list[Route]:
    next_routes: list[Route] = []
    target_position = routes[0].target_position if routes else "bb"
    target_k = args.target_top_k
    opponent_k = args.opponent_top_k
    model = models[turn]

    for route in routes:
        route = route_with_deck(route, after_round_deck)
        if target_position == "bb":
            target_routes = expand_position(
                route=route,
                position="bb",
                dealt=bb_dealt,
                turn=turn,
                top_k=target_k,
                model=model,
                device=device,
                batch_size=args.batch_size,
                branch_role="target_branch",
            )
            for target_route in target_routes:
                next_routes.extend(
                    expand_position(
                        route=target_route,
                        position="btn",
                        dealt=btn_dealt,
                        turn=turn,
                        top_k=opponent_k,
                        model=model,
                        device=device,
                        batch_size=args.batch_size,
                        branch_role="opponent_branch" if opponent_k > 1 else "opponent_top1",
                    )
                )
        else:
            opponent_routes = expand_position(
                route=route,
                position="bb",
                dealt=bb_dealt,
                turn=turn,
                top_k=opponent_k,
                model=model,
                device=device,
                batch_size=args.batch_size,
                branch_role="opponent_branch" if opponent_k > 1 else "opponent_top1",
            )
            for opponent_route in opponent_routes:
                next_routes.extend(
                    expand_position(
                        route=opponent_route,
                        position="btn",
                        dealt=btn_dealt,
                        turn=turn,
                        top_k=target_k,
                        model=model,
                        device=device,
                        batch_size=args.batch_size,
                        branch_role="target_branch",
                    )
                )
    return next_routes


def routes_after_t1(
    *,
    root_index: int,
    target_position: Position,
    deck: tuple[str, ...],
    models: dict[int, ActionValueReranker],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[list[Route], dict[str, Any]]:
    route = root_route(root_index, target_position, deck)
    bb_t0, deck_after_bb_t0 = deal(route.deck, 5)
    btn_t0, deck_after_t0 = deal(deck_after_bb_t0, 5)
    routes = [
        Route(
            bb=route.bb,
            btn=route.btn,
            deck=deck_after_t0,
            target_position=target_position,
            root_index=root_index,
        )
    ]
    routes = advance_round(
        routes=routes,
        turn=0,
        bb_dealt=bb_t0,
        btn_dealt=btn_t0,
        after_round_deck=deck_after_t0,
        models=models,
        device=device,
        args=args,
    )

    bb_t1, deck_after_bb_t1 = deal(routes[0].deck, 3)
    btn_t1, deck_after_t1 = deal(deck_after_bb_t1, 3)
    routes = advance_round(
        routes=routes,
        turn=1,
        bb_dealt=bb_t1,
        btn_dealt=btn_t1,
        after_round_deck=deck_after_t1,
        models=models,
        device=device,
        args=args,
    )
    return routes, {
        "bb_t0": bb_t0,
        "btn_t0": btn_t0,
        "bb_t1": bb_t1,
        "btn_t1": btn_t1,
    }


def t2_candidate_stub(action: Action, index: int) -> dict[str, Any]:
    raw = action_to_dict(action)
    return {
        "source_index": int(index),
        "action": raw,
        "key": action_key(raw),
        "source_score": 0.0,
    }


def decision_record(route: Route, position: Position, dealt: list[str]) -> dict[str, Any]:
    player = player_for(route, position)
    opponent = player_for(route, opponent_position(position))
    return {
        "source": "t0t1_topk_t2_all_actions_t3_model",
        "source_line": route.root_index + 1,
        "turn": 2,
        "board": board_to_dict(player.board),
        "opponent_board": board_to_dict(opponent.board),
        "dealt": list(dealt),
        "known_discards": list(player.known_discards),
        "exclude": exclude_for(player, opponent),
        "is_btn": position == "btn",
        "position": position,
    }


def score_t2_record(
    *,
    record: dict[str, Any],
    route: Route,
    route_index: int,
    root_deals: dict[str, Any],
    t3_model: ActionValueReranker,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    player = player_for(route, route.target_position)
    actions = get_turn_actions(record["dealt"], player.board)
    if not actions:
        return None

    started = time.perf_counter()
    candidates = []
    for action_index, action in enumerate(actions):
        candidate = evaluate_t2_candidate(
            record,
            t2_candidate_stub(action, action_index),
            t3_model,
            device=device,
            draw_limit=args.draw_limit,
            seed=args.seed + route.root_index * 1_000_003 + route_index * 1009 + action_index * 37,
            batch_size=args.t3_batch_size,
            t3_rank_bust_weight=args.t3_rank_bust_weight,
            t3_rank_fl_any_weight=args.t3_rank_fl_any_weight,
        )
        candidates.append(candidate)

    sort_key = "model_t3_priority_score" if args.t2_sort_score == "priority" else "model_t3_value_score"
    candidates.sort(key=lambda row: float(row.get(sort_key, float("-inf"))), reverse=True)
    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank
        candidate["rank_score_key"] = sort_key
        candidate["t3_model_ev"] = float(candidate.get("model_t3_value_score", float("-inf")))

    return {
        **record,
        "mode": "t2_all_actions_via_t3_model",
        "estimated": True,
        "exact": False,
        "draw_limit": int(args.draw_limit),
        "candidate_count": len(candidates),
        "best_idx": 0 if candidates else None,
        "best": candidates[0] if candidates else None,
        "candidates": candidates,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "branch": {
            "root_index": route.root_index,
            "route_index": route_index,
            "branch_id": route.branch_id,
            "target_position": route.target_position,
            "target_t0_rank": route.target_t0_rank,
            "target_t1_rank": route.target_t1_rank,
        },
        "root_deals": root_deals,
        "trace": list(route.trace),
        "generator_config": {
            "target_top_k": int(args.target_top_k),
            "opponent_top_k": int(args.opponent_top_k),
            "draw_limit": int(args.draw_limit),
            "t2_sort_score": args.t2_sort_score,
            "include_jokers": bool(args.include_jokers),
        },
    }


def t2_records_for_root(
    *,
    root_index: int,
    target_position: Position,
    deck: tuple[str, ...],
    models: dict[int, ActionValueReranker],
    t3_models: dict[Position, ActionValueReranker],
    device: torch.device,
    args: argparse.Namespace,
    remaining_budget: int,
) -> list[dict[str, Any]]:
    routes, root_deals = routes_after_t1(
        root_index=root_index,
        target_position=target_position,
        deck=deck,
        models=models,
        device=device,
        args=args,
    )
    bb_t2, deck_after_bb_t2 = deal(routes[0].deck, 3)
    btn_t2, deck_after_t2 = deal(deck_after_bb_t2, 3)

    rows: list[dict[str, Any]] = []
    for route_index, route in enumerate(routes):
        if remaining_budget > 0 and len(rows) >= remaining_budget:
            break
        if target_position == "bb":
            record = decision_record(route, "bb", bb_t2)
            record["future_hidden_deals"] = {"btn_t2": btn_t2}
            scored = score_t2_record(
                record=record,
                route=route_with_deck(route, deck_after_t2),
                route_index=route_index,
                root_deals={**root_deals, "bb_t2": bb_t2, "btn_t2": btn_t2},
                t3_model=t3_models["bb"],
                device=device,
                args=args,
            )
            if scored is not None:
                rows.append(scored)
            continue

        opponent_routes = expand_position(
            route=route_with_deck(route, deck_after_t2),
            position="bb",
            dealt=bb_t2,
            turn=2,
            top_k=args.opponent_t2_top_k,
            model=models[2],
            device=device,
            batch_size=args.batch_size,
            branch_role="opponent_t2_branch" if args.opponent_t2_top_k > 1 else "opponent_t2_top1",
        )
        for opp_idx, advanced in enumerate(opponent_routes):
            if remaining_budget > 0 and len(rows) >= remaining_budget:
                break
            record = decision_record(advanced, "btn", btn_t2)
            scored = score_t2_record(
                record=record,
                route=advanced,
                route_index=route_index * max(1, args.opponent_t2_top_k) + opp_idx,
                root_deals={**root_deals, "bb_t2": bb_t2, "btn_t2": btn_t2},
                t3_model=t3_models["btn"],
                device=device,
                args=args,
            )
            if scored is not None:
                rows.append(scored)
    return rows


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# T2 From T0/T1 TopK via T3 Model",
        "",
        f"- output: `{summary['output']}`",
        f"- records: {summary['records']}",
        f"- positions: {summary['positions_written']}",
        f"- target topK T0/T1: {summary['target_top_k']}",
        f"- opponent topK T0/T1: {summary['opponent_top_k']}",
        f"- opponent T2 topK for BTN states: {summary['opponent_t2_top_k']}",
        f"- draw limit per T2 candidate: {summary['draw_limit']}",
        f"- candidates total: {summary['candidate_total']}",
        f"- avg candidates/record: {summary['avg_candidates_per_record']:.2f}",
        f"- T3 states scored: {summary['t3_states_scored']:,}",
        f"- elapsed: {summary['elapsed_seconds']:.1f}s",
        f"- records/sec: {summary['records_per_second']:.2f}",
        f"- T3 states/sec: {summary['t3_states_per_second']:.0f}",
        "",
        "## Models",
        "",
        f"- T0: `{summary['model_t0']}`",
        f"- T1: `{summary['model_t1']}`",
        f"- T2 opponent: `{summary['model_t2']}`",
        f"- T3 BB: `{summary['model_t3_bb']}`",
        f"- T3 BTN: `{summary['model_t3_btn']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    models: dict[int, ActionValueReranker] = {
        0: load_model(args.model_t0, device),
        1: load_model(args.model_t1, device),
        2: load_model(args.model_t2, device),
    }
    t3_models: dict[Position, ActionValueReranker] = {
        "bb": load_model(args.model_t3_bb, device),
        "btn": load_model(args.model_t3_btn, device),
    }

    rng = random.Random(args.seed)
    positions = parse_positions(args.position)
    started = time.perf_counter()
    written = 0
    candidate_total = 0
    t3_states_scored = 0
    positions_written = {"bb": 0, "btn": 0}
    root_summaries: list[dict[str, Any]] = []

    with output.open("w", encoding="utf-8") as handle:
        for root_i in range(args.roots):
            root_index = int(args.root_start) + int(root_i)
            deck = make_deck(rng, include_jokers=args.include_jokers)
            root_written = 0
            root_positions_written = {"bb": 0, "btn": 0}
            for position in positions:
                remaining_global = max(0, args.max_records - written) if args.max_records > 0 else 0
                remaining_position = (
                    max(0, args.max_records_per_position - positions_written[position])
                    if args.max_records_per_position > 0
                    else 0
                )
                remaining_root_position = (
                    max(0, args.max_records_per_root_per_position - root_positions_written[position])
                    if args.max_records_per_root_per_position > 0
                    else 0
                )
                if args.max_records > 0 and remaining_global <= 0:
                    break
                if args.max_records_per_position > 0 and remaining_position <= 0:
                    continue
                if args.max_records_per_root_per_position > 0 and remaining_root_position <= 0:
                    continue
                budgets = [v for v in (remaining_global, remaining_position, remaining_root_position) if v > 0]
                remaining = min(budgets) if budgets else 0
                rows = t2_records_for_root(
                    root_index=root_index,
                    target_position=position,
                    deck=deck,
                    models=models,
                    t3_models=t3_models,
                    device=device,
                    args=args,
                    remaining_budget=remaining,
                )
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    written += 1
                    root_written += 1
                    positions_written[str(row["position"])] += 1
                    root_positions_written[str(row["position"])] += 1
                    candidate_total += int(row.get("candidate_count") or 0)
                    t3_states_scored += sum(int(c.get("t3_states_scored") or 0) for c in row.get("candidates") or [])
                    if args.max_records > 0 and written >= args.max_records:
                        break
                if args.max_records > 0 and written >= args.max_records:
                    break
            root_summaries.append(
                {
                    "root_index": root_index,
                    "written": root_written,
                    "positions_written": root_positions_written,
                }
            )
            if (root_i + 1) % max(1, args.log_interval) == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"roots={root_i + 1}/{args.roots} records={written} "
                    f"candidates={candidate_total} elapsed={elapsed:.1f}s",
                    flush=True,
                )
            if args.max_records > 0 and written >= args.max_records:
                break

    elapsed = time.perf_counter() - started
    summary = {
        "output": str(output),
        "records": int(written),
        "roots_requested": int(args.roots),
        "root_start": int(args.root_start),
        "positions": positions,
        "positions_written": positions_written,
        "target_top_k": int(args.target_top_k),
        "opponent_top_k": int(args.opponent_top_k),
        "opponent_t2_top_k": int(args.opponent_t2_top_k),
        "draw_limit": int(args.draw_limit),
        "t2_sort_score": args.t2_sort_score,
        "candidate_total": int(candidate_total),
        "avg_candidates_per_record": candidate_total / max(written, 1),
        "t3_states_scored": int(t3_states_scored),
        "elapsed_seconds": elapsed,
        "records_per_second": written / max(elapsed, 1e-9),
        "t3_states_per_second": t3_states_scored / max(elapsed, 1e-9),
        "include_jokers": bool(args.include_jokers),
        "model_t0": str(args.model_t0),
        "model_t1": str(args.model_t1),
        "model_t2": str(args.model_t2),
        "model_t3_bb": str(args.model_t3_bb),
        "model_t3_btn": str(args.model_t3_btn),
        "root_summaries": root_summaries,
        "max_records": int(args.max_records),
        "max_records_per_position": int(args.max_records_per_position),
        "max_records_per_root_per_position": int(args.max_records_per_root_per_position),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(output.with_suffix(".summary.md"), summary)
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build T2 all-action rows from T0/T1 TopK and score with T3 model")
    parser.add_argument("--output", required=True)
    parser.add_argument("--roots", type=int, default=1)
    parser.add_argument("--root-start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--position", default="both", help="bb, btn, or both")
    parser.add_argument("--target-top-k", type=int, default=10)
    parser.add_argument("--opponent-top-k", type=int, default=1)
    parser.add_argument("--opponent-t2-top-k", type=int, default=1)
    parser.add_argument("--draw-limit", type=int, default=3)
    parser.add_argument("--max-records", type=int, default=0, help="0 means no cap")
    parser.add_argument("--max-records-per-position", type=int, default=0, help="0 means no per-position cap")
    parser.add_argument("--max-records-per-root-per-position", type=int, default=0,
                        help="0 means no per-root per-position cap")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for T0/T1/T2 branch models")
    parser.add_argument("--t3-batch-size", type=int, default=4096)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--include-jokers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--t2-sort-score", choices=["value", "priority"], default="value")
    parser.add_argument("--t3-rank-bust-weight", type=float, default=0.0)
    parser.add_argument("--t3-rank-fl-any-weight", type=float, default=0.0)
    parser.add_argument("--model-t0", default=DEFAULT_T0_MODEL)
    parser.add_argument("--model-t1", default=DEFAULT_T1_MODEL)
    parser.add_argument("--model-t2", default=DEFAULT_T2_MODEL)
    parser.add_argument("--model-t3-bb", default=DEFAULT_T3_BB_MODEL)
    parser.add_argument("--model-t3-btn", default=DEFAULT_T3_BTN_MODEL)
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(build(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
