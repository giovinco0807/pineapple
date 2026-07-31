"""Build T3 exact-solver inputs by expanding T0/T1/T2 model branches.

The generator keeps every target-player branch selected by the requested
shortlists (default T0 top50, T1/T2 top10).  The opponent is advanced with the
same reranker family using top1 actions so the target boards come from a
plausible two-player route instead of random T3 board sampling.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.action_space import Action, get_initial_actions, get_turn_actions
from ai.engine.encoding import ALL_CARDS, Board, Observation, encode_state
from ai.models.action_value_reranker import ActionValueReranker, FL_TYPE_KEYS
from ai.training.action_feature_encoding import adapt_np_state_with_action
from ai.training.convert_action_value_teacher import post_action_observation


Position = Literal["bb", "btn"]
ROWS = ("top", "middle", "bottom")
DEFAULT_T0_MODEL = (
    "ai/models/candidate_runs/"
    "tutor-route10-t0top128-route3-mc30-1000-ft-20260525/model/action_value_best.pt"
)
DEFAULT_T1_MODEL = (
    "ai/models/candidate_runs/"
    "t1-runtime112-top40regret-x20-plus-currenthard15-x20-ft-20260604/model/action_value_best.pt"
)
DEFAULT_T2_MODEL = "ai/models/candidate_runs/t2-mix-broad80k-hard121-x10-ft-20260603/action_value_best.pt"
DEFAULT_T3_BB_MODEL = "ai/models/candidate_runs/t3-jokerfix-20k-bb-20260605/action_value_best.pt"


@dataclass(frozen=True)
class PlayerState:
    board: Board = field(default_factory=Board)
    known_discards: tuple[str, ...] = ()


@dataclass(frozen=True)
class Route:
    bb: PlayerState
    btn: PlayerState
    deck: tuple[str, ...]
    target_position: Position
    root_index: int
    target_t0_rank: int = -1
    target_t1_rank: int = -1
    target_t2_rank: int = -1
    branch_id: str = ""
    trace: tuple[dict[str, Any], ...] = ()


def board_from_rows(top: Iterable[str] = (), middle: Iterable[str] = (), bottom: Iterable[str] = ()) -> Board:
    return Board(top=list(top), middle=list(middle), bottom=list(bottom))


def board_to_dict(board: Board) -> dict[str, list[str]]:
    return {"top": list(board.top), "middle": list(board.middle), "bottom": list(board.bottom)}


def player_for(route: Route, position: Position) -> PlayerState:
    return route.btn if position == "btn" else route.bb


def with_player(route: Route, position: Position, player: PlayerState, *, deck: tuple[str, ...] | None = None) -> Route:
    kwargs = {"bb": route.bb, "btn": route.btn, "deck": route.deck if deck is None else deck}
    kwargs[position] = player
    return Route(
        **kwargs,
        target_position=route.target_position,
        root_index=route.root_index,
        target_t0_rank=route.target_t0_rank,
        target_t1_rank=route.target_t1_rank,
        target_t2_rank=route.target_t2_rank,
        branch_id=route.branch_id,
        trace=route.trace,
    )


def apply_action(board: Board, action: Action) -> Board:
    next_board = board.copy()
    for card, row in action.placements:
        getattr(next_board, row).append(card)
    return next_board


def apply_to_player(player: PlayerState, action: Action) -> PlayerState:
    discards = list(player.known_discards)
    if action.discard:
        discards.append(action.discard)
    return PlayerState(board=apply_action(player.board, action), known_discards=tuple(discards))


def action_to_dict(action: Action) -> dict[str, Any]:
    return {
        "placements": [[card, row] for card, row in action.placements],
        "discard": action.discard,
    }


def deal(deck: tuple[str, ...], n: int) -> tuple[list[str], tuple[str, ...]]:
    if len(deck) < n:
        raise ValueError(f"not enough cards left to deal {n}; remaining={len(deck)}")
    return list(deck[:n]), tuple(deck[n:])


def opponent_position(position: Position) -> Position:
    return "bb" if position == "btn" else "btn"


def exclude_for(player: PlayerState, opponent: PlayerState) -> list[str]:
    out = list(opponent.board.all_cards())
    out.extend(player.known_discards)
    return out


def decision_payload(
    *,
    board: Board,
    opponent_board: Board,
    dealt: list[str],
    known_discards: tuple[str, ...],
    turn: int,
    position: Position,
) -> dict[str, Any]:
    player = PlayerState(board=board, known_discards=known_discards)
    opponent = PlayerState(board=opponent_board)
    return {
        "turn": int(turn),
        "board": board_to_dict(board),
        "opponent_board": board_to_dict(opponent_board),
        "dealt": list(dealt),
        "known_discards": list(known_discards),
        "exclude": exclude_for(player, opponent),
        "is_btn": position == "btn",
        "position": position,
    }


def legal_actions(turn: int, dealt: list[str], board: Board) -> list[Action]:
    if turn == 0:
        return get_initial_actions(dealt, board)
    return get_turn_actions(dealt, board)


def load_model(path: str | Path, device: torch.device) -> ActionValueReranker:
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    model = ActionValueReranker.from_checkpoint(model_path, map_location=device)
    model.to(device)
    model.eval()
    return model


def score_actions(
    *,
    model: ActionValueReranker,
    device: torch.device,
    decision: dict[str, Any],
    actions: list[Action],
    batch_size: int,
) -> list[dict[str, Any]]:
    states: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for idx, action in enumerate(actions):
        candidate = action_to_dict(action)
        try:
            obs = post_action_observation(decision, candidate)
            state = adapt_np_state_with_action(
                np.asarray(encode_state(obs), dtype=np.float32),
                int(model.input_dim),
                decision,
                candidate,
            )
        except Exception as exc:
            rows.append(
                {
                    "rank": 999999,
                    "action_index": idx,
                    "action": action,
                    "candidate": candidate,
                    "model_score": float("-inf"),
                    "predicted_bust": 1.0,
                    "predicted_fl": 0.0,
                    "predicted_fl_types": {key: 0.0 for key in FL_TYPE_KEYS},
                    "error": str(exc),
                }
            )
            continue
        states.append(state)
        rows.append(
            {
                "rank": 999999,
                "action_index": idx,
                "action": action,
                "candidate": candidate,
            }
        )

    state_rows = [row for row in rows if "model_score" not in row]
    if states:
        scores: list[float] = []
        busts: list[float] = []
        fls: list[float] = []
        fl_types: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(states), batch_size):
                batch = np.stack(states[start : start + batch_size])
                tensor = torch.from_numpy(batch).to(device)
                turn_tensor = torch.full(
                    (tensor.shape[0],),
                    int(decision.get("turn", 0)),
                    dtype=torch.long,
                    device=device,
                )
                try:
                    out = model.predict_components(tensor, turn=turn_tensor)
                except TypeError:
                    out = model.predict_components(tensor)
                scores.extend(float(v) for v in out["score"].detach().cpu().numpy())
                busts.extend(float(v) for v in out["bust_prob"].detach().cpu().numpy())
                fls.extend(float(v) for v in out["fl_prob"].detach().cpu().numpy())
                fl_types.extend(out["fl_type_probs"].detach().cpu().numpy())
        for row, score, bust, fl, types in zip(state_rows, scores, busts, fls, fl_types):
            row["model_score"] = score
            row["predicted_bust"] = bust
            row["predicted_fl"] = fl
            row["predicted_fl_types"] = {
                key: float(types[i])
                for i, key in enumerate(FL_TYPE_KEYS)
            }

    rows.sort(key=lambda row: (float(row.get("model_score", float("-inf"))), -int(row["action_index"])), reverse=True)
    for rank, row in enumerate(rows):
        row["rank"] = rank
    return rows


def choose_ranked_actions(
    *,
    model: ActionValueReranker,
    device: torch.device,
    board: Board,
    opponent_board: Board,
    dealt: list[str],
    known_discards: tuple[str, ...],
    turn: int,
    position: Position,
    top_k: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    actions = legal_actions(turn, dealt, board)
    if not actions:
        return []
    decision = decision_payload(
        board=board,
        opponent_board=opponent_board,
        dealt=dealt,
        known_discards=known_discards,
        turn=turn,
        position=position,
    )
    rows = score_actions(model=model, device=device, decision=decision, actions=actions, batch_size=batch_size)
    return rows[: max(1, min(top_k, len(rows)))]


def trace_entry(
    *,
    seat: Position,
    turn: int,
    dealt: list[str],
    row: dict[str, Any],
    branch_role: str,
) -> dict[str, Any]:
    return {
        "seat": seat,
        "turn": int(turn),
        "dealt": list(dealt),
        "branch_role": branch_role,
        "model_rank": int(row["rank"]),
        "model_score": float(row["model_score"]),
        "predicted_bust": float(row["predicted_bust"]),
        "predicted_fl": float(row["predicted_fl"]),
        "predicted_fl_types": row["predicted_fl_types"],
        "action": row["candidate"],
    }


def advance_opponent_top1(
    *,
    route: Route,
    position: Position,
    dealt: list[str],
    turn: int,
    model: ActionValueReranker,
    device: torch.device,
    batch_size: int,
) -> Route:
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
        top_k=1,
        batch_size=batch_size,
    )
    if not rows:
        raise RuntimeError(f"no legal opponent actions: position={position} turn={turn}")
    next_player = apply_to_player(player, rows[0]["action"])
    next_route = with_player(route, position, next_player)
    return Route(
        bb=next_route.bb,
        btn=next_route.btn,
        deck=next_route.deck,
        target_position=next_route.target_position,
        root_index=next_route.root_index,
        target_t0_rank=next_route.target_t0_rank,
        target_t1_rank=next_route.target_t1_rank,
        target_t2_rank=next_route.target_t2_rank,
        branch_id=next_route.branch_id,
        trace=next_route.trace
        + (trace_entry(seat=position, turn=turn, dealt=dealt, row=rows[0], branch_role="opponent_top1"),),
    )


def expand_target(
    *,
    route: Route,
    dealt: list[str],
    turn: int,
    top_k: int,
    model: ActionValueReranker,
    device: torch.device,
    batch_size: int,
) -> list[Route]:
    position = route.target_position
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
        rank = int(row["rank"])
        t0_rank = rank if turn == 0 else route.target_t0_rank
        t1_rank = rank if turn == 1 else route.target_t1_rank
        t2_rank = rank if turn == 2 else route.target_t2_rank
        branch_id = "-".join(str(v) for v in (t0_rank, t1_rank, t2_rank) if v >= 0)
        out.append(
            Route(
                bb=base.bb,
                btn=base.btn,
                deck=base.deck,
                target_position=base.target_position,
                root_index=base.root_index,
                target_t0_rank=t0_rank,
                target_t1_rank=t1_rank,
                target_t2_rank=t2_rank,
                branch_id=branch_id,
                trace=base.trace
                + (trace_entry(seat=position, turn=turn, dealt=dealt, row=row, branch_role="target_branch"),),
            )
        )
    return out


def make_deck(rng: random.Random, include_jokers: bool) -> tuple[str, ...]:
    cards = list(ALL_CARDS if include_jokers else [card for card in ALL_CARDS if not card.startswith("X")])
    rng.shuffle(cards)
    return tuple(cards)


def root_route(root_index: int, target_position: Position, deck: tuple[str, ...]) -> Route:
    return Route(
        bb=PlayerState(board=board_from_rows()),
        btn=PlayerState(board=board_from_rows()),
        deck=deck,
        target_position=target_position,
        root_index=root_index,
    )


def expand_root(
    *,
    root_index: int,
    target_position: Position,
    deck: tuple[str, ...],
    models: dict[int | str, ActionValueReranker],
    device: torch.device,
    rng: random.Random,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    route = root_route(root_index, target_position, deck)
    bb_t0, deck_after_bb_t0 = deal(route.deck, 5)
    btn_t0, deck_after_t0 = deal(deck_after_bb_t0, 5)
    route = Route(
        bb=route.bb,
        btn=route.btn,
        deck=deck_after_t0,
        target_position=target_position,
        root_index=root_index,
    )

    if target_position == "bb":
        target_routes = expand_target(
            route=route,
            dealt=bb_t0,
            turn=0,
            top_k=args.t0_top_k,
            model=models[0],
            device=device,
            batch_size=args.batch_size,
        )
        routes = [
            advance_opponent_top1(
                route=r,
                position="btn",
                dealt=btn_t0,
                turn=0,
                model=models[0],
                device=device,
                batch_size=args.batch_size,
            )
            for r in target_routes
        ]
    else:
        route = advance_opponent_top1(
            route=route,
            position="bb",
            dealt=bb_t0,
            turn=0,
            model=models[0],
            device=device,
            batch_size=args.batch_size,
        )
        routes = expand_target(
            route=route,
            dealt=btn_t0,
            turn=0,
            top_k=args.t0_top_k,
            model=models[0],
            device=device,
            batch_size=args.batch_size,
        )

    routes = sample_routes(routes, rng=rng, max_routes=args.max_routes_per_position)

    for turn in (1, 2):
        bb_dealt, after_bb_deal = deal(routes[0].deck, 3)
        btn_dealt, after_round_deal = deal(after_bb_deal, 3)
        next_routes: list[Route] = []
        for r in routes:
            r = Route(
                bb=r.bb,
                btn=r.btn,
                deck=after_round_deal,
                target_position=r.target_position,
                root_index=r.root_index,
                target_t0_rank=r.target_t0_rank,
                target_t1_rank=r.target_t1_rank,
                target_t2_rank=r.target_t2_rank,
                branch_id=r.branch_id,
                trace=r.trace,
            )
            if target_position == "bb":
                for expanded in expand_target(
                    route=r,
                    dealt=bb_dealt,
                    turn=turn,
                    top_k=args.regular_top_k,
                    model=models[turn],
                    device=device,
                    batch_size=args.batch_size,
                ):
                    next_routes.append(
                        advance_opponent_top1(
                            route=expanded,
                            position="btn",
                            dealt=btn_dealt,
                            turn=turn,
                            model=models[turn],
                            device=device,
                            batch_size=args.batch_size,
                        )
                    )
            else:
                advanced = advance_opponent_top1(
                    route=r,
                    position="bb",
                    dealt=bb_dealt,
                    turn=turn,
                    model=models[turn],
                    device=device,
                    batch_size=args.batch_size,
                )
                next_routes.extend(
                    expand_target(
                        route=advanced,
                        dealt=btn_dealt,
                        turn=turn,
                        top_k=args.regular_top_k,
                        model=models[turn],
                        device=device,
                        batch_size=args.batch_size,
                    )
                )
        routes = sample_routes(
            next_routes,
            rng=rng,
            max_routes=args.max_routes_per_position,
        )

    rows: list[dict[str, Any]] = []
    if target_position == "bb":
        t3_dealt, _after_t3 = deal(routes[0].deck, 3)
        routes = sample_routes(routes, rng=rng, max_routes=args.max_routes_per_position)
        for route_index, r in enumerate(routes):
            rows.append(output_row(route=r, route_index=route_index, t3_dealt=t3_dealt, args=args))
        return rows

    # If the target is BTN, BB acts on T3 before BTN receives the T3 hand.
    bb_t3_dealt, after_bb_t3_deal = deal(routes[0].deck, 3)
    btn_t3_dealt, _after_btn_t3_deal = deal(after_bb_t3_deal, 3)
    routes = sample_routes(routes, rng=rng, max_routes=args.max_routes_per_position)
    for route_index, r in enumerate(routes):
        advanced = advance_opponent_top1(
            route=Route(
                bb=r.bb,
                btn=r.btn,
                deck=after_bb_t3_deal,
                target_position=r.target_position,
                root_index=r.root_index,
                target_t0_rank=r.target_t0_rank,
                target_t1_rank=r.target_t1_rank,
                target_t2_rank=r.target_t2_rank,
                branch_id=r.branch_id,
                trace=r.trace,
            ),
            position="bb",
            dealt=bb_t3_dealt,
            turn=3,
            model=models["t3_bb"],
            device=device,
            batch_size=args.batch_size,
        )
        rows.append(output_row(route=advanced, route_index=route_index, t3_dealt=btn_t3_dealt, args=args))
    return rows


def sample_routes(routes: list[Route], *, rng: random.Random, max_routes: int) -> list[Route]:
    if max_routes <= 0 or len(routes) <= max_routes:
        return routes
    indexes = sorted(rng.sample(range(len(routes)), max_routes))
    return [routes[i] for i in indexes]


def output_row(*, route: Route, route_index: int, t3_dealt: list[str], args: argparse.Namespace) -> dict[str, Any]:
    target = player_for(route, route.target_position)
    opponent = player_for(route, opponent_position(route.target_position))
    return {
        "source": "branch_expanded_t0_t1_t2",
        "source_line": route.root_index + 1,
        "turn": 3,
        "board": board_to_dict(target.board),
        "opponent_board": board_to_dict(opponent.board),
        "dealt": list(t3_dealt),
        "known_discards": list(target.known_discards),
        "exclude": exclude_for(target, opponent),
        "is_btn": route.target_position == "btn",
        "position": route.target_position,
        "reasons": [
            "turn_3",
            "branch_expanded",
            f"t0_top{args.t0_top_k}",
            f"t1t2_top{args.regular_top_k}",
            "exact_input",
        ],
        "branch": {
            "root_index": route.root_index,
            "route_index": route_index,
            "branch_id": route.branch_id,
            "target_position": route.target_position,
            "target_t0_rank": route.target_t0_rank,
            "target_t1_rank": route.target_t1_rank,
            "target_t2_rank": route.target_t2_rank,
        },
        "trace": list(route.trace),
        "generator_config": {
            "include_jokers": bool(args.include_jokers),
            "t0_top_k": int(args.t0_top_k),
            "regular_top_k": int(args.regular_top_k),
            "max_routes_per_position": int(args.max_routes_per_position),
            "opponent_policy": "model_top1",
            "btn_t3_opponent_policy": "bb_t3_model_top1",
        },
    }


def parse_positions(value: str) -> list[Position]:
    raw = value.lower().strip()
    if raw == "both":
        return ["bb", "btn"]
    if raw not in {"bb", "btn"}:
        raise argparse.ArgumentTypeError("--position must be bb, btn, or both")
    return [raw]  # type: ignore[list-item]


def build(args: argparse.Namespace) -> dict[str, Any]:
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    models: dict[int | str, ActionValueReranker] = {
        0: load_model(args.model_t0, device),
        1: load_model(args.model_t1, device),
        2: load_model(args.model_t2, device),
        "t3_bb": load_model(args.model_t3_bb, device),
    }

    started = time.time()
    rng = random.Random(args.seed)
    positions = parse_positions(args.position)
    written = 0
    root_summaries: list[dict[str, Any]] = []
    with out_path.open("w", encoding="utf-8") as f:
        for root_idx in range(args.roots):
            deck = make_deck(rng, include_jokers=args.include_jokers)
            root_written = 0
            for position in positions:
                rows = expand_root(
                    root_index=root_idx,
                    target_position=position,
                    deck=deck,
                    models=models,
                    device=device,
                    rng=rng,
                    args=args,
                )
                for row in rows:
                    if args.max_outputs and written >= args.max_outputs:
                        break
                    f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    written += 1
                    root_written += 1
                if args.max_outputs and written >= args.max_outputs:
                    break
            root_summaries.append({"root_index": root_idx, "written": root_written})
            if (root_idx + 1) % max(1, args.log_interval) == 0:
                elapsed = time.time() - started
                print(
                    f"roots={root_idx + 1:,}/{args.roots:,} written={written:,} "
                    f"elapsed={elapsed:.1f}s speed={written / max(elapsed, 1e-9):.1f}/s",
                    flush=True,
                )
            if args.max_outputs and written >= args.max_outputs:
                break

    summary = {
        "output": str(out_path),
        "roots_requested": int(args.roots),
        "positions": positions,
        "written": int(written),
        "t0_top_k": int(args.t0_top_k),
        "regular_top_k": int(args.regular_top_k),
        "max_routes_per_position": int(args.max_routes_per_position),
        "include_jokers": bool(args.include_jokers),
        "model_t0": str(args.model_t0),
        "model_t1": str(args.model_t1),
        "model_t2": str(args.model_t2),
        "model_t3_bb": str(args.model_t3_bb),
        "elapsed_seconds": time.time() - started,
        "root_summaries": root_summaries,
    }
    out_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build branch-expanded T3 exact-solver inputs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--roots", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--position", default="both", help="bb, btn, or both")
    parser.add_argument("--t0-top-k", type=int, default=50)
    parser.add_argument("--regular-top-k", type=int, default=10)
    parser.add_argument(
        "--max-routes-per-position",
        type=int,
        default=0,
        help="0 means keep every branch; otherwise sample this many target routes per position after each stage",
    )
    parser.add_argument("--max-outputs", type=int, default=0, help="0 means no cap")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--include-jokers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-t0", default=DEFAULT_T0_MODEL)
    parser.add_argument("--model-t1", default=DEFAULT_T1_MODEL)
    parser.add_argument("--model-t2", default=DEFAULT_T2_MODEL)
    parser.add_argument("--model-t3-bb", default=DEFAULT_T3_BB_MODEL)
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(build(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
