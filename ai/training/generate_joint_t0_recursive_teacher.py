"""Generate BB-first and BTN-after-BB recursive T0 teacher data.

Each output line is one seat's T0 decision in the recursive-teacher format used
by ``convert_recursive_teacher_to_reranker.py``.  For BTN records, the BB T0
board is stored as ``opponent_board`` and as dead cards in ``known_discards`` so
the learned state can condition on BB's public placement.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ai.engine.encoding import ALL_CARDS  # noqa: E402
from ai.prob_engine_wrapper import evaluate_board_recursive_mc, evaluate_candidates  # noqa: E402
from ai.training.generate_recursive_t0_random import (  # noqa: E402
    action_label,
    apply_regular_candidate,
    board_from_candidate,
    card_rank,
    pick_regular_pool,
)


ROWS = ("top", "middle", "bottom")


def board_cards(board: dict[str, list[str]]) -> list[str]:
    return [card for row in ROWS for card in board.get(row, [])]


def normalize_pool_size_for_turn(turn: int, args: argparse.Namespace) -> int:
    sizes = [int(x) for x in args.trace_pool_sizes.split(",") if x.strip()]
    if not sizes:
        return args.trace_pool_size
    if turn - 1 < len(sizes):
        return sizes[turn - 1]
    return sizes[-1]


def choose_regular_pool(candidates: list[dict[str, Any]], turn: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    pool_size = normalize_pool_size_for_turn(turn, args)
    if pool_size <= 0 or pool_size >= len(candidates):
        return list(candidates)
    return pick_regular_pool(candidates, pool_size)


def objective_score(mc: dict[str, Any], args: argparse.Namespace) -> tuple[float, dict[str, float]]:
    rates = mc.get("fl_type_rates", {}) or {}
    qq = float(rates.get("qq", 0.0))
    kk = float(rates.get("kk", 0.0))
    aa = float(rates.get("aa", 0.0))
    trips = float(rates.get("trips", 0.0))
    fl_any = float(mc.get("fl_rate", 0.0))
    bust = float(mc.get("bust_rate", 0.0))
    ev = float(mc.get("avg_score", 0.0))
    score = (
        args.fl_any_weight * fl_any
        + args.fl_qq_weight * qq
        + args.fl_kk_weight * kk
        + args.fl_aa_weight * aa
        + args.fl_trips_weight * trips
        + args.ev_weight * ev
        - args.bust_weight * bust
    )
    return float(score), {
        "fl_any": fl_any,
        "fl_qq": qq,
        "fl_kk": kk,
        "fl_aa": aa,
        "fl_trips": trips,
        "bust": bust,
        "ev": ev,
    }


def attach_target(candidate: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    mc = (candidate.get("recursive") or {}).get("mc") or {}
    score, components = objective_score(mc, args)
    candidate["target_score"] = score
    candidate["target_components"] = components
    return candidate


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        f.flush()


def partial_path_for(output: Path) -> Path:
    return output.with_suffix(".partial.jsonl")


def progress_path_for(output: Path) -> Path:
    return output.with_suffix(".progress.json")


def load_partial_candidates(partial_path: Path, hand_index: int, seat: str) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    if not partial_path.exists():
        return candidates
    with partial_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("kind") != "t0_candidate":
                continue
            if int(item.get("hand_index", -1)) != hand_index or item.get("seat") != seat:
                continue
            candidate = item.get("candidate")
            if not isinstance(candidate, dict):
                continue
            action = str(candidate.get("action") or item.get("action") or "")
            if not action:
                continue
            candidates[action] = candidate
    return candidates


def load_partial_groups(partial_path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    groups: dict[tuple[int, str], dict[str, Any]] = {}
    if not partial_path.exists():
        return groups
    with partial_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("kind") != "t0_candidate":
                continue
            try:
                key = (int(item.get("hand_index", -1)), str(item.get("seat")))
            except (TypeError, ValueError):
                continue
            candidate = item.get("candidate") or {}
            action = str(candidate.get("action") or item.get("action") or "")
            if not action:
                continue
            group = groups.setdefault(key, {
                "hand_index": key[0],
                "seat": key[1],
                "position": item.get("position"),
                "pool_size": int(item.get("pool_size") or 0),
                "candidates": {},
                "last_updated_at": item.get("completed_at"),
            })
            group["pool_size"] = max(int(group.get("pool_size") or 0), int(item.get("pool_size") or 0))
            group["position"] = item.get("position") or group.get("position")
            group["last_updated_at"] = item.get("completed_at") or group.get("last_updated_at")
            group["candidates"][action] = candidate
    return groups


def write_progress(
    progress_path: Path | None,
    *,
    hand_index: int,
    seat: str,
    position: str,
    phase: str,
    completed_candidates: int,
    pool_size: int,
    current_rank: int | None = None,
    current_action: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    if progress_path is None:
        return
    payload: dict[str, Any] = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hand_index": hand_index,
        "seat": seat,
        "position": position,
        "phase": phase,
        "completed_candidates": completed_candidates,
        "pool_size": pool_size,
        "progress_percent": (100.0 * completed_candidates / pool_size) if pool_size else 100.0,
    }
    if current_rank is not None:
        payload["current_rank"] = current_rank
    if current_action is not None:
        payload["current_action"] = current_action
    if extra:
        payload.update(extra)
    write_json_atomic(progress_path, payload)


def select_t0_pool(candidates: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.t0_pool_size <= 0 or args.t0_pool_size >= len(candidates):
        return list(candidates)
    if not args.route_aware_t0_pool:
        from ai.training.generate_recursive_t0_random import pick_candidate_pool
        return pick_candidate_pool(candidates, args.t0_pool_size)

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def key(candidate: dict[str, Any]) -> str:
        return action_label(candidate)

    def add(items: Iterable[dict[str, Any]], target_total: int) -> None:
        for cand in items:
            if len(selected) >= args.t0_pool_size or len(selected) >= target_total:
                return
            k = key(cand)
            if k in seen:
                continue
            selected.append(cand)
            seen.add(k)

    def ranks(cards: list[str]) -> list[int]:
        return [card_rank(card) for card in cards]

    def top_has(candidate: dict[str, Any], wanted: set[int | str]) -> bool:
        top = board_from_candidate(candidate)["top"]
        for card in top:
            if card.startswith("X") and "X" in wanted:
                return True
            if card_rank(card) in wanted:
                return True
        return False

    def top_pair_or_trips(candidate: dict[str, Any]) -> bool:
        top = board_from_candidate(candidate)["top"]
        if len(top) < 2:
            return False
        jokers = sum(1 for c in top if c.startswith("X"))
        counts: dict[int, int] = {}
        for card in top:
            if not card.startswith("X"):
                counts[card_rank(card)] = counts.get(card_rank(card), 0) + 1
        return any(count + jokers >= 2 and rank >= 12 for rank, count in counts.items()) or jokers >= 2

    def top_trips(candidate: dict[str, Any]) -> bool:
        top = board_from_candidate(candidate)["top"]
        if len(top) < 3:
            return False
        jokers = sum(1 for c in top if c.startswith("X"))
        counts: dict[int, int] = {}
        for card in top:
            if not card.startswith("X"):
                counts[card_rank(card)] = counts.get(card_rank(card), 0) + 1
        return any(count + jokers >= 3 for count in counts.values()) or jokers >= 2

    def strong_bottom(candidate: dict[str, Any]) -> bool:
        bottom = board_from_candidate(candidate)["bottom"]
        if len(bottom) < 2:
            return False
        rs = ranks([card for card in bottom if not card.startswith("X")])
        broadways = sum(1 for rank in rs if rank >= 10)
        if broadways >= 2:
            return True
        if 12 in rs and 10 in rs:
            return True
        if len(set(rs)) < len(rs):
            return True
        if len(rs) >= 2 and max(rs) - min(rs) <= 2:
            return True
        suits = [card[-1] for card in bottom if not card.startswith("X") and len(card) >= 2]
        return len(suits) >= 2 and len(set(suits)) == 1

    def k_top_strong_bottom(candidate: dict[str, Any]) -> bool:
        return top_has(candidate, {13}) and strong_bottom(candidate)

    ev_sorted = sorted(candidates, key=lambda c: float(c.get("ev", -1e9)), reverse=True)
    fl_sorted = sorted(candidates, key=lambda c: float(c.get("fl_rate", 0.0)), reverse=True)
    safe_sorted = sorted(candidates, key=lambda c: (float(c.get("bust_prob", 1.0)), -float(c.get("ev", -1e9))))
    a_joker_top = sorted(
        [c for c in candidates if top_has(c, {14, "X"})],
        key=lambda c: (float(c.get("fl_rate", 0.0)), float(c.get("ev", -1e9))),
        reverse=True,
    )
    k_routes = sorted(
        [c for c in candidates if k_top_strong_bottom(c)],
        key=lambda c: (float(c.get("fl_rate", 0.0)), float(c.get("ev", -1e9))),
        reverse=True,
    )
    trips_routes = sorted(
        [c for c in candidates if top_trips(c)],
        key=lambda c: (float(c.get("fl_rate", 0.0)), float(c.get("ev", -1e9))),
        reverse=True,
    )
    fl_shape = sorted(
        [c for c in candidates if top_pair_or_trips(c)],
        key=lambda c: (float(c.get("fl_rate", 0.0)), float(c.get("ev", -1e9))),
        reverse=True,
    )

    # Quotas are cumulative target totals.  They keep EV leaders, but reserve
    # enough width for A/Joker and K-top route candidates that PE can rank low
    # before recursive continuation has a chance to evaluate them.
    n = args.t0_pool_size
    add(ev_sorted, max(1, round(n * 0.34)))
    add(a_joker_top, max(1, round(n * 0.54)))
    add(trips_routes, max(1, round(n * 0.64)))
    add(k_routes, max(1, round(n * 0.78)))
    add(fl_shape, max(1, round(n * 0.87)))
    add(fl_sorted, max(1, round(n * 0.94)))
    add(safe_sorted, n)
    add(ev_sorted, n)
    return selected[: args.t0_pool_size]


def trace_turn_decisions(
    initial_board: dict[str, list[str]],
    initial_dead: list[str],
    t0_dealt: list[str],
    opponent_board: dict[str, list[str]],
    position: str,
    rng: random.Random,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if not args.save_turn_traces:
        return []
    traces: list[dict[str, Any]] = []
    opponent_cards = board_cards(opponent_board)
    for rollout_idx in range(args.trace_rollouts):
        board = {row: list(cards) for row, cards in initial_board.items()}
        dead = list(dict.fromkeys(initial_dead + opponent_cards))
        deck_seen = set(t0_dealt + dead + board_cards(board))
        deck = [card for card in ALL_CARDS if card not in deck_seen]
        rng.shuffle(deck)
        for turn in range(1, 5):
            if len(deck) < 3:
                break
            dealt = deck[:3]
            deck = deck[3:]
            pe = evaluate_candidates(
                top=board["top"],
                mid=board["middle"],
                bot=board["bottom"],
                dealt=dealt,
                exclude=dead,
                turn=turn,
                position=position,
                engine_path=args.engine_path,
            )
            pool = choose_regular_pool(pe["candidates"], turn, args)
            evaluated = []
            for pool_rank, cand in enumerate(pool, start=1):
                next_board = apply_regular_candidate(board, cand)
                discard = cand.get("discard")
                next_dead = dead + ([discard] if discard else [])
                rec_start = time.time()
                if turn >= 4:
                    result = {
                        "mc": {
                            "avg_score": float(cand.get("ev", 0.0) or 0.0),
                            "bust_rate": float(cand.get("bust_prob", 0.0) or 0.0),
                            "fl_rate": float(cand.get("fl_rate", 0.0) or 0.0),
                            "fl_type_rates": cand.get("fl_type_rates", {}) or {},
                        },
                        "mode": "t4_exact_candidates",
                    }
                else:
                    result = evaluate_board_recursive_mc(
                        top=next_board["top"],
                        mid=next_board["middle"],
                        bot=next_board["bottom"],
                        exclude=next_dead,
                        start_turn=turn + 1,
                        sims=args.trace_sims,
                        beam_width=args.beam,
                        child_sims=args.child_sims,
                        engine_path=args.engine_path,
                    )
                evaluated.append(attach_target({
                    "pool_rank": pool_rank,
                    "placements": [(card, row) for card, row in cand.get("placements", [])],
                    "discard": discard,
                    "next_board": next_board,
                    "pe": {
                        "ev": cand.get("ev"),
                        "fl_rate": cand.get("fl_rate"),
                        "bust_prob": cand.get("bust_prob"),
                    },
                    "recursive": result,
                    "elapsed_s": time.time() - rec_start,
                }, args))
            evaluated.sort(key=lambda item: item["target_score"], reverse=True)
            if not evaluated:
                break
            chosen = evaluated[0]
            traces.append({
                "rollout": rollout_idx + 1,
                "turn": turn,
                "position": position,
                "opponent_board": opponent_board,
                "board_before": board,
                "dealt": dealt,
                "dead_before": list(dead),
                "chosen": chosen,
                "candidates": evaluated,
            })
            board = chosen["next_board"]
            if chosen.get("discard"):
                dead.append(chosen["discard"])
    return traces


def evaluate_t0_decision(
    hand_index: int,
    seat: str,
    position: str,
    dealt: list[str],
    opponent_board: dict[str, list[str]],
    known_discards: list[str],
    rng: random.Random,
    args: argparse.Namespace,
    partial_path: Path | None = None,
    progress_path: Path | None = None,
) -> dict[str, Any]:
    hand_start = time.time()
    exclude = list(dict.fromkeys(known_discards + board_cards(opponent_board)))
    pe = evaluate_candidates(
        top=[],
        mid=[],
        bot=[],
        dealt=dealt,
        exclude=exclude,
        turn=0,
        position=position,
        engine_path=args.engine_path,
    )
    pool = select_t0_pool(pe["candidates"], args)
    cached = load_partial_candidates(partial_path, hand_index, seat) if partial_path and args.resume else {}
    evaluated = []
    for pool_rank, cand in enumerate(pool, start=1):
        action = action_label(cand)
        if action in cached:
            reused = dict(cached[action])
            reused["pool_rank"] = pool_rank
            reused["checkpoint_reused"] = True
            evaluated.append(reused)
            write_progress(
                progress_path,
                hand_index=hand_index,
                seat=seat,
                position=position,
                phase="t0_candidate_reused",
                completed_candidates=len(evaluated),
                pool_size=len(pool),
                current_rank=pool_rank,
                current_action=action,
            )
            continue

        board = board_from_candidate(cand)
        write_progress(
            progress_path,
            hand_index=hand_index,
            seat=seat,
            position=position,
            phase="t0_candidate_running",
            completed_candidates=len(evaluated),
            pool_size=len(pool),
            current_rank=pool_rank,
            current_action=action,
        )
        rec_start = time.time()
        result = evaluate_board_recursive_mc(
            top=board["top"],
            mid=board["middle"],
            bot=board["bottom"],
            exclude=exclude,
            start_turn=1,
            sims=args.sims,
            beam_width=args.beam,
            child_sims=args.child_sims,
            engine_path=args.engine_path,
        )
        evaluated_candidate = attach_target({
            "pool_rank": pool_rank,
            "action": action,
            "board": board,
            "pe": {
                "ev": cand.get("ev"),
                "fl_rate": cand.get("fl_rate"),
                "bust_prob": cand.get("bust_prob"),
            },
            "recursive": result,
            "elapsed_s": time.time() - rec_start,
        }, args)
        evaluated.append(evaluated_candidate)
        if partial_path:
            append_jsonl(partial_path, {
                "kind": "t0_candidate",
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "hand_index": hand_index,
                "seat": seat,
                "position": position,
                "pool_size": len(pool),
                "completed_candidates": len(evaluated),
                "progress_percent": (100.0 * len(evaluated) / len(pool)) if pool else 100.0,
                "action": action,
                "candidate": evaluated_candidate,
            })
        write_progress(
            progress_path,
            hand_index=hand_index,
            seat=seat,
            position=position,
            phase="t0_candidate_completed",
            completed_candidates=len(evaluated),
            pool_size=len(pool),
            current_rank=pool_rank,
            current_action=action,
        )
    evaluated.sort(key=lambda item: item["target_score"], reverse=True)
    best_board = evaluated[0]["board"] if evaluated else {"top": [], "middle": [], "bottom": []}
    traces = []
    if evaluated:
        write_progress(
            progress_path,
            hand_index=hand_index,
            seat=seat,
            position=position,
            phase="turn_trace_running",
            completed_candidates=len(pool),
            pool_size=len(pool),
            extra={"trace_rollouts": args.trace_rollouts, "trace_sims": args.trace_sims},
        )
        traces = trace_turn_decisions(
            initial_board=best_board,
            initial_dead=exclude,
            t0_dealt=dealt,
            opponent_board=opponent_board,
            position=position,
            rng=rng,
            args=args,
        )
        write_progress(
            progress_path,
            hand_index=hand_index,
            seat=seat,
            position=position,
            phase="record_completed",
            completed_candidates=len(pool),
            pool_size=len(pool),
            extra={"trace_count": len(traces)},
        )
    return {
        "hand_index": hand_index,
        "seat": seat,
        "position": position,
        "dealt": dealt,
        "opponent_board": opponent_board,
        "known_discards": exclude,
        "objective": args.objective,
        "objective_weights": {
            "fl_any": args.fl_any_weight,
            "fl_qq": args.fl_qq_weight,
            "fl_kk": args.fl_kk_weight,
            "fl_aa": args.fl_aa_weight,
            "fl_trips": args.fl_trips_weight,
            "ev": args.ev_weight,
            "bust": args.bust_weight,
        },
        "sims": args.sims,
        "beam": args.beam,
        "child_sims": args.child_sims,
        "pool_size": len(pool),
        "best": evaluated[0] if evaluated else None,
        "candidates": evaluated,
        "turn_traces": traces,
        "elapsed_s": time.time() - hand_start,
    }


def sample_joint_deal(rng: random.Random) -> tuple[list[str], list[str]]:
    deck = list(ALL_CARDS)
    rng.shuffle(deck)
    return deck[:5], deck[5:10]


def load_done(output: Path) -> set[tuple[int, str]]:
    done: set[tuple[int, str]] = set()
    if not output.exists():
        return done
    with output.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "hand_index" in item and "seat" in item:
                done.add((int(item["hand_index"]), str(item["seat"])))
    return done


def summarize_output(
    output: Path,
    summary_path: Path,
    started_at: float,
    args: argparse.Namespace,
    partial_path: Path | None = None,
    progress_path: Path | None = None,
) -> None:
    records: list[dict[str, Any]] = []
    bad_lines = 0
    if output.exists():
        with output.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    bad_lines += 1
    best = [r["best"] for r in records if r.get("best")]
    best_mc = [(b.get("recursive") or {}).get("mc", {}) for b in best]
    by_seat: dict[str, int] = {}
    for record in records:
        by_seat[record.get("seat", "?")] = by_seat.get(record.get("seat", "?"), 0) + 1
    expected_records = args.hands * 2
    completed_keys = {(int(r.get("hand_index", -1)), str(r.get("seat"))) for r in records}
    partial_groups = load_partial_groups(partial_path) if partial_path else {}
    active_partial = {key: group for key, group in partial_groups.items() if key not in completed_keys}
    active_partial_summary = []
    partial_completed = 0
    partial_total = 0
    for key, group in sorted(active_partial.items()):
        completed = len(group.get("candidates") or {})
        pool_size = int(group.get("pool_size") or args.t0_pool_size or completed)
        partial_completed += completed
        partial_total += pool_size
        active_partial_summary.append({
            "hand_index": key[0],
            "seat": key[1],
            "position": group.get("position"),
            "completed_candidates": completed,
            "pool_size": pool_size,
            "progress_percent": (100.0 * completed / pool_size) if pool_size else 100.0,
            "last_updated_at": group.get("last_updated_at"),
        })
    completed_candidate_equiv = sum(int(r.get("pool_size") or args.t0_pool_size or 0) for r in records)
    completed_candidate_equiv += partial_completed
    estimated_total_candidates = max(expected_records * max(int(args.t0_pool_size or 0), 1), 1)
    current_progress: dict[str, Any] = {}
    if progress_path and progress_path.exists():
        try:
            current_progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current_progress = {"error": "could_not_parse_progress_file"}
    summary = {
        "hands_requested": args.hands,
        "records_expected": expected_records,
        "records_completed": len(records),
        "record_progress_percent": (100.0 * len(records) / expected_records) if expected_records else 100.0,
        "records_by_seat": by_seat,
        "bad_jsonl_lines": bad_lines,
        "partial_path": str(partial_path) if partial_path else None,
        "progress_path": str(progress_path) if progress_path else None,
        "partial_groups_active": active_partial_summary,
        "partial_candidate_groups": len(active_partial_summary),
        "partial_candidates_completed": partial_completed,
        "partial_candidates_total_active": partial_total,
        "estimated_candidates_completed": completed_candidate_equiv,
        "estimated_candidates_total": estimated_total_candidates,
        "estimated_candidate_progress_percent": 100.0 * completed_candidate_equiv / estimated_total_candidates,
        "current_progress": current_progress,
        "seed": args.seed,
        "objective": args.objective,
        "t0_pool_size": args.t0_pool_size,
        "route_aware_t0_pool": bool(args.route_aware_t0_pool),
        "sims": args.sims,
        "beam": args.beam,
        "child_sims": args.child_sims,
        "trace_sims": args.trace_sims,
        "trace_pool_sizes": args.trace_pool_sizes,
        "elapsed_s": time.time() - started_at,
        "output": str(output),
        "avg_target_score": sum(float(b.get("target_score", 0.0)) for b in best) / max(len(best), 1),
        "avg_best_fl": sum(float(m.get("fl_rate", 0.0)) for m in best_mc) / max(len(best_mc), 1),
        "avg_best_aa": sum(float((m.get("fl_type_rates", {}) or {}).get("aa", 0.0)) for m in best_mc) / max(len(best_mc), 1),
        "avg_best_kk": sum(float((m.get("fl_type_rates", {}) or {}).get("kk", 0.0)) for m in best_mc) / max(len(best_mc), 1),
        "avg_best_trips": sum(float((m.get("fl_type_rates", {}) or {}).get("trips", 0.0)) for m in best_mc) / max(len(best_mc), 1),
        "avg_best_bust": sum(float(m.get("bust_rate", 0.0)) for m in best_mc) / max(len(best_mc), 1),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate BB and BTN-after-BB recursive T0 teacher data")
    parser.add_argument("--hands", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--beam", type=int, default=10)
    parser.add_argument("--child-sims", type=int, default=2)
    parser.add_argument("--t0-pool-size", type=int, default=30)
    parser.add_argument("--route-aware-t0-pool", action=argparse.BooleanOptionalAction, default=True,
                        help="Reserve T0 pool slots for A/Joker top, K-top strong-bottom, and FL-shape routes")
    parser.add_argument("--save-turn-traces", action="store_true")
    parser.add_argument("--trace-rollouts", type=int, default=1)
    parser.add_argument("--trace-sims", type=int, default=32)
    parser.add_argument("--trace-pool-size", type=int, default=10)
    parser.add_argument("--trace-pool-sizes", default="0,15,10",
                        help="Comma-separated T1,T2,T3 pool sizes; 0 means all legal candidates")
    parser.add_argument("--objective", default="aa_focus")
    parser.add_argument("--fl-any-weight", type=float, default=0.0)
    parser.add_argument("--fl-qq-weight", type=float, default=0.0)
    parser.add_argument("--fl-kk-weight", type=float, default=4.0)
    parser.add_argument("--fl-aa-weight", type=float, default=30.0)
    parser.add_argument("--fl-trips-weight", type=float, default=20.0)
    parser.add_argument("--ev-weight", type=float, default=0.02)
    parser.add_argument("--bust-weight", type=float, default=0.0)
    parser.add_argument("--out-dir", default="ai/models/candidate_runs/joint-t0-aa-focus-20260521")
    parser.add_argument("--name", default="joint_t0_aa_focus")
    parser.add_argument("--engine-path", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--bb-exclude-btn", action="store_true",
                        help="Use BTN's private T0 cards as dead cards for BB evaluation")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{args.name}.jsonl"
    summary_path = out_dir / f"{args.name}.summary.json"
    partial_path = partial_path_for(output)
    progress_path = progress_path_for(output)
    if not args.resume:
        for stale_path in (partial_path, progress_path, summary_path):
            try:
                stale_path.unlink()
            except FileNotFoundError:
                pass
    done = load_done(output) if args.resume else set()
    rng = random.Random(args.seed)
    deals = [sample_joint_deal(rng) for _ in range(args.hands)]
    started_at = time.time()

    summarize_output(output, summary_path, started_at, args, partial_path, progress_path)
    with output.open("a" if args.resume else "w", encoding="utf-8") as f:
        for hand_index, (bb_dealt, btn_dealt) in enumerate(deals, start=1):
            bb_record = None
            if (hand_index, "bb") not in done:
                bb_known = btn_dealt if args.bb_exclude_btn else []
                bb_record = evaluate_t0_decision(
                    hand_index=hand_index,
                    seat="bb",
                    position="bb",
                    dealt=bb_dealt,
                    opponent_board={"top": [], "middle": [], "bottom": []},
                    known_discards=bb_known,
                    rng=rng,
                    args=args,
                    partial_path=partial_path,
                    progress_path=progress_path,
                )
                bb_record["bb_dealt"] = bb_dealt
                bb_record["btn_dealt"] = btn_dealt
                f.write(json.dumps(bb_record, ensure_ascii=False) + "\n")
                f.flush()
                done.add((hand_index, "bb"))
                summarize_output(output, summary_path, started_at, args, partial_path, progress_path)
            else:
                # Resume path still needs the BB board for BTN.  Prefer reading
                # the existing record, falling back to recomputing if needed.
                with output.open("r", encoding="utf-8") as existing:
                    for line in existing:
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if int(item.get("hand_index", -1)) == hand_index and item.get("seat") == "bb":
                            bb_record = item
                            break
            if not bb_record or not bb_record.get("best"):
                continue
            bb_board = bb_record["best"]["board"]

            if (hand_index, "btn") not in done:
                btn_record = evaluate_t0_decision(
                    hand_index=hand_index,
                    seat="btn",
                    position="btn",
                    dealt=btn_dealt,
                    opponent_board=bb_board,
                    known_discards=board_cards(bb_board),
                    rng=rng,
                    args=args,
                    partial_path=partial_path,
                    progress_path=progress_path,
                )
                btn_record["bb_dealt"] = bb_dealt
                btn_record["btn_dealt"] = btn_dealt
                btn_record["bb_t0_board"] = bb_board
                f.write(json.dumps(btn_record, ensure_ascii=False) + "\n")
                f.flush()
                done.add((hand_index, "btn"))

            summarize_output(output, summary_path, started_at, args, partial_path, progress_path)
            bb_best = bb_record["best"]
            bb_mc = bb_best["recursive"]["mc"]
            print(
                f"hand={hand_index}/{args.hands} BB {' '.join(bb_dealt)} "
                f"best={bb_best['action']} AA={bb_mc.get('fl_type_rates', {}).get('aa', 0.0):.3f} "
                f"FL={bb_mc.get('fl_rate', 0.0):.3f}",
                flush=True,
            )

    summarize_output(output, summary_path, started_at, args, partial_path, progress_path)
    print(f"wrote {output}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
