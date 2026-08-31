"""Generate random T0 recursive-teacher review data.

For each random 5-card T0 hand, this script builds a mixed candidate pool and
evaluates each fixed T0 placement with recursive T1-T3 rollout and exact T4.
It writes one JSONL record per T0 hand so the run can be inspected or resumed.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ai.engine.encoding import ALL_CARDS  # noqa: E402
from ai.prob_engine_wrapper import evaluate_board_recursive_mc, evaluate_candidates  # noqa: E402


def normalize_row(row: str) -> str:
    if row in ("mid", "middle"):
        return "middle"
    if row in ("bot", "bottom"):
        return "bottom"
    return row


def card_rank(card: str) -> int:
    if card.startswith("X"):
        return 15
    return "23456789TJQKA".index(card[0]) + 2


def board_from_candidate(candidate: dict[str, Any]) -> dict[str, list[str]]:
    board = {"top": [], "middle": [], "bottom": []}
    for card, row in candidate.get("placements", []):
        board[normalize_row(row)].append(card)
    for row in board:
        board[row].sort(key=lambda c: (card_rank(c), c))
    return board


def candidate_key(candidate: dict[str, Any]) -> str:
    board = board_from_candidate(candidate)
    parts = []
    for row in ("top", "middle", "bottom"):
        parts.append(f"{row}:{','.join(board[row])}")
    return "|".join(parts)


def action_label(candidate: dict[str, Any]) -> str:
    board = board_from_candidate(candidate)
    return (
        f"T:{' '.join(board['top']) or '-'} | "
        f"M:{' '.join(board['middle']) or '-'} | "
        f"B:{' '.join(board['bottom']) or '-'}"
    )


def action_from_board(board: dict[str, list[str]]) -> str:
    return (
        f"T:{' '.join(board['top']) or '-'} | "
        f"M:{' '.join(board['middle']) or '-'} | "
        f"B:{' '.join(board['bottom']) or '-'}"
    )


def apply_regular_candidate(board: dict[str, list[str]], candidate: dict[str, Any]) -> dict[str, list[str]]:
    out = {row: list(cards) for row, cards in board.items()}
    for card, row in candidate.get("placements", []):
        out[normalize_row(row)].append(card)
    for row in out:
        out[row].sort(key=lambda c: (card_rank(c), c))
    return out


def has_high_or_joker(cards: list[str]) -> bool:
    return any(card.startswith("X") or card_rank(card) >= 12 for card in cards)


def is_fl_shape(candidate: dict[str, Any]) -> bool:
    top = board_from_candidate(candidate)["top"]
    if not top:
        return False
    jokers = sum(1 for c in top if c.startswith("X"))
    ranks: dict[int, int] = {}
    for card in top:
        if not card.startswith("X"):
            ranks[card_rank(card)] = ranks.get(card_rank(card), 0) + 1
    if any(ranks.get(rank, 0) + jokers >= 2 for rank in (12, 13, 14)):
        return True
    return any(count + jokers >= 3 for count in ranks.values()) or jokers >= 2


def is_top_trips_shape(candidate: dict[str, Any]) -> bool:
    top = board_from_candidate(candidate)["top"]
    if len(top) < 3:
        return False
    jokers = sum(1 for c in top if c.startswith("X"))
    ranks: dict[int, int] = {}
    for card in top:
        if not card.startswith("X"):
            ranks[card_rank(card)] = ranks.get(card_rank(card), 0) + 1
    return any(count + jokers >= 3 for count in ranks.values()) or jokers >= 2


def is_bottom_power(candidate: dict[str, Any]) -> bool:
    board = board_from_candidate(candidate)
    bottom = board["bottom"]
    top = board["top"]
    return has_high_or_joker(top) and any(c.startswith("X") for c in bottom) and any(card_rank(c) >= 12 for c in bottom)


def pick_candidate_pool(candidates: list[dict[str, Any]], pool_size: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(items: list[dict[str, Any]], max_count: int) -> None:
        for cand in items:
            if len(selected) >= pool_size or len(selected) >= max_count:
                return
            key = candidate_key(cand)
            if key in seen:
                continue
            selected.append(cand)
            seen.add(key)

    ev_sorted = sorted(candidates, key=lambda c: float(c.get("ev", -1e9)), reverse=True)
    fl_sorted = sorted(candidates, key=lambda c: float(c.get("fl_rate", 0.0)), reverse=True)
    safe_sorted = sorted(candidates, key=lambda c: (float(c.get("bust_prob", 1.0)), -float(c.get("ev", -1e9))))
    route_sorted = sorted(
        [c for c in candidates if is_fl_shape(c) or is_bottom_power(c)],
        key=lambda c: (float(c.get("fl_rate", 0.0)), float(c.get("ev", -1e9))),
        reverse=True,
    )
    trips_sorted = sorted(
        [c for c in candidates if is_top_trips_shape(c)],
        key=lambda c: (float(c.get("fl_rate", 0.0)), float(c.get("ev", -1e9))),
        reverse=True,
    )

    add(ev_sorted, max(2, pool_size // 3))
    add(fl_sorted, max(4, pool_size * 2 // 3))
    add(trips_sorted, max(5, pool_size * 3 // 4))
    add(route_sorted, max(6, pool_size * 5 // 6))
    add(safe_sorted, pool_size)
    add(ev_sorted, pool_size)
    return selected


def pick_regular_pool(candidates: list[dict[str, Any]], pool_size: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def key(cand: dict[str, Any]) -> str:
        placements = sorted((card, normalize_row(row)) for card, row in cand.get("placements", []))
        return json.dumps({"placements": placements, "discard": cand.get("discard")}, sort_keys=True)

    def add(items: list[dict[str, Any]], max_count: int) -> None:
        for cand in items:
            if len(selected) >= pool_size or len(selected) >= max_count:
                return
            k = key(cand)
            if k in seen:
                continue
            selected.append(cand)
            seen.add(k)

    ev_sorted = sorted(candidates, key=lambda c: float(c.get("ev", -1e9)), reverse=True)
    fl_sorted = sorted(candidates, key=lambda c: float(c.get("fl_rate", 0.0)), reverse=True)
    safe_sorted = sorted(candidates, key=lambda c: (float(c.get("bust_prob", 1.0)), -float(c.get("ev", -1e9))))
    add(ev_sorted, max(2, pool_size // 2))
    add(fl_sorted, max(3, pool_size * 3 // 4))
    add(safe_sorted, pool_size)
    add(ev_sorted, pool_size)
    return selected


def trace_turn_decisions(
    initial_board: dict[str, list[str]],
    initial_dead: list[str],
    t0_dealt: list[str],
    rng: random.Random,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Collect T1-T3 recursive labels along sampled post-T0 trajectories."""
    if not args.save_turn_traces:
        return []
    traces: list[dict[str, Any]] = []
    for rollout_idx in range(args.trace_rollouts):
        board = {row: list(cards) for row, cards in initial_board.items()}
        dead = list(initial_dead)
        deck = [card for card in ALL_CARDS if card not in set(t0_dealt + dead + board["top"] + board["middle"] + board["bottom"])]
        rng.shuffle(deck)
        for turn in range(1, 4):
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
                position="bb",
                engine_path=args.engine_path,
            )
            pool = pick_regular_pool(pe["candidates"], args.trace_pool_size)
            evaluated = []
            for pool_rank, cand in enumerate(pool, start=1):
                next_board = apply_regular_candidate(board, cand)
                next_dead = dead + ([cand["discard"]] if cand.get("discard") else [])
                rec_start = time.time()
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
                evaluated.append({
                    "pool_rank": pool_rank,
                    "placements": [(card, normalize_row(row)) for card, row in cand.get("placements", [])],
                    "discard": cand.get("discard"),
                    "next_board": next_board,
                    "pe": {
                        "ev": cand.get("ev"),
                        "fl_rate": cand.get("fl_rate"),
                        "bust_prob": cand.get("bust_prob"),
                    },
                    "recursive": result,
                    "elapsed_s": time.time() - rec_start,
                })
            evaluated.sort(key=lambda x: x["recursive"]["mc"]["avg_score"], reverse=True)
            if not evaluated:
                break
            chosen = evaluated[0]
            traces.append({
                "rollout": rollout_idx + 1,
                "turn": turn,
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


def sample_deal(rng: random.Random) -> list[str]:
    deck = list(ALL_CARDS)
    rng.shuffle(deck)
    return deck[:5]


def load_done(output: Path) -> set[int]:
    done: set[int] = set()
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
            if "hand_index" in item:
                done.add(int(item["hand_index"]))
    return done


def write_summary(output: Path, summary_path: Path, started_at: float, args: argparse.Namespace) -> None:
    records = []
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
    best = [r["best"]["recursive"]["mc"] for r in records if r.get("best")]
    summary = {
        "hands_requested": args.hands,
        "hands_completed": len(records),
        "bad_jsonl_lines": bad_lines,
        "seed": args.seed,
        "sims": args.sims,
        "beam": args.beam,
        "child_sims": args.child_sims,
        "pool_size": args.pool_size,
        "elapsed_s": time.time() - started_at,
        "output": str(output),
        "avg_best_score": sum(m["avg_score"] for m in best) / max(len(best), 1),
        "avg_best_fl": sum(m["fl_rate"] for m in best) / max(len(best), 1),
        "avg_best_bust": sum(m["bust_rate"] for m in best) / max(len(best), 1),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate random T0 recursive review data")
    parser.add_argument("--hands", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--sims", type=int, default=64)
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--child-sims", type=int, default=2)
    parser.add_argument("--pool-size", type=int, default=10)
    parser.add_argument("--save-turn-traces", action="store_true")
    parser.add_argument("--trace-rollouts", type=int, default=1)
    parser.add_argument("--trace-sims", type=int, default=16)
    parser.add_argument("--trace-pool-size", type=int, default=5)
    parser.add_argument("--out-dir", default="ai/models/candidate_runs/fl-route-reranker-v2-medium-20260520")
    parser.add_argument("--name", default="recursive_t0_random100_s64_b5_c2_pool10")
    parser.add_argument("--engine-path", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{args.name}.jsonl"
    summary_path = out_dir / f"{args.name}.summary.json"
    done = load_done(output) if args.resume else set()
    rng = random.Random(args.seed)
    deals = [sample_deal(rng) for _ in range(args.hands)]
    started_at = time.time()

    with output.open("a" if args.resume else "w", encoding="utf-8") as f:
        for hand_index, dealt in enumerate(deals, start=1):
            if hand_index in done:
                continue
            hand_start = time.time()
            pe = evaluate_candidates(
                top=[],
                mid=[],
                bot=[],
                dealt=dealt,
                exclude=[],
                turn=0,
                position="bb",
                engine_path=args.engine_path,
            )
            pool = pick_candidate_pool(pe["candidates"], args.pool_size)
            evaluated = []
            for pool_rank, cand in enumerate(pool, start=1):
                board = board_from_candidate(cand)
                rec_start = time.time()
                result = evaluate_board_recursive_mc(
                    top=board["top"],
                    mid=board["middle"],
                    bot=board["bottom"],
                    exclude=[],
                    start_turn=1,
                    sims=args.sims,
                    beam_width=args.beam,
                    child_sims=args.child_sims,
                    engine_path=args.engine_path,
                )
                evaluated.append({
                    "pool_rank": pool_rank,
                    "action": action_label(cand),
                    "board": board,
                    "pe": {
                        "ev": cand.get("ev"),
                        "fl_rate": cand.get("fl_rate"),
                        "bust_prob": cand.get("bust_prob"),
                    },
                    "recursive": result,
                    "elapsed_s": time.time() - rec_start,
                })
            evaluated.sort(key=lambda x: x["recursive"]["mc"]["avg_score"], reverse=True)
            traces = []
            if evaluated:
                traces = trace_turn_decisions(
                    initial_board=evaluated[0]["board"],
                    initial_dead=[],
                    t0_dealt=dealt,
                    rng=rng,
                    args=args,
                )
            record = {
                "hand_index": hand_index,
                "dealt": dealt,
                "sims": args.sims,
                "beam": args.beam,
                "child_sims": args.child_sims,
                "pool_size": len(pool),
                "best": evaluated[0] if evaluated else None,
                "candidates": evaluated,
                "turn_traces": traces,
                "elapsed_s": time.time() - hand_start,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            write_summary(output, summary_path, started_at, args)
            best = record["best"]
            mc = best["recursive"]["mc"]
            print(
                f"hand={hand_index}/{args.hands} dealt={' '.join(dealt)} "
                f"best={best['action']} score={mc['avg_score']:+.3f} "
                f"FL={mc['fl_rate']:.3f} bust={mc['bust_rate']:.3f} "
                f"elapsed={record['elapsed_s']:.1f}s",
                flush=True,
            )

    write_summary(output, summary_path, started_at, args)
    print(f"wrote {output}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
