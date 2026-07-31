"""Compare fast and deeper local tutor play on the same shuffled hands."""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.encoding import ALL_CARDS
from ai.engine.game_engine import GameEngine, Hand
from ai.tutor.demo_current_hand import (
    CurrentRuntimePlayer,
    action_text,
    board_text,
    metric_text,
    namespace_from_runtime,
)


def canonical_action(action) -> tuple[tuple[tuple[str, str], ...], str]:
    return (
        tuple(sorted((str(card), str(row)) for card, row in action.placements)),
        str(action.discard or ""),
    )


def play_hand(seed: int, player: CurrentRuntimePlayer) -> dict[str, Any]:
    rng = random.Random(seed)
    deck = list(ALL_CARDS)
    rng.shuffle(deck)
    hand = Hand(deck=deck, btn=0)
    logs: list[dict[str, Any]] = []
    started = time.perf_counter()

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
                    "elapsed_ms": float(meta.get("elapsed_ms", 0.0) or 0.0),
                    "metrics": metric_text(meta),
                }
            )
        if not hand.is_hand_complete():
            hand.deal_next_turn()

    result = GameEngine.compute_result(hand)
    return {
        "seed": seed,
        "logs": logs,
        "result": result,
        "elapsed_s": time.perf_counter() - started,
    }


def changed_decisions(fast: dict[str, Any], deep: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    out: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for fast_log, deep_log in zip(fast["logs"], deep["logs"]):
        if canonical_action(fast_log["action"]) != canonical_action(deep_log["action"]):
            out.append((fast_log, deep_log))
    return out


def row_summary(fast: dict[str, Any], deep: dict[str, Any]) -> str:
    fast_result = fast["result"]
    deep_result = deep["result"]
    changed = changed_decisions(fast, deep)
    changed_tags = ",".join(f"P{f['seat']}T{f['turn']}" for f, _d in changed) or "-"
    delta = int(deep_result.raw_score[0]) - int(fast_result.raw_score[0])
    return (
        f"{fast['seed']:>10} | "
        f"{fast_result.raw_score[0]:>4} | {deep_result.raw_score[0]:>4} | {delta:>+4} | "
        f"{str(fast_result.busted[0])[0]}/{str(deep_result.busted[0])[0]} | "
        f"{len(changed):>2} | {changed_tags} | "
        f"{fast['elapsed_s']:.1f}s/{deep['elapsed_s']:.1f}s"
    )


def detail_summary(fast: dict[str, Any], deep: dict[str, Any]) -> str:
    lines: list[str] = []
    seed = fast["seed"]
    fast_result = fast["result"]
    deep_result = deep["result"]
    lines.append(f"\nseed {seed}")
    lines.append(
        f"  fast score P0={fast_result.raw_score[0]} busted={fast_result.busted[0]} "
        f"P0={board_text(fast_result.boards[0])}"
    )
    lines.append(
        f"  deep score P0={deep_result.raw_score[0]} busted={deep_result.busted[0]} "
        f"P0={board_text(deep_result.boards[0])}"
    )
    changed = changed_decisions(fast, deep)
    if not changed:
        lines.append("  changed decisions: none")
        return "\n".join(lines)
    lines.append("  changed decisions:")
    for fast_log, deep_log in changed:
        tag = f"P{fast_log['seat']} T{fast_log['turn']}"
        lines.append(f"    {tag} dealt {' '.join(fast_log['dealt'])}")
        lines.append(f"      fast: {action_text(fast_log['action'])}")
        lines.append(f"            after {board_text(fast_log['after'])}")
        lines.append(f"      deep: {action_text(deep_log['action'])}")
        lines.append(f"            after {board_text(deep_log['after'])}")
    return "\n".join(lines)


def full_decision_summary(label: str, played: dict[str, Any]) -> str:
    lines: list[str] = [f"  {label} decisions:"]
    for item in played["logs"]:
        tag = f"P{item['seat']} T{item['turn']}"
        lines.append(f"    {tag} dealt {' '.join(item['dealt'])}")
        lines.append(f"      choose {action_text(item['action'])}")
        lines.append(f"      after  {board_text(item['after'])}")
    result = played["result"]
    lines.append(f"  {label} final:")
    for seat, board in enumerate(result.boards):
        lines.append(
            f"    P{seat}: {board_text(board)} "
            f"busted={result.busted[seat]} roy={result.royalties[seat]['total']}"
        )
    lines.append(f"    score P0={result.raw_score[0]} P1={result.raw_score[1]}")
    return "\n".join(lines)


def parse_seeds(raw: str, seed_start: int, hands: int) -> list[int]:
    if raw.strip():
        return [int(part.strip()) for part in raw.split(",") if part.strip()]
    return [seed_start + idx for idx in range(hands)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare current fast/deep local play on identical decks")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--seed-start", type=int, default=20260604)
    parser.add_argument("--hands", type=int, default=3)
    parser.add_argument(
        "--runtime-config",
        default="ai/data/hybrid_t1t2_active_20260531/t1_runtime_current_local_20260604.json",
    )
    parser.add_argument("--runtime-mode", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fast-t0-sims", type=int, default=16)
    parser.add_argument("--deep-stage-sims", default="4,12,32,96")
    parser.add_argument("--deep-stage-limits", default="80,32,12,5")
    parser.add_argument("--t2-sims", type=int, default=300)
    parser.add_argument("--engine-path", default="ai/rust_solver/target/release/prob_engine.exe")
    parser.add_argument("--show-all-decisions", action="store_true")
    opts = parser.parse_args()

    seeds = parse_seeds(opts.seeds, opts.seed_start, opts.hands)
    fast_args = namespace_from_runtime(Path(opts.runtime_config), opts.runtime_mode, opts.device)
    deep_args = namespace_from_runtime(Path(opts.runtime_config), opts.runtime_mode, opts.device)
    fast_player = CurrentRuntimePlayer(
        fast_args,
        t0_sims=opts.fast_t0_sims,
        t2_sims=opts.t2_sims,
        engine_path=opts.engine_path,
        t0_mode="mc",
    )
    deep_player = CurrentRuntimePlayer(
        deep_args,
        t0_sims=opts.fast_t0_sims,
        t2_sims=opts.t2_sims,
        engine_path=opts.engine_path,
        t0_mode="ladder",
        t0_stage_sims=opts.deep_stage_sims,
        t0_stage_limits=opts.deep_stage_limits,
    )

    comparisons: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for seed in seeds:
        fast = play_hand(seed, fast_player)
        deep = play_hand(seed, deep_player)
        comparisons.append((fast, deep))

    print("seed       | fast | deep | dP0  | bust F/D | ch | changed | elapsed F/D")
    print("-----------+------+------+------+----------+----+---------+------------")
    for fast, deep in comparisons:
        print(row_summary(fast, deep))
    for fast, deep in comparisons:
        print(detail_summary(fast, deep))
        if opts.show_all_decisions:
            print(full_decision_summary("fast", fast))
            print(full_decision_summary("deep", deep))


if __name__ == "__main__":
    main()
