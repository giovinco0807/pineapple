"""Play to T4 with the current runtime, then print exact T4 candidates."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.encoding import ALL_CARDS
from ai.engine.game_engine import GameEngine, Hand
from ai.tutor.demo_current_hand import (
    CurrentRuntimePlayer,
    action_from_candidate,
    action_text,
    board_text,
    exclude_cards,
    namespace_from_runtime,
)
from ai.tutor.exact_late import evaluate_late_position


def play_until_t4(seed: int, player: CurrentRuntimePlayer) -> tuple[Hand, list[dict[str, Any]]]:
    rng = random.Random(seed)
    deck = list(ALL_CARDS)
    rng.shuffle(deck)
    hand = Hand(deck=deck, btn=0)
    logs: list[dict[str, Any]] = []

    while hand.turn < 4:
        for seat in [0, 1]:
            dealt = list(hand.dealt_cards[seat])
            before = hand.boards[seat].copy()
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
                }
            )
        hand.deal_next_turn()
    return hand, logs


def candidate_line(index: int, candidate: dict[str, Any], dealt: list[str]) -> str:
    action = action_from_candidate(candidate["action"], dealt, turn=4)
    metrics = candidate.get("metrics") or {}
    board = candidate.get("board") or {}
    top = " ".join(board.get("top") or []) or "-"
    middle = " ".join(board.get("middle") or []) or "-"
    bottom = " ".join(board.get("bottom") or []) or "-"
    return (
        f"#{index:02d} score={float(metrics.get('score', 0.0)):.3f} "
        f"raw={float(metrics.get('raw_score', 0.0)):.3f} "
        f"roy={float(metrics.get('royalty', 0.0)):.1f} "
        f"bust={bool(metrics.get('bust_rate', 0.0))} "
        f"FL={float(metrics.get('fl_rate', 0.0)):.1f} "
        f"| {action_text(action)} "
        f"| T:{top} | M:{middle} | B:{bottom}"
    )


def show_t4_for_seat(hand: Hand, seat: int, top_n: int) -> dict[str, Any]:
    board = hand.boards[seat]
    dealt = list(hand.dealt_cards[seat])
    opponent = hand.boards[1 - seat]
    result = evaluate_late_position(
        board,
        dealt,
        4,
        opponent_board=opponent,
        exclude=exclude_cards(hand, seat),
        top_n=top_n,
        prefer_rust=False,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Show exact T4 candidates from a current-runtime hand")
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument(
        "--runtime-config",
        default="ai/data/hybrid_t1t2_active_20260531/t1_runtime_current_local_20260604.json",
    )
    parser.add_argument("--runtime-mode", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--t0-sims", type=int, default=16)
    parser.add_argument("--t0-mode", choices=["mc", "ladder"], default="mc")
    parser.add_argument("--t2-sims", type=int, default=300)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--engine-path", default="ai/rust_solver/target/release/prob_engine.exe")
    opts = parser.parse_args()

    runtime_args = namespace_from_runtime(Path(opts.runtime_config), opts.runtime_mode, opts.device)
    player = CurrentRuntimePlayer(
        runtime_args,
        t0_sims=opts.t0_sims,
        t2_sims=opts.t2_sims,
        engine_path=opts.engine_path,
        t0_mode=opts.t0_mode,
    )
    hand, logs = play_until_t4(opts.seed, player)

    print(f"seed={opts.seed} t0_mode={opts.t0_mode} t0_sims={opts.t0_sims}")
    print("")
    print("Played through T3")
    for item in logs:
        print(f"P{item['seat']} T{item['turn']} dealt {' '.join(item['dealt'])}")
        print(f"  choose {action_text(item['action'])}")
        print(f"  after  {board_text(item['after'])}")
    print("")

    print("T4 exact candidates")
    for seat in [0, 1]:
        result = show_t4_for_seat(hand, seat, opts.top_n)
        print(f"P{seat} T4 dealt {' '.join(hand.dealt_cards[seat])}")
        print(f"  before {board_text(hand.boards[seat])}")
        print(f"  opponent currently {board_text(hand.boards[1 - seat])}")
        print(f"  source={result['source']} legal_actions={result['legal_actions']}")
        for idx, candidate in enumerate(result["candidates"], start=1):
            print("  " + candidate_line(idx, candidate, hand.dealt_cards[seat]))
        best_action = action_from_candidate(result["best"]["action"], hand.dealt_cards[seat], turn=4)
        print(f"  BEST {action_text(best_action)}")
        hand.apply_action(seat, best_action)
        print(f"  applied after {board_text(hand.boards[seat])}")
        print("")

    final = GameEngine.compute_result(hand)
    print("Final after exact T4 choices")
    for seat, board in enumerate(final.boards):
        print(
            f"P{seat}: {board_text(board)} "
            f"busted={final.busted[seat]} roy={final.royalties[seat]['total']} "
            f"FL={final.fl_entry[seat]} names={final.hand_names[seat]}"
        )
    print(f"raw_score: P0={final.raw_score[0]} P1={final.raw_score[1]}")
    print(f"line_results(P0): {final.line_results} scoop={final.scoop}")


if __name__ == "__main__":
    main()
