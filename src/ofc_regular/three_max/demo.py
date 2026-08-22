"""Print a 3-max hand and a short finite-stack session.

    python -m ofc_regular.three_max.demo --hands 3 --seed 42 --stacks 200

The policies are uniform random: this shows the game core running end to end,
not any strength.
"""

from __future__ import annotations

import argparse

from ..state import Board
from .play import play_session, uniform_random_policy
from .scoring import DEFAULT_FL_EV_PER_PAIR, fl_ev_hand_total
from .seating import ACT_ORDER, PLAYER_COUNT, Seat3


def _format_board(board: Board) -> str:
    return " | ".join(
        " ".join(row) for row in (board.top, board.middle, board.bottom)
    )


def _format_seat(seat: Seat3, player: int) -> str:
    return f"{seat:<6} (p{player})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stacks",
        type=float,
        default=None,
        help="starting chips per player; omit for the infinite stacks the AI assumes",
    )
    args = parser.parse_args()

    starting = (
        {player: args.stacks for player in range(PLAYER_COUNT)}
        if args.stacks is not None
        else None
    )
    hands = play_session(
        base_seed=args.seed,
        policies_by_player={p: uniform_random_policy for p in range(PLAYER_COUNT)},
        hands=args.hands,
        button_player=0,
        starting_stacks=starting,
    )

    print(
        f"fl_ev per pair {DEFAULT_FL_EV_PER_PAIR}  "
        f"hand total {fl_ev_hand_total(DEFAULT_FL_EV_PER_PAIR)}"
    )
    for hand in hands:
        settlement = hand.result.settlement
        players = {seat: player for player, seat in hand.seats.items()}
        print(f"\n=== hand {hand.hand_index}  button=p{hand.button_player} ===")
        for seat in ACT_ORDER:
            board = hand.result.boards[seat]
            print(
                f"  {_format_seat(seat, players[seat])}  {_format_board(board)}"
                f"   raw {settlement.raw_totals[seat]:+.1f}"
            )
        for pair in settlement.pairs:
            note = "" if not pair.capped else "  (capped by stack)"
            print(
                f"    {pair.seats[0]:>6} vs {pair.seats[1]:<6} "
                f"raw {pair.raw:+.1f}  transferred {pair.transferred:+.1f}{note}"
            )
        if hand.stacks_after is not None:
            chips = "  ".join(
                f"p{player}={value:.1f}" for player, value in sorted(hand.stacks_after.items())
            )
            print(f"    stacks: {chips}")
        total = sum(settlement.transferred_totals.values())
        print(f"    zero-sum check: {total:+.1f}")


if __name__ == "__main__":
    main()
