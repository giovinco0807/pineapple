"""
OFC Pineapple - Self-Play Test

Random AI vs Random AI using the headless game engine.
Validates the full game loop: dealing, action selection, scoring.
"""
import sys
import random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.engine.encoding import Board, Observation
from ai.engine.action_space import get_initial_actions, get_turn_actions
from ai.engine.game_engine import GameEngine, Hand, Session, HandResult


def play_hand(hand: Hand, verbose: bool = True) -> HandResult:
    """Play one hand with random action selection."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"  New Hand (BTN=P{hand.btn})")
        print(f"{'='*60}")

    # Turn 0: place 5 cards
    for seat in [hand.btn, 1 - hand.btn]:
        cards = hand.dealt_cards[seat]
        actions = get_initial_actions(cards, hand.boards[seat])
        if not actions:
            print(f"  [ERROR] No valid actions for seat {seat}!")
            continue
        action = random.choice(actions)
        hand.apply_action(seat, action)
        if verbose:
            print(f"  P{seat} dealt: {cards}")
            placed = {pos: [c for c, p in action.placements if p == pos] for pos in ['top','middle','bottom']}
            print(f"    -> Top:{placed['top']} Mid:{placed['middle']} Bot:{placed['bottom']}")

    # Turns 1-8
    for turn_num in range(1, 9):
        if hand.is_hand_complete():
            break
        hand.deal_next_turn()
        if verbose:
            print(f"\n  --- Turn {turn_num} ---")

        for seat in [hand.btn, 1 - hand.btn]:
            cards = hand.dealt_cards[seat]
            if not cards:
                continue
            actions = get_turn_actions(cards, hand.boards[seat])
            if not actions:
                if verbose:
                    print(f"  P{seat}: No valid actions (board full?)")
                continue
            action = random.choice(actions)
            hand.apply_action(seat, action)
            if verbose:
                print(f"  P{seat} dealt: {cards} -> discard={action.discard}")
                for c, p in action.placements:
                    print(f"    {c} -> {p}")

    # Score
    result = GameEngine.compute_result(hand)

    if verbose:
        print(f"\n  {'='*40}")
        print(f"  RESULT:")
        for seat in [0, 1]:
            b = result.boards[seat]
            names = result.hand_names[seat]
            r = result.royalties[seat]
            bust = "[BUST]" if result.busted[seat] else ""
            fl = "[FL!]" if result.fl_entry[seat] else ""
            print(f"  P{seat}: {bust} {fl}")
            print(f"    Top:    {str(b.top):30s} {names['top']:15s} R={r['top']}")
            print(f"    Middle: {str(b.middle):30s} {names['middle']:15s} R={r['middle']}")
            print(f"    Bottom: {str(b.bottom):30s} {names['bottom']:15s} R={r['bottom']}")
            print(f"    Total Royalty: {r['total']}")

        lines = result.line_results
        print(f"\n  Lines: Top={lines[0]:+d} Mid={lines[1]:+d} Bot={lines[2]:+d}")
        print(f"  Scoop: {'YES!' if result.scoop else 'No'}")
        print(f"  Score: P0={result.raw_score[0]:+d} P1={result.raw_score[1]:+d}")

    return result


def play_session(num_hands: int = 5, verbose: bool = True):
    """Play a full session of multiple hands."""
    session = Session(chips=[200, 200], max_hands=num_hands)

    print(f"{'#'*60}")
    print(f"  OFC Pineapple Self-Play Test")
    print(f"  {num_hands} hands, starting chips: {session.chips}")
    print(f"{'#'*60}")

    stats = {"busts": [0, 0], "fl_entries": [0, 0], "hands": 0}

    while not session.is_finished() and session.hand_count < num_hands:
        hand = session.new_hand()
        result = play_hand(hand, verbose=verbose)
        session.apply_result(result)

        for s in [0, 1]:
            if result.busted[s]:
                stats["busts"][s] += 1
            if result.fl_entry[s]:
                stats["fl_entries"][s] += 1
        stats["hands"] += 1

        if verbose:
            print(f"\n  Chips: P0={session.chips[0]} P1={session.chips[1]}")

    print(f"\n{'#'*60}")
    print(f"  SESSION COMPLETE")
    print(f"  Hands played: {stats['hands']}")
    print(f"  Final chips: P0={session.chips[0]} P1={session.chips[1]}")
    print(f"  Busts: P0={stats['busts'][0]} P1={stats['busts'][1]}")
    print(f"  FL entries: P0={stats['fl_entries'][0]} P1={stats['fl_entries'][1]}")
    winner = "P0" if session.chips[0] > session.chips[1] else "P1" if session.chips[1] > session.chips[0] else "DRAW"
    print(f"  Winner: {winner}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hands", type=int, default=5, help="Number of hands")
    parser.add_argument("--quiet", action="store_true", help="Less output")
    args = parser.parse_args()

    random.seed(42)
    play_session(num_hands=args.hands, verbose=not args.quiet)
