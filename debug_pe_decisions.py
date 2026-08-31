"""
Debug prob_engine decisions turn by turn.
Deals a random hand, shows all candidates ranked by EV at each turn.

Usage:
    python debug_pe_decisions.py                # random hand
    python debug_pe_decisions.py --seed 42      # reproducible
    python debug_pe_decisions.py --seed 42 -n 3 # 3 hands
"""
import sys
import random
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ai.engine.encoding import ALL_CARDS, Board
from ai.engine.game_engine import Hand, GameEngine
from ai.engine.action_space import get_initial_actions, get_turn_actions
from ai.prob_engine_wrapper import evaluate_candidates, evaluate_mc_t0, evaluate_mc_candidates


def fmt_placements(placements):
    """Format placements as compact string."""
    parts = []
    for card, row in placements:
        r = row[0].upper()  # T/M/B
        parts.append(f"{card}>{r}")
    return " ".join(parts)


def show_candidates_t0(dealt, exclude, mc_sims=10):
    """Show T0 candidates via MC simulation."""
    print(f"\n  T0: dealt = {dealt}")
    print(f"  exclude = {exclude}")

    result = evaluate_mc_t0(dealt=dealt, exclude=exclude, sims=mc_sims)
    candidates = result.get("candidates", [])

    print(f"  {len(candidates)} candidates ({mc_sims} sims each)")
    print(f"  {'#':>3}  {'Score':>7}  {'Bust%':>5}  {'FL%':>5}  {'Placement':<40}  {'Discard'}")
    print(f"  {'-'*3}  {'-'*7}  {'-'*5}  {'-'*5}  {'-'*40}  {'-'*7}")

    for i, c in enumerate(candidates[:20]):
        mc = c["mc"]
        place_str = fmt_placements(c["placements"])
        disc = c.get("discard", "-")
        print(f"  {i+1:3d}  {mc['avg_score']:+7.2f}  {mc['bust_rate']*100:5.1f}  "
              f"{mc['fl_rate']*100:5.1f}  {place_str:<40}  {disc}")

    if len(candidates) > 20:
        print(f"  ... ({len(candidates) - 20} more)")

    return candidates[0] if candidates else None


def show_candidates_t1plus(top, mid, bot, dealt, exclude, turn, mc_sims=0):
    """Show T1+ candidates. If mc_sims > 0, use MC simulation instead of exact."""
    print(f"\n  T{turn}: dealt = {dealt}")
    print(f"  Board: Top={top}  Mid={mid}  Bot={bot}")
    print(f"  exclude = {exclude}")

    if mc_sims > 0:
        result = evaluate_mc_candidates(
            top=top, mid=mid, bot=bot, dealt=dealt,
            exclude=exclude, turn=turn, sims=mc_sims,
        )
        candidates = result.get("candidates", [])
        elapsed = result.get("elapsed_ms", 0)

        print(f"  {len(candidates)} candidates (MC {mc_sims} sims, {elapsed}ms)")
        print(f"  {'#':>3}  {'Score':>7}  {'Bust%':>5}  {'FL%':>5}  {'Placement':<40}  {'Discard'}")
        print(f"  {'-'*3}  {'-'*7}  {'-'*5}  {'-'*5}  {'-'*40}  {'-'*7}")

        for i, c in enumerate(candidates[:15]):
            mc = c["mc"]
            place_str = fmt_placements(c["placements"])
            disc = c.get("discard", "-")
            print(f"  {i+1:3d}  {mc['avg_score']:+7.2f}  {mc['bust_rate']*100:5.1f}  "
                  f"{mc['fl_rate']*100:5.1f}  {place_str:<40}  {disc}")
    else:
        result = evaluate_candidates(
            top=top, mid=mid, bot=bot, dealt=dealt,
            exclude=exclude, turn=turn, position="bb"
        )
        candidates = result.get("candidates", [])

        print(f"  {len(candidates)} candidates (exact)")
        print(f"  {'#':>3}  {'EV':>7}  {'Bust%':>5}  {'FL%':>5}  {'Roy':>5}  {'Placement':<40}  {'Discard'}")
        print(f"  {'-'*3}  {'-'*7}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*40}  {'-'*7}")

        for i, c in enumerate(candidates[:15]):
            place_str = fmt_placements(c["placements"])
            disc = c.get("discard", "-")
            print(f"  {i+1:3d}  {c['ev']:+7.2f}  {c['bust_prob']*100:5.1f}  "
                  f"{c['fl_rate']*100:5.1f}  {c['expected_royalty']:5.2f}  "
                  f"{place_str:<40}  {disc}")

    if len(candidates) > 15:
        print(f"  ... ({len(candidates) - 15} more)")

    return candidates[0] if candidates else None


def play_and_debug(deck, mc_sims=10):
    """Play one hand showing all decisions."""
    hand = Hand(deck=list(deck), btn=0)
    hero = 0
    opp = 1

    print("=" * 80)
    print(f"  Hand Debug - prob_engine decisions")
    print("=" * 80)

    # T0 - opponent plays first (BC greedy would be used, but we skip for debug)
    # For simplicity, opponent just gets dealt and we apply a simple placement
    # Hero's T0
    hero_obs = hand.get_observation(hero)
    dealt_hero = list(hero_obs.dealt_cards)

    # Opponent's cards as exclude
    opp_obs = hand.get_observation(opp)
    opp_dealt = list(opp_obs.dealt_cards)

    # For hero T0, exclude = opponent's dealt cards (they're on opp board after T0)
    # But at T0, opponent hasn't placed yet. So exclude is empty for first player.
    # Actually in the game, BTN acts first, then BB. Hero is seat 0 = BTN.
    # After BTN places, BB can see BTN's board.

    # Let's just show hero's candidates at T0 (no exclude yet)
    exclude = []
    best_t0 = show_candidates_t0(dealt_hero, exclude, mc_sims=mc_sims)

    if not best_t0:
        print("  No candidates!")
        return

    # Apply T0 for both players (use prob_engine best for hero, first valid for opp)
    hero_actions = get_initial_actions(hero_obs.dealt_cards, hero_obs.board_self)
    opp_actions = get_initial_actions(opp_obs.dealt_cards, opp_obs.board_self)

    # Find matching action for hero
    best_placements = [(c, p) for c, p in best_t0["placements"]]
    best_discard = best_t0.get("discard", "-")
    hero_action = None
    for a in hero_actions:
        a_placements = [(c, p) for c, p in a.placements]
        a_disc = a.discard or "-"
        # Match by placements (card, row)
        if set((c, p) for c, p in a_placements) == set((c, p) for c, p in best_placements):
            hero_action = a
            break
    if hero_action is None:
        # Joker mapping: Xj -> X1/X2
        # Just use first action as fallback
        hero_action = hero_actions[0]
        print(f"  WARNING: couldn't match T0 action, using first valid")

    hand.apply_action(hero, hero_action)
    hand.apply_action(opp, opp_actions[0])  # Opponent: first valid (arbitrary)

    # T1-T4
    for turn_num in range(1, 5):
        if hand.is_hand_complete():
            break
        hand.deal_next_turn()

        for seat in [hand.btn, 1 - hand.btn]:
            cards = hand.dealt_cards[seat]
            if not cards or hand.boards[seat].is_complete():
                continue

            if seat != hero:
                # Opponent: apply first valid action
                obs = hand.get_observation(seat)
                actions = get_turn_actions(obs.dealt_cards, obs.board_self)
                if actions:
                    hand.apply_action(seat, actions[0])
                continue

            # Hero: show all candidates
            obs = hand.get_observation(seat)
            top = list(obs.board_self.top)
            mid = list(obs.board_self.middle)
            bot = list(obs.board_self.bottom)
            dealt = list(obs.dealt_cards)
            cc = obs.board_self.card_count()

            # Exclude: opponent board + hero discards
            exclude = []
            exclude.extend(obs.board_opponent.top)
            exclude.extend(obs.board_opponent.middle)
            exclude.extend(obs.board_opponent.bottom)
            exclude.extend(obs.known_discards_self)

            if cc == 11:
                print(f"\n  T4 (last turn): dealt = {dealt}")
                print(f"  Board: Top={top}  Mid={mid}  Bot={bot}")
                print(f"  -> Exhaustive evaluation (all placements scored directly)")

            best = show_candidates_t1plus(top, mid, bot, dealt, exclude, turn_num, mc_sims=mc_sims)

            if best:
                # Apply best action
                best_placements = [(c, p) for c, p in best["placements"]]
                best_disc = best.get("discard", "-")
                actions = get_turn_actions(obs.dealt_cards, obs.board_self)
                applied = False
                for a in actions:
                    a_set = set((c, p) for c, p in a.placements)
                    b_set = set((c, p) for c, p in best_placements)
                    if a_set == b_set:
                        hand.apply_action(hero, a)
                        applied = True
                        break
                if not applied and actions:
                    hand.apply_action(hero, actions[0])
                    print(f"  WARNING: couldn't match action, using first valid")

    # Final result
    result = GameEngine.compute_result(hand)
    hero_board = hand.boards[hero]
    print(f"\n  {'-'*76}")
    print(f"  Final Board:")
    print(f"    Top: {list(hero_board.top)}")
    print(f"    Mid: {list(hero_board.middle)}")
    print(f"    Bot: {list(hero_board.bottom)}")
    print(f"  Busted: {result.busted[hero]}")
    print(f"  FL entry: {result.fl_entry[hero]}")
    print(f"  Royalty: {result.royalties[hero]['total']}")
    print(f"  Score: {result.raw_score[hero]:+.1f}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("-n", type=int, default=1, help="Number of hands to debug")
    parser.add_argument("--mc-sims", type=int, default=10, help="MC sims for T0 (default: 10)")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    for i in range(args.n):
        deck = list(ALL_CARDS)
        rng.shuffle(deck)
        if args.n > 1:
            print(f"\n{'#'*80}")
            print(f"  HAND {i+1}/{args.n}")
            print(f"{'#'*80}")
        play_and_debug(deck, mc_sims=args.mc_sims)


if __name__ == "__main__":
    main()
