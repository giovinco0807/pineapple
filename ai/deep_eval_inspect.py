"""
OFC Pineapple - Deep Evaluation Inspector

Play games with deep rollout evaluation at every turn, displaying detailed
per-action statistics (avg_score, bust_prob, fl_prob, FL type distribution)
in a human-readable format for manual verification.

Usage:
    python ai/deep_eval_inspect.py --games 1 --rollouts 2000 --seed 42
    python ai/deep_eval_inspect.py --games 3 --rollouts 500 --seed 42   # faster test
"""

import sys
import time
import random
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import os
sys.path.insert(0, os.getcwd())

import torch
from ai.engine.game_engine import (
    Hand, evaluate_hand, check_fl_entry,
    get_top_royalty, get_middle_royalty, get_bottom_royalty,
)
from ai.engine.encoding import encode_state, ALL_CARDS, Board, Observation
from ai.engine.action_space import get_initial_actions, get_turn_actions, create_action_mask
from ai.models.networks import PolicyNetwork, ValueNetwork
from ai.mcts.rollout_evaluator import RolloutEvaluator


def format_card(card):
    """Format card for display: 'As' -> 'As', 'X1' -> 'Jo'"""
    if card.startswith('X'):
        return 'Jo'
    return card


def format_cards(cards):
    """Format a list of cards."""
    return ' '.join(format_card(c) for c in cards) if cards else '(empty)'


def classify_action_fl(action, board_top):
    """Classify an action's FL intent."""
    top_placements = [(c, p) for c, p in action.placements if p == 'top']
    if not top_placements:
        # Check if existing top has FL potential
        if board_top:
            return 'top保護'
        return 'non-FL'

    for card, _ in top_placements:
        if card.startswith('X'):
            return 'FL: Jo→top'
        rank = card[:-1] if len(card) >= 2 else card[0]
        if rank == 'A':
            return 'FL: AA狙い'
        elif rank == 'K':
            return 'FL: KK狙い'
        elif rank == 'Q':
            return 'FL: QQ狙い'
    return 'non-FL'


def format_action(action, turn):
    """Format an action as human-readable text."""
    parts = []
    for card, pos in action.placements:
        parts.append(f"{format_card(card)}→{pos}")
    if action.discard:
        parts.append(f"捨:{format_card(action.discard)}")
    return ', '.join(parts)


def format_fl_type_dist(fl_type_dist, n_rollouts):
    """Format FL type distribution."""
    if not fl_type_dist:
        return ''
    # Map card counts to FL type names
    type_names = {14: 'QQ', 15: 'KK', 16: 'AA', 17: 'Trips'}
    parts = []
    for cards, count in sorted(fl_type_dist.items(), key=lambda x: -x[1]):
        name = type_names.get(cards, f'{cards}c')
        pct = count / n_rollouts * 100
        parts.append(f"{name}:{pct:.1f}%")
    return ' '.join(parts)


def bc_greedy_select(policy_net, obs, actions=None):
    """BC greedy action selection (for opponent)."""
    with torch.no_grad():
        state = torch.FloatTensor(encode_state(obs)).unsqueeze(0)
        if actions is None:
            actions = get_turn_actions(obs.dealt_cards, obs.board_self)
        if not actions:
            return None
        if len(actions) == 1:
            return actions[0]
        mask = create_action_mask(actions)
        mask_t = torch.BoolTensor(mask).unsqueeze(0)
        probs = policy_net(state, mask_t)
        idx = torch.argmax(probs, dim=-1).item()
        return actions[idx] if idx < len(actions) else actions[0]


def direct_eval_last_turn(obs):
    """Deterministic last-turn evaluation."""
    actions = get_turn_actions(obs.dealt_cards, obs.board_self)
    if not actions:
        return None
    if len(actions) == 1:
        return actions[0]
    best_score = float('-inf')
    best_act = actions[0]
    for act in actions:
        board = obs.board_self.copy()
        for card, pos in act.placements:
            getattr(board, pos).append(card)
        top_v = evaluate_hand(board.top, 3)
        mid_v = evaluate_hand(board.middle, 5)
        bot_v = evaluate_hand(board.bottom, 5)
        if top_v > mid_v or mid_v > bot_v:
            score = -100
        else:
            score = (get_top_royalty(board.top) +
                     get_middle_royalty(board.middle) +
                     get_bottom_royalty(board.bottom))
        if score > best_score:
            best_score = score
            best_act = act
    return best_act


def play_game_deep(game_idx, rollout_eval, policy_net, n_rollouts, seed, top_n=0):
    """Play one game with deep evaluation at every hero turn."""
    rng = random.Random(seed)
    deck = list(ALL_CARDS)
    rng.shuffle(deck)

    hand = Hand(deck=list(deck), btn=0)
    hero_seat = 0

    print(f"\n{'='*70}")
    print(f"  Game {game_idx + 1} (seed={seed})")
    print(f"{'='*70}")

    # T0: both players act
    for seat in [hand.btn, 1 - hand.btn]:
        obs = hand.get_observation(seat)
        if seat == hero_seat:
            if obs.turn == 0:
                valid_actions = get_initial_actions(obs.dealt_cards, obs.board_self)
            else:
                valid_actions = get_turn_actions(obs.dealt_cards, obs.board_self)

            board = obs.board_self
            card_count = len(board.top) + len(board.middle) + len(board.bottom)

            print(f"\n--- Turn 0 ({len(obs.dealt_cards)} cards: {format_cards(obs.dealt_cards)}) "
                  f"--- [全{len(valid_actions)}候補]")
            print(f"Board: Top[{format_cards(board.top)}] "
                  f"Mid[{format_cards(board.middle)}] "
                  f"Bot[{format_cards(board.bottom)}]")

            # Deep evaluate ALL candidates
            action_stats = []
            t0 = time.time()
            for i, action in enumerate(valid_actions):
                stats = rollout_eval._evaluate_action_detailed(obs, action, n_rollouts)
                stats['action_idx'] = i
                stats['action'] = action
                action_stats.append(stats)

                # Progress indicator for large action spaces
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - t0
                    print(f"    ... {i+1}/{len(valid_actions)} candidates evaluated "
                          f"({elapsed:.0f}s)", flush=True)

            eval_time = time.time() - t0

            # Sort by avg_score descending
            action_stats.sort(key=lambda x: x['avg_score'], reverse=True)

            # Display results
            print()
            display_n = top_n if top_n > 0 else len(action_stats)
            for rank, stats in enumerate(action_stats[:display_n]):
                action = stats['action']
                fl_class = classify_action_fl(action, board.top)
                fl_dist_str = format_fl_type_dist(stats['fl_type_dist'], n_rollouts)
                fl_detail = f" ({fl_dist_str})" if fl_dist_str else ""

                marker = ">>>" if rank == 0 else "   "
                print(f"  {marker} #{rank+1:2d}  {format_action(action, 0):50s}  [{fl_class}]")
                print(f"        r={n_rollouts}: avg={stats['avg_score']:+.2f} "
                      f"\u00b1{stats['std_score']:.1f} | "
                      f"bust={stats['bust_prob']*100:.1f}% | "
                      f"FL={stats['fl_prob']*100:.1f}%{fl_detail} | "
                      f"roy={stats['avg_royalty']:.1f}")
            if top_n > 0 and len(action_stats) > top_n:
                print(f"  ... ({len(action_stats) - top_n} more actions omitted)")

            print(f"\n  [{eval_time:.1f}s] BEST: #{1} "
                  f"(avg={action_stats[0]['avg_score']:+.2f})")

            # Select best action
            best_action = action_stats[0]['action']
            hand.apply_action(seat, best_action)
        else:
            va = get_initial_actions(obs.dealt_cards, obs.board_self)
            action = bc_greedy_select(policy_net, obs, actions=va)
            if action:
                hand.apply_action(seat, action)

    # T1+
    for turn_num in range(1, 9):
        if hand.is_hand_complete():
            break
        hand.deal_next_turn()
        for seat in [hand.btn, 1 - hand.btn]:
            cards = hand.dealt_cards[seat]
            if not cards or hand.boards[seat].is_complete():
                continue
            obs = hand.get_observation(seat)
            board = obs.board_self
            card_count = len(board.top) + len(board.middle) + len(board.bottom)

            if seat == hero_seat:
                valid_actions = get_turn_actions(obs.dealt_cards, obs.board_self)

                print(f"\n--- Turn {turn_num} ({len(obs.dealt_cards)} cards: "
                      f"{format_cards(obs.dealt_cards)}, discard 1) "
                      f"--- [全{len(valid_actions)}候補]")
                print(f"Board: Top[{format_cards(board.top)}] "
                      f"Mid[{format_cards(board.middle)}] "
                      f"Bot[{format_cards(board.bottom)}]")

                if card_count == 11:
                    # Last turn: deterministic
                    action = direct_eval_last_turn(obs)
                    print(f"  (最終ターン: 直接評価)")
                    if action:
                        print(f"  >>> {format_action(action, turn_num)}")
                    hand.apply_action(seat, action)
                    continue

                # Deep evaluate ALL candidates
                action_stats = []
                t0 = time.time()
                for i, action in enumerate(valid_actions):
                    stats = rollout_eval._evaluate_action_detailed(obs, action, n_rollouts)
                    stats['action_idx'] = i
                    stats['action'] = action
                    action_stats.append(stats)

                    if (i + 1) % 10 == 0:
                        elapsed = time.time() - t0
                        print(f"    ... {i+1}/{len(valid_actions)} candidates evaluated "
                              f"({elapsed:.0f}s)", flush=True)

                eval_time = time.time() - t0

                action_stats.sort(key=lambda x: x['avg_score'], reverse=True)

                print()
                for rank, stats in enumerate(action_stats):
                    action = stats['action']
                    fl_class = classify_action_fl(action, board.top)
                    fl_dist_str = format_fl_type_dist(stats['fl_type_dist'], n_rollouts)
                    fl_detail = f" ({fl_dist_str})" if fl_dist_str else ""

                    marker = ">>>" if rank == 0 else "   "
                    print(f"  {marker} #{rank+1:2d}  {format_action(action, turn_num):50s}  [{fl_class}]")
                    print(f"        r={n_rollouts}: avg={stats['avg_score']:+.2f} "
                          f"\u00b1{stats['std_score']:.1f} | "
                          f"bust={stats['bust_prob']*100:.1f}% | "
                          f"FL={stats['fl_prob']*100:.1f}%{fl_detail} | "
                          f"roy={stats['avg_royalty']:.1f}")

                print(f"\n  [{eval_time:.1f}s] BEST: #{1} "
                      f"(avg={action_stats[0]['avg_score']:+.2f})")

                best_action = action_stats[0]['action']
                hand.apply_action(seat, best_action)
            else:
                if card_count == 11:
                    action = direct_eval_last_turn(obs)
                else:
                    action = bc_greedy_select(policy_net, obs)
                if action:
                    hand.apply_action(seat, action)

    # Final result
    print(f"\n{'='*70}")
    print(f"  Game Result")
    print(f"{'='*70}")

    hero_board = hand.boards[hero_seat]
    opp_board = hand.boards[1 - hero_seat]

    print(f"  Hero:  Top[{format_cards(hero_board.top)}] "
          f"Mid[{format_cards(hero_board.middle)}] "
          f"Bot[{format_cards(hero_board.bottom)}]")
    print(f"  Opp:   Top[{format_cards(opp_board.top)}] "
          f"Mid[{format_cards(opp_board.middle)}] "
          f"Bot[{format_cards(opp_board.bottom)}]")

    # Evaluate final boards
    hero_tv = evaluate_hand(hero_board.top, 3)
    hero_mv = evaluate_hand(hero_board.middle, 5)
    hero_bv = evaluate_hand(hero_board.bottom, 5)
    hero_busted = hero_tv > hero_mv or hero_mv > hero_bv

    hero_royalty = 0
    if not hero_busted:
        hero_royalty = (get_top_royalty(hero_board.top) +
                        get_middle_royalty(hero_board.middle) +
                        get_bottom_royalty(hero_board.bottom))

    fl_cards = RolloutEvaluator._check_fl_cards(hero_board.top)
    fl_entry = not hero_busted and fl_cards > 0

    score = RolloutEvaluator._compute_score(hero_board, opp_board)

    fl_type_names = {14: 'QQ', 15: 'KK', 16: 'AA', 17: 'Trips'}

    print(f"\n  Busted: {'YES' if hero_busted else 'No'}")
    if fl_entry:
        print(f"  FL Entry: Yes ({fl_type_names.get(fl_cards, '?')}, {fl_cards} cards)")
    else:
        print(f"  FL Entry: No")
    print(f"  Royalty: {hero_royalty}")
    print(f"  Score vs opponent: {score:+.1f}")

    return {
        'busted': hero_busted,
        'fl_entry': fl_entry,
        'fl_cards': fl_cards,
        'royalty': hero_royalty,
        'score': score,
    }


def main():
    parser = argparse.ArgumentParser(description="Deep Evaluation Inspector")
    parser.add_argument('--games', type=int, default=1)
    parser.add_argument('--rollouts', type=int, default=2000,
                        help='Rollouts per action (higher = more accurate)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--top-n', type=int, default=0,
                        help='Only show top N actions per turn (0=all)')
    parser.add_argument('--model', default='ai/models/selfplay_iter17/bc_policy_best.pt')
    parser.add_argument('--vn-model', default='ai/models/value_v1/value_best.pt')
    args = parser.parse_args()

    print("=" * 70)
    print("  OFC Pineapple - Deep Evaluation Inspector")
    print("=" * 70)
    print(f"  Policy: {args.model}")
    print(f"  Rollouts per action: {args.rollouts}")
    print(f"  Games: {args.games}")
    print(f"  Seed: {args.seed}")

    # Load policy
    policy_net = PolicyNetwork()
    ck = torch.load(args.model, map_location='cpu', weights_only=False)
    policy_net.load_state_dict(
        ck['model_state_dict'] if 'model_state_dict' in ck else ck)
    policy_net.eval()

    # Create rollout evaluator (rule playout, no VN prefilter for deep eval)
    rollout_eval = RolloutEvaluator(
        policy_net=policy_net, n_rollouts=args.rollouts, top_k=999,
        device='cpu',
    )

    total_t0 = time.time()
    results = []
    for i in range(args.games):
        game_seed = args.seed + i * 7919
        r = play_game_deep(i, rollout_eval, policy_net, args.rollouts, game_seed, args.top_n)
        results.append(r)

    total_time = time.time() - total_t0

    # Summary
    print(f"\n{'='*70}")
    print(f"  Summary ({args.games} games, {total_time:.0f}s total)")
    print(f"{'='*70}")
    n = len(results)
    busts = sum(1 for r in results if r['busted'])
    fls = sum(1 for r in results if r['fl_entry'])
    scores = [r['score'] for r in results]
    print(f"  Bust rate: {busts}/{n} ({busts/n*100:.0f}%)")
    print(f"  FL rate:   {fls}/{n} ({fls/n*100:.0f}%)")
    print(f"  Avg score: {sum(scores)/n:+.2f}")
    print(f"  Speed:     {total_time/n:.1f}s/game")


if __name__ == '__main__':
    main()
