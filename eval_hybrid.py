"""
OFC Pineapple - Hybrid Inference v2 with FL Round Play

Combines PPO (non-FL hands) with BC+RolloutEvaluator (FL-capable hands).
Includes full Fantasyland round play with Rust FL solver and FL re-entry chains.

Strategy:
  T0: If hand has 2+ Q/K/A/Joker → RolloutEval, else PPO
  T1+: If FL path active → RolloutEval for all turns
       Otherwise → PPO

Usage:
    python eval_hybrid.py --games 50
"""
import sys
import random
import argparse
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np

from ai.engine.encoding import Board, Observation, ALL_CARDS, encode_state
from ai.engine.action_space import (
    get_initial_actions, get_turn_actions, create_action_mask, MAX_ACTIONS
)
from ai.engine.scoring import check_fl_stay_from_cards
from ai.engine.game_engine import (
    GameEngine, Hand, evaluate_hand, hand_category, _B5,
    get_top_royalty, get_middle_royalty, get_bottom_royalty,
    check_fl_entry, RANK_VALUES,
)
from ai.models.networks import PolicyNetwork
from ai.mcts.rollout_evaluator import RolloutEvaluator
from ai.rust_solver_wrapper import RustFLSolver


# ─── Card format conversion ─────────────────────────────────────────

SUIT_MAP = {'s': 0, 'h': 1, 'd': 2, 'c': 3}

def card_str_to_rust(card_str):
    """Convert card string (e.g. 'Ah', 'X1') to Rust solver format (rank, suit)."""
    if card_str.startswith('X'):
        return (0, 4)  # Joker
    rank_char = card_str[:-1]
    suit_char = card_str[-1]
    rank = RANK_VALUES.get(rank_char, 0)
    suit = SUIT_MAP.get(suit_char, 0)
    return (rank, suit)


def rust_card_to_str(rank, suit):
    """Convert Rust solver card back to string format."""
    if suit == 4 or rank == 0:
        return 'X1'  # Joker (simplified)
    rank_map = {2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8',
                9: '9', 10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A'}
    suit_map = {0: 's', 1: 'h', 2: 'd', 3: 'c'}
    return rank_map.get(rank, '?') + suit_map.get(suit, '?')


# ─── FL potential detection ──────────────────────────────────────────

FL_HIGH_RANKS = {'Q', 'K', 'A'}

def _is_fl_high(card_str):
    if card_str.startswith('X'):
        return True
    rank = card_str[:-1]
    return rank in FL_HIGH_RANKS

def has_fl_potential_t0(dealt_cards):
    """T0: Does this 5-card hand have FL potential? (2+ Q/K/A/Joker)"""
    return sum(1 for c in dealt_cards if _is_fl_high(c)) >= 2


# ─── FL Stay Check (delegated to engine) ─────────────────────────────

def check_fl_stay(top_cards, bottom_cards, current_fl_cards=17, middle_cards=None):
    """Check FL Stay conditions. Delegates to ai.engine.scoring."""
    return check_fl_stay_from_cards(
        top_cards,
        bottom_cards,
        current_fl_cards,
        middle_cards=middle_cards,
    )


# ─── FL Round Play ───────────────────────────────────────────────────

def play_fl_round(fl_solver, fl_card_count, available_cards):
    """Play a Fantasyland round using Rust FL solver.
    
    Args:
        fl_solver: RustFLSolver instance
        fl_card_count: Number of cards to deal (14-17)
        available_cards: List of card strings available for this round
    
    Returns:
        (royalty, top_strs, mid_strs, bot_strs, discard_strs, can_stay, next_fl_cards)
    """
    if len(available_cards) < fl_card_count:
        return 0, [], [], [], [], False, 0
    
    dealt = available_cards[:fl_card_count]
    rust_cards = [card_str_to_rust(c) for c in dealt]
    result = fl_solver.solve(rust_cards)
    
    if result is None:
        return 0, [], [], [], [], False, 0
    
    royalty = (result.get('top_royalty', 0) +
               result.get('middle_royalty', 0) +
               result.get('bottom_royalty', 0))
    
    # Convert placed cards back to string format
    top_strs = [rust_card_to_str(c['rank'], c['suit']) for c in result.get('top', [])]
    mid_strs = [rust_card_to_str(c['rank'], c['suit']) for c in result.get('middle', [])]
    bot_strs = [rust_card_to_str(c['rank'], c['suit']) for c in result.get('bottom', [])]
    discard_strs = [rust_card_to_str(c['rank'], c['suit']) for c in result.get('discards', [])]
    
    # FL Stay check uses correct rules (Top Trips OR Bottom Quads+)
    can_stay, next_fl_cards = check_fl_stay(
        top_strs, bot_strs, fl_card_count, middle_cards=mid_strs
    )
    
    return royalty, top_strs, mid_strs, bot_strs, discard_strs, can_stay, next_fl_cards


def play_fl_chain(fl_solver, initial_fl_cards, deck, deck_idx):
    """Play FL rounds with re-entry chains (no round limit).
    
    Between rounds, placed cards (13) are locked but discards return to the
    available pool for the next round.  The pool is reshuffled each round.
    
    Returns:
        (total_fl_royalty, total_fl_rounds, fl_boards, cards_consumed)
        fl_boards: list of (top, mid, bot, royalty) per FL round
    """
    import random as _rand
    total_royalty = 0
    total_rounds = 0
    fl_boards = []
    fl_cards = initial_fl_cards
    
    # Build available pool from deck starting at deck_idx
    pool = list(deck[deck_idx:])
    total_consumed = len(pool)  # we "claim" all remaining cards
    
    while fl_cards > 0:
        _rand.shuffle(pool)
        royalty, top, mid, bot, discards, can_stay, new_fl_cards = play_fl_round(
            fl_solver, fl_cards, pool
        )
        if not top:  # solver failed or not enough cards
            break
        
        total_royalty += royalty
        total_rounds += 1
        fl_boards.append((top, mid, bot, royalty))
        
        # Remove PLACED cards from pool; discards return to pool
        placed = set(top + mid + bot)
        pool = [c for c in pool if c not in placed]
        
        if can_stay and new_fl_cards > 0:
            fl_cards = new_fl_cards
        else:
            break
    
    return total_royalty, total_rounds, fl_boards, total_consumed


def score_fl_vs_normal(fl_top, fl_mid, fl_bot, fl_royalty,
                       opp_top, opp_mid, opp_bot, opp_busted):
    """Score FL player's board vs opponent's normal board.
    
    OFC scoring: 3 line comparisons + scoop bonus + royalties.
    FL board is solver-optimal so never busted.
    
    Returns: score from FL player's perspective.
    """
    if opp_busted:
        # Opponent busted → FL player gets scoop + royalties
        return 6 + fl_royalty
    
    # Line comparisons
    fl_top_val = evaluate_hand(fl_top, 3)
    fl_mid_val = evaluate_hand(fl_mid, 5)
    fl_bot_val = evaluate_hand(fl_bot, 5)
    opp_top_val = evaluate_hand(opp_top, 3)
    opp_mid_val = evaluate_hand(opp_mid, 5)
    opp_bot_val = evaluate_hand(opp_bot, 5)
    
    lines_won = 0
    lines_lost = 0
    if fl_top_val > opp_top_val: lines_won += 1
    elif fl_top_val < opp_top_val: lines_lost += 1
    if fl_mid_val > opp_mid_val: lines_won += 1
    elif fl_mid_val < opp_mid_val: lines_lost += 1
    if fl_bot_val > opp_bot_val: lines_won += 1
    elif fl_bot_val < opp_bot_val: lines_lost += 1
    
    line_score = lines_won - lines_lost
    # Scoop bonus
    if lines_won == 3: line_score += 3
    elif lines_lost == 3: line_score -= 3
    
    # Royalty differential
    opp_roy = (get_top_royalty(opp_top) + get_middle_royalty(opp_mid)
               + get_bottom_royalty(opp_bot))
    
    return line_score + fl_royalty - opp_roy


def compute_fl_score(hero_fl_boards, opp_fl_boards, hero_fl_royalty, opp_fl_royalty):
    """Compute FL score when both players are in FL (both use solver).
    
    Each FL round is matched 1:1. Extra rounds get scoop + royalty.
    Returns: score from hero's perspective.
    """
    score = 0
    max_rounds = max(len(hero_fl_boards), len(opp_fl_boards))
    
    for i in range(max_rounds):
        if i < len(hero_fl_boards) and i < len(opp_fl_boards):
            h_top, h_mid, h_bot, h_roy = hero_fl_boards[i]
            o_top, o_mid, o_bot, o_roy = opp_fl_boards[i]
            # Line comparison between two FL boards
            score += score_fl_vs_normal(h_top, h_mid, h_bot, h_roy,
                                        o_top, o_mid, o_bot, False)
            # But we double-counted opponent royalty — they also have royalty
            # Actually score_fl_vs_normal subtracts opp_roy, which is correct
        elif i < len(hero_fl_boards):
            # Hero has extra FL round, opponent out
            _, _, _, h_roy = hero_fl_boards[i]
            score += 6 + h_roy  # scoop + royalty
        else:
            # Opponent has extra FL round
            _, _, _, o_roy = opp_fl_boards[i]
            score -= (6 + o_roy)
    
    return score


# ─── PPO bridge ──────────────────────────────────────────────────────

def ppo_select(ppo_model, obs, valid_actions):
    """Select action using PPO from GameEngine state."""
    if not valid_actions:
        return None
    if len(valid_actions) == 1:
        return valid_actions[0]
    state_vec = encode_state(obs)
    mask = np.zeros(MAX_ACTIONS, dtype=bool)
    mask[:len(valid_actions)] = True
    action_idx, _ = ppo_model.predict(state_vec, action_masks=mask, deterministic=True)
    action_idx = int(action_idx)
    if action_idx < len(valid_actions):
        return valid_actions[action_idx]
    return valid_actions[0]


# ─── BC greedy selector ─────────────────────────────────────────────

def bc_greedy_select(policy_net, obs, actions=None):
    if actions is None:
        actions = get_turn_actions(obs.dealt_cards, obs.board_self)
    if not actions:
        return None
    if len(actions) == 1:
        return actions[0]
    state_vec = encode_state(obs)
    state_t = torch.FloatTensor(state_vec).unsqueeze(0)
    mask = create_action_mask(actions)
    mask_t = torch.BoolTensor(mask).unsqueeze(0)
    with torch.no_grad():
        probs = policy_net(state_t, mask_t).squeeze(0).numpy()
    return actions[int(np.argmax(probs[:len(actions)]))]


# ─── Direct eval for last turn ───────────────────────────────────────

def direct_eval_last_turn(obs):
    valid_actions = get_turn_actions(obs.dealt_cards, obs.board_self)
    if not valid_actions:
        return None
    if len(valid_actions) == 1:
        return valid_actions[0]

    best_score = float("-inf")
    best_action = valid_actions[0]

    for action in valid_actions:
        top = list(obs.board_self.top)
        mid = list(obs.board_self.middle)
        bot = list(obs.board_self.bottom)
        for card, pos in action.placements:
            if pos == 'top': top.append(card)
            elif pos == 'middle': mid.append(card)
            else: bot.append(card)

        top_val = evaluate_hand(top, 3)
        mid_val = evaluate_hand(mid, 5)
        bot_val = evaluate_hand(bot, 5)
        is_busted = top_val > mid_val or mid_val > bot_val

        if is_busted:
            score = -100
        else:
            roy = (get_top_royalty(top) + get_middle_royalty(mid)
                   + get_bottom_royalty(bot))
            fl_bonus = 10 if check_fl_entry(top)[0] else 0
            score = roy + fl_bonus

        if score > best_score:
            best_score = score
            best_action = action

    return best_action


# ─── Full game with FL rounds ────────────────────────────────────────

def play_full_game_hybrid(deck, ppo_model, policy_net, rollout_eval, rollout_eval_light, fl_solver, all_rollout=False):
    """Play one complete game: normal hand + FL rounds for both seats.
    
    If all_rollout=True, hero uses RolloutEval for ALL turns (no PPO branching).
    Returns dict with aggregated results.
    """
    hand = Hand(deck=list(deck), btn=0)
    hero_seat = 0
    use_rollout_for_hero = False
    
    # --- T0 ---
    for seat in [hand.btn, 1 - hand.btn]:
        obs = hand.get_observation(seat)
        if seat == hero_seat:
            dealt = obs.dealt_cards
            if all_rollout or has_fl_potential_t0(dealt):
                use_rollout_for_hero = True
                _, action = rollout_eval.select_action(obs)
            else:
                use_rollout_for_hero = False
                va = get_initial_actions(obs.dealt_cards, obs.board_self)
                action = ppo_select(ppo_model, obs, va)
        else:
            va = get_initial_actions(obs.dealt_cards, obs.board_self)
            action = bc_greedy_select(policy_net, obs, actions=va)
        if action:
            hand.apply_action(seat, action)
    
    # --- T1+ ---
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
                if card_count == 11:
                    action = direct_eval_last_turn(obs)
                elif all_rollout or use_rollout_for_hero:
                    _, action = rollout_eval_light.select_action(obs)
                else:
                    va = get_turn_actions(obs.dealt_cards, obs.board_self)
                    action = ppo_select(ppo_model, obs, va)
            else:
                if card_count == 11:
                    action = direct_eval_last_turn(obs)
                else:
                    action = bc_greedy_select(policy_net, obs)
            if action:
                hand.apply_action(seat, action)
    
    # --- Compute normal round result ---
    result = GameEngine.compute_result(hand)
    normal_score = result.raw_score[0]  # hero's score
    
    # --- FL rounds ---
    rng = random.Random(hash(tuple(deck[:5])))
    fl_deck = list(ALL_CARDS)
    used_cards = set()
    for seat in range(2):
        for c in hand.boards[seat].top: used_cards.add(c)
        for c in hand.boards[seat].middle: used_cards.add(c)
        for c in hand.boards[seat].bottom: used_cards.add(c)
        for c in hand.discards[seat]: used_cards.add(c)
    fl_deck = [c for c in fl_deck if c not in used_cards]
    rng.shuffle(fl_deck)
    
    hero_fl_royalty = 0; hero_fl_rounds = 0; hero_fl_boards = []
    opp_fl_royalty = 0; opp_fl_rounds = 0; opp_fl_boards = []
    fl_deck_idx = 0
    
    if result.fl_entry[0] and not result.busted[0]:
        hero_fl_royalty, hero_fl_rounds, hero_fl_boards, used = play_fl_chain(
            fl_solver, result.fl_card_count[0], fl_deck, fl_deck_idx)
        fl_deck_idx += used
    if result.fl_entry[1] and not result.busted[1]:
        opp_fl_royalty, opp_fl_rounds, opp_fl_boards, used = play_fl_chain(
            fl_solver, result.fl_card_count[1], fl_deck, fl_deck_idx)
        fl_deck_idx += used
    
    # Score FL rounds with proper line comparison
    fl_score_delta = 0
    if hero_fl_rounds > 0 and opp_fl_rounds > 0:
        # Both in FL → line comparison between FL boards
        fl_score_delta = compute_fl_score(
            hero_fl_boards, opp_fl_boards, hero_fl_royalty, opp_fl_royalty)
    elif hero_fl_rounds > 0:
        # Hero in FL, opponent plays normal-level hands
        # Estimate: FL boards vs typical BC opponent → scoop likely
        for top, mid, bot, roy in hero_fl_boards:
            fl_score_delta += 6 + roy  # near-guaranteed scoop + royalties
    elif opp_fl_rounds > 0:
        for top, mid, bot, roy in opp_fl_boards:
            fl_score_delta -= (6 + roy)
    
    total_score = normal_score + fl_score_delta
    
    return {
        'busted': result.busted[0],
        'fl_entry': result.fl_entry[0],
        'royalty': result.royalties[0]['total'],
        'normal_score': normal_score,
        'fl_score': fl_score_delta,
        'total_score': total_score,
        'hero_fl_rounds': hero_fl_rounds,
        'hero_fl_royalty': hero_fl_royalty,
        'opp_fl_rounds': opp_fl_rounds,
        'opp_fl_royalty': opp_fl_royalty,
        'fl_card_count': result.fl_card_count[0] if result.fl_entry[0] else 0,
    }


def play_full_game_ppo_only(deck, ppo_model, policy_net, fl_solver):
    """Play one complete game with PPO only + FL rounds."""
    hand = Hand(deck=list(deck), btn=0)
    hero_seat = 0

    for seat in [hand.btn, 1 - hand.btn]:
        obs = hand.get_observation(seat)
        va = get_initial_actions(obs.dealt_cards, obs.board_self)
        if seat == hero_seat:
            action = ppo_select(ppo_model, obs, va)
        else:
            action = bc_greedy_select(policy_net, obs, actions=va)
        if action:
            hand.apply_action(seat, action)

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
            cc = len(board.top) + len(board.middle) + len(board.bottom)
            if seat == hero_seat:
                if cc == 11:
                    action = direct_eval_last_turn(obs)
                else:
                    va = get_turn_actions(obs.dealt_cards, obs.board_self)
                    action = ppo_select(ppo_model, obs, va)
            else:
                if cc == 11:
                    action = direct_eval_last_turn(obs)
                else:
                    action = bc_greedy_select(policy_net, obs)
            if action:
                hand.apply_action(seat, action)

    result = GameEngine.compute_result(hand)
    
    rng = random.Random(hash(tuple(deck[:5])))
    fl_deck = list(ALL_CARDS)
    used_cards = set()
    for seat in range(2):
        for c in hand.boards[seat].top: used_cards.add(c)
        for c in hand.boards[seat].middle: used_cards.add(c)
        for c in hand.boards[seat].bottom: used_cards.add(c)
        for c in hand.discards[seat]: used_cards.add(c)
    fl_deck = [c for c in fl_deck if c not in used_cards]
    rng.shuffle(fl_deck)
    
    hero_fl_royalty = 0; hero_fl_rounds = 0; hero_fl_boards = []
    opp_fl_royalty = 0; opp_fl_rounds = 0; opp_fl_boards = []
    fl_deck_idx = 0
    
    if result.fl_entry[0] and not result.busted[0]:
        hero_fl_royalty, hero_fl_rounds, hero_fl_boards, used = play_fl_chain(
            fl_solver, result.fl_card_count[0], fl_deck, fl_deck_idx)
        fl_deck_idx += used
    if result.fl_entry[1] and not result.busted[1]:
        opp_fl_royalty, opp_fl_rounds, opp_fl_boards, used = play_fl_chain(
            fl_solver, result.fl_card_count[1], fl_deck, fl_deck_idx)
        fl_deck_idx += used
    
    fl_score_delta = 0
    if hero_fl_rounds > 0 and opp_fl_rounds > 0:
        fl_score_delta = compute_fl_score(
            hero_fl_boards, opp_fl_boards, hero_fl_royalty, opp_fl_royalty)
    elif hero_fl_rounds > 0:
        for top, mid, bot, roy in hero_fl_boards:
            fl_score_delta += 6 + roy
    elif opp_fl_rounds > 0:
        for top, mid, bot, roy in opp_fl_boards:
            fl_score_delta -= (6 + roy)
    
    return {
        'busted': result.busted[0],
        'fl_entry': result.fl_entry[0],
        'royalty': result.royalties[0]['total'],
        'normal_score': result.raw_score[0],
        'fl_score': fl_score_delta,
        'total_score': result.raw_score[0] + fl_score_delta,
        'hero_fl_rounds': hero_fl_rounds,
        'hero_fl_royalty': hero_fl_royalty,
        'opp_fl_rounds': opp_fl_rounds,
        'opp_fl_royalty': opp_fl_royalty,
        'fl_card_count': result.fl_card_count[0] if result.fl_entry[0] else 0,
    }


def play_full_game_rollout(deck, policy_net, rollout_eval, rollout_eval_light, fl_solver):
    """Play one complete game with BC+RolloutEval + FL rounds."""
    hand = Hand(deck=list(deck), btn=0)

    for seat in [hand.btn, 1 - hand.btn]:
        obs = hand.get_observation(seat)
        _, action = rollout_eval.select_action(obs)
        if action:
            hand.apply_action(seat, action)

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
            cc = len(board.top) + len(board.middle) + len(board.bottom)
            if cc == 11:
                action = direct_eval_last_turn(obs)
            else:
                _, action = rollout_eval_light.select_action(obs)
            if action:
                hand.apply_action(seat, action)

    result = GameEngine.compute_result(hand)
    
    rng = random.Random(hash(tuple(deck[:5])))
    fl_deck = list(ALL_CARDS)
    used_cards = set()
    for seat in range(2):
        for c in hand.boards[seat].top: used_cards.add(c)
        for c in hand.boards[seat].middle: used_cards.add(c)
        for c in hand.boards[seat].bottom: used_cards.add(c)
        for c in hand.discards[seat]: used_cards.add(c)
    fl_deck = [c for c in fl_deck if c not in used_cards]
    rng.shuffle(fl_deck)
    
    hero_fl_royalty = 0; hero_fl_rounds = 0; hero_fl_boards = []
    opp_fl_royalty = 0; opp_fl_rounds = 0; opp_fl_boards = []
    fl_deck_idx = 0
    
    if result.fl_entry[0] and not result.busted[0]:
        hero_fl_royalty, hero_fl_rounds, hero_fl_boards, used = play_fl_chain(
            fl_solver, result.fl_card_count[0], fl_deck, fl_deck_idx)
        fl_deck_idx += used
    if result.fl_entry[1] and not result.busted[1]:
        opp_fl_royalty, opp_fl_rounds, opp_fl_boards, used = play_fl_chain(
            fl_solver, result.fl_card_count[1], fl_deck, fl_deck_idx)
        fl_deck_idx += used
    
    fl_score_delta = 0
    if hero_fl_rounds > 0 and opp_fl_rounds > 0:
        fl_score_delta = compute_fl_score(
            hero_fl_boards, opp_fl_boards, hero_fl_royalty, opp_fl_royalty)
    elif hero_fl_rounds > 0:
        for top, mid, bot, roy in hero_fl_boards:
            fl_score_delta += 6 + roy
    elif opp_fl_rounds > 0:
        for top, mid, bot, roy in opp_fl_boards:
            fl_score_delta -= (6 + roy)
    
    return {
        'busted': result.busted[0],
        'fl_entry': result.fl_entry[0],
        'royalty': result.royalties[0]['total'],
        'normal_score': result.raw_score[0],
        'fl_score': fl_score_delta,
        'total_score': result.raw_score[0] + fl_score_delta,
        'hero_fl_rounds': hero_fl_rounds,
        'hero_fl_royalty': hero_fl_royalty,
        'opp_fl_rounds': opp_fl_rounds,
        'opp_fl_royalty': opp_fl_royalty,
        'fl_card_count': result.fl_card_count[0] if result.fl_entry[0] else 0,
    }


# ─── Stats ───────────────────────────────────────────────────────────

class Stats:
    def __init__(self, label):
        self.label = label
        self.n = 0
        self.busts = 0
        self.fl_entries = 0
        self.total_royalty = 0
        self.normal_scores = []
        self.fl_scores = []
        self.total_scores = []
        self.total_hero_fl_rounds = 0
        self.total_hero_fl_royalty = 0
        self.total_opp_fl_rounds = 0

    def add(self, r):
        self.n += 1
        if r['busted']: self.busts += 1
        if r['fl_entry']: self.fl_entries += 1
        self.total_royalty += r['royalty']
        self.normal_scores.append(r['normal_score'])
        self.fl_scores.append(r['fl_score'])
        self.total_scores.append(r['total_score'])
        self.total_hero_fl_rounds += r['hero_fl_rounds']
        self.total_hero_fl_royalty += r['hero_fl_royalty']
        self.total_opp_fl_rounds += r['opp_fl_rounds']

    def to_dict(self):
        ns = np.array(self.normal_scores)
        ts = np.array(self.total_scores)
        fs = np.array(self.fl_scores)
        return {
            'bust': f'{self.busts/self.n*100:.1f}%',
            'fl': f'{self.fl_entries/self.n*100:.1f}%',
            'royalty': f'{self.total_royalty/self.n:.2f}',
            'normal_score': f'{ns.mean():.2f}',
            'fl_score': f'{fs.mean():.2f}',
            'total_score': f'{ts.mean():.2f}',
            'total_std': f'{ts.std():.2f}',
            'win': f'{(ts>0).sum()/self.n*100:.1f}%',
            'fl_rounds': self.total_hero_fl_rounds,
            'fl_royalty': self.total_hero_fl_royalty,
        }

    def report(self):
        d = self.to_dict()
        print(f"\n  [{self.label}] ({self.n} hands)")
        print(f"    Bust: {d['bust']}  FL: {d['fl']}  Royalty: {d['royalty']}")
        print(f"    Normal: {d['normal_score']}  FL bonus: {d['fl_score']}  "
              f"Total: {d['total_score']} +/- {d['total_std']}")
        print(f"    Win: {d['win']}  FL rounds: {d['fl_rounds']}  "
              f"FL royalty: {d['fl_royalty']}")


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OFC Pineapple Hybrid v2 + FL")
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--bc-model", type=str,
                        default="ai/models/selfplay_iter17/bc_policy_best.pt")
    parser.add_argument("--ppo-model", type=str,
                        default="ai/models/ppo_curriculum/phase3_standalone/final.zip")
    parser.add_argument("--rollouts", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="eval_results_v2.json")
    args = parser.parse_args()

    print("=" * 60)
    print("  OFC Pineapple - Hybrid v2 + FL Rounds")
    print("=" * 60)
    print(f"  Games:     {args.games}")
    print(f"  BC model:  {args.bc_model}")
    print(f"  PPO model: {args.ppo_model}")

    # Load BC
    policy_net = PolicyNetwork()
    ck = torch.load(args.bc_model, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and "model_state_dict" in ck:
        policy_net.load_state_dict(ck["model_state_dict"])
    else:
        policy_net.load_state_dict(ck)
    policy_net.eval()

    from sb3_contrib import MaskablePPO
    ppo = MaskablePPO.load(args.ppo_model)

    rollout_eval = RolloutEvaluator(
        policy_net=policy_net, n_rollouts=args.rollouts,
        top_k=args.top_k, device="cpu"
    )
    rollout_eval_light = RolloutEvaluator(
        policy_net=policy_net, n_rollouts=max(args.rollouts // 2, 10),
        top_k=args.top_k, device="cpu"
    )

    # Initialize Rust FL solver
    fl_solver = RustFLSolver()
    print("  Rust FL solver: ready")

    # Generate decks
    rng = random.Random(args.seed)
    decks = []
    for _ in range(args.games):
        d = list(ALL_CARDS)
        rng.shuffle(d)
        decks.append(d)

    fl_pot = sum(1 for d in decks if has_fl_potential_t0(d[:5]))
    print(f"  FL-potential hands: {fl_pot}/{args.games} ({fl_pot/args.games*100:.0f}%)")
    print()

    results = {}

    # --- 1. Hybrid v2 (baseline, for comparison) ---
    print("  [1/2] Hybrid v2 (PPO + RolloutEval + FL)...", flush=True)
    stats = Stats("Hybrid v2")
    t0 = time.time()
    for i, deck in enumerate(decks):
        r = play_full_game_hybrid(deck, ppo, policy_net, rollout_eval,
                                  rollout_eval_light, fl_solver, all_rollout=False)
        stats.add(r)
        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{args.games}] {time.time()-t0:.0f}s", flush=True)
    stats.report()
    results["Hybrid_v2"] = stats.to_dict()

    # --- 2. All-turn RolloutEval ---
    print("\n  [2/2] All-turn RolloutEval + FL...", flush=True)
    stats = Stats("AllRolloutEval")
    t0 = time.time()
    for i, deck in enumerate(decks):
        r = play_full_game_hybrid(deck, ppo, policy_net, rollout_eval,
                                  rollout_eval_light, fl_solver, all_rollout=True)
        stats.add(r)
        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{args.games}] {time.time()-t0:.0f}s", flush=True)
    stats.report()
    results["AllRolloutEval"] = stats.to_dict()

    # Save
    output = {"fl_potential_hands": f"{fl_pot}/{args.games}", "results": results}
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
