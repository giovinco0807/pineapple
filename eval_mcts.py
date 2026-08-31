"""
OFC Pineapple - MCTS 2-ply Evaluation

Hero: MultiTurnMCTS(T0) + RolloutEvaluator(T1+) + direct_eval(last turn)
Opponent: BC greedy
FL rounds: Rust FL solver with chain re-entry

Usage:
    python eval_mcts.py --games 200 --seed 42
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
    get_initial_actions, get_turn_actions, create_action_mask
)
from ai.engine.game_engine import (
    GameEngine, Hand, evaluate_hand,
    get_top_royalty, get_middle_royalty, get_bottom_royalty,
    check_fl_entry, RANK_VALUES,
)
from ai.models.networks import PolicyNetwork, ValueNetwork, ValueNetworkV3
from ai.mcts.rollout_evaluator import RolloutEvaluator
from ai.mcts.multi_turn_mcts import MultiTurnMCTS, MultiTurnConfig
from ai.mcts.ofc_mcts import OFC_MCTS, OFCMCTSConfig
from ai.rust_solver_wrapper import RustFLSolver
from ai.engine.scoring import check_fl_stay_from_cards
from ai.engine.action_space import Action
from ai.prob_engine_wrapper import evaluate_candidates as pe_evaluate_candidates
from ai.prob_engine_wrapper import evaluate_mc_t0 as pe_evaluate_mc_t0
from ai.prob_engine_wrapper import evaluate_mc_candidates as pe_evaluate_mc_candidates


# ─── Card format conversion ─────────────────────────────────────────

_bc_input_dim = 520  # Set after model load

SUIT_MAP = {'s': 0, 'h': 1, 'd': 2, 'c': 3}

def card_str_to_rust(card_str):
    if card_str.startswith('X'):
        return (0, 4)
    rank = RANK_VALUES.get(card_str[:-1], 0)
    suit = SUIT_MAP.get(card_str[-1], 0)
    return (rank, suit)

def rust_card_to_str(rank, suit):
    if suit == 4 or rank == 0:
        return 'X1'
    rank_map = {2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',
                9:'9',10:'T',11:'J',12:'Q',13:'K',14:'A'}
    suit_map = {0:'s',1:'h',2:'d',3:'c'}
    return rank_map.get(rank,'?') + suit_map.get(suit,'?')


# ─── FL round play ──────────────────────────────────────────────────

def play_fl_round(fl_solver, fl_card_count, available_cards):
    if len(available_cards) < fl_card_count:
        return 0, [], [], [], [], False, 0
    dealt = available_cards[:fl_card_count]
    rust_cards = [card_str_to_rust(c) for c in dealt]
    result = fl_solver.solve(rust_cards)
    if result is None:
        return 0, [], [], [], [], False, 0
    royalty = (result.get('top_royalty', 0) + result.get('middle_royalty', 0)
               + result.get('bottom_royalty', 0))
    top_strs = [rust_card_to_str(c['rank'], c['suit']) for c in result.get('top', [])]
    mid_strs = [rust_card_to_str(c['rank'], c['suit']) for c in result.get('middle', [])]
    bot_strs = [rust_card_to_str(c['rank'], c['suit']) for c in result.get('bottom', [])]
    can_stay, next_fl = check_fl_stay_from_cards(
        top_strs, bot_strs, fl_card_count, middle_cards=mid_strs
    )
    return royalty, top_strs, mid_strs, bot_strs, [], can_stay, next_fl


def play_fl_chain(fl_solver, initial_fl_cards, deck, deck_idx):
    total_royalty = 0
    total_rounds = 0
    fl_boards = []
    fl_cards = initial_fl_cards
    pool = list(deck[deck_idx:])
    total_consumed = len(pool)
    while fl_cards > 0:
        random.shuffle(pool)
        royalty, top, mid, bot, _, can_stay, new_fl = play_fl_round(
            fl_solver, fl_cards, pool)
        if not top:
            break
        total_royalty += royalty
        total_rounds += 1
        fl_boards.append((top, mid, bot, royalty))
        placed = set(top + mid + bot)
        pool = [c for c in pool if c not in placed]
        if can_stay and new_fl > 0:
            fl_cards = new_fl
        else:
            break
    return total_royalty, total_rounds, fl_boards, total_consumed


def compute_fl_score(hero_boards, opp_boards, hero_roy, opp_roy):
    score = 0
    n = max(len(hero_boards), len(opp_boards))
    for i in range(n):
        if i < len(hero_boards) and i < len(opp_boards):
            h_top, h_mid, h_bot, h_r = hero_boards[i]
            o_top, o_mid, o_bot, o_r = opp_boards[i]
            score += _score_fl_vs(h_top, h_mid, h_bot, h_r, o_top, o_mid, o_bot, False)
        elif i < len(hero_boards):
            _, _, _, h_r = hero_boards[i]
            score += 6 + h_r
        else:
            _, _, _, o_r = opp_boards[i]
            score -= (6 + o_r)
    return score


def _score_fl_vs(fl_top, fl_mid, fl_bot, fl_roy, opp_top, opp_mid, opp_bot, opp_busted):
    if opp_busted:
        return 6 + fl_roy
    fl_tv = evaluate_hand(fl_top, 3)
    fl_mv = evaluate_hand(fl_mid, 5)
    fl_bv = evaluate_hand(fl_bot, 5)
    o_tv = evaluate_hand(opp_top, 3)
    o_mv = evaluate_hand(opp_mid, 5)
    o_bv = evaluate_hand(opp_bot, 5)
    w, l = 0, 0
    if fl_tv > o_tv: w += 1
    elif fl_tv < o_tv: l += 1
    if fl_mv > o_mv: w += 1
    elif fl_mv < o_mv: l += 1
    if fl_bv > o_bv: w += 1
    elif fl_bv < o_bv: l += 1
    ls = w - l
    if w == 3: ls += 3
    elif l == 3: ls -= 3
    opp_r = get_top_royalty(opp_top) + get_middle_royalty(opp_mid) + get_bottom_royalty(opp_bot)
    return ls + fl_roy - opp_r


# ─── BC greedy ──────────────────────────────────────────────────────

def _project_state(state_t, model_dim):
    """Project 522-dim state to model's expected input dim (520 or 522)."""
    if state_t.shape[-1] == model_dim:
        return state_t
    if state_t.shape[-1] == 522 and model_dim == 520:
        # Remove is_fl/opp_is_fl at indices 488,489 (meta[2:4])
        return torch.cat([state_t[:, :488], state_t[:, 490:]], dim=1)
    return state_t


def bc_greedy(policy_net, obs, actions=None):
    if actions is None:
        actions = get_turn_actions(obs.dealt_cards, obs.board_self)
    if not actions:
        return None
    if len(actions) == 1:
        return actions[0]
    state_t = torch.FloatTensor(encode_state(obs)).unsqueeze(0)
    state_t = _project_state(state_t, _bc_input_dim)
    mask = create_action_mask(actions)
    mask_t = torch.BoolTensor(mask).unsqueeze(0)
    with torch.no_grad():
        probs = policy_net(state_t, mask_t).squeeze(0).numpy()
    return actions[int(np.argmax(probs[:len(actions)]))]


def direct_eval_last_turn(obs):
    valid = get_turn_actions(obs.dealt_cards, obs.board_self)
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]
    best_s = float("-inf")
    best_a = valid[0]
    for action in valid:
        t = list(obs.board_self.top); m = list(obs.board_self.middle); b = list(obs.board_self.bottom)
        for c, p in action.placements:
            if p == 'top': t.append(c)
            elif p == 'middle': m.append(c)
            else: b.append(c)
        my_b = Board(top=t, middle=m, bottom=b)
        s = RolloutEvaluator._compute_score(my_b, obs.board_opponent)
        if s > best_s:
            best_s = s
            best_a = action
    return best_a


# ─── Prob engine ─────────────────────────────────────────────────────

def prob_engine_select_action(obs, turn):
    """Use Rust prob_engine to select best action by EV."""
    top = list(obs.board_self.top)
    mid = list(obs.board_self.middle)
    bot = list(obs.board_self.bottom)
    dealt = list(obs.dealt_cards)

    # Exclude: opponent board + hero discards
    exclude = []
    exclude.extend(obs.board_opponent.top)
    exclude.extend(obs.board_opponent.middle)
    exclude.extend(obs.board_opponent.bottom)
    exclude.extend(obs.known_discards_self)

    position = "btn" if obs.is_btn else "bb"

    result = pe_evaluate_candidates(
        top=top, mid=mid, bot=bot,
        dealt=dealt, exclude=exclude,
        turn=turn, position=position,
    )

    if not result.get("candidates"):
        return None

    best = result["candidates"][0]
    placements = [(c, p) for c, p in best["placements"]]
    discard = best.get("discard")

    # Map "Xj" back to original joker card names (X1/X2)
    jokers_in_dealt = [c for c in dealt if c.startswith("X")]
    used_jokers = set()
    new_placements = []
    for c, p in placements:
        if c == "Xj":
            for j in jokers_in_dealt:
                if j not in used_jokers:
                    c = j
                    used_jokers.add(j)
                    break
        new_placements.append((c, p))
    placements = new_placements

    if discard == "Xj":
        for j in jokers_in_dealt:
            if j not in used_jokers:
                discard = j
                used_jokers.add(j)
                break
        else:
            discard = None
    if discard in ("-", "Xj"):
        discard = None

    return Action(placements=placements, discard=discard)


def prob_engine_mc_select_t0(obs, sims=5):
    """Use Rust prob_engine MC simulation for T0 placement selection."""
    dealt = list(obs.dealt_cards)

    # Exclude: opponent board cards
    exclude = []
    exclude.extend(obs.board_opponent.top)
    exclude.extend(obs.board_opponent.middle)
    exclude.extend(obs.board_opponent.bottom)

    result = pe_evaluate_mc_t0(dealt=dealt, exclude=exclude, sims=sims)

    if not result.get("candidates"):
        return None

    best = result["candidates"][0]
    placements = [(c, p) for c, p in best["placements"]]
    discard = best.get("discard")

    # Map "Xj" back to original joker card names (X1/X2)
    jokers_in_dealt = [c for c in dealt if c.startswith("X")]
    used_jokers = set()
    new_placements = []
    for c, p in placements:
        if c == "Xj":
            for j in jokers_in_dealt:
                if j not in used_jokers:
                    c = j
                    used_jokers.add(j)
                    break
        new_placements.append((c, p))
    placements = new_placements

    if discard == "Xj":
        for j in jokers_in_dealt:
            if j not in used_jokers:
                discard = j
                used_jokers.add(j)
                break
        else:
            discard = None
    if discard in ("-", "Xj"):
        discard = None

    return Action(placements=placements, discard=discard)


def prob_engine_mc_select_t1plus(obs, turn, sims=5):
    """Use Rust prob_engine MC simulation for T1+ placement selection."""
    top = list(obs.board_self.top)
    mid = list(obs.board_self.middle)
    bot = list(obs.board_self.bottom)
    dealt = list(obs.dealt_cards)

    # Exclude: opponent board + hero discards
    exclude = []
    exclude.extend(obs.board_opponent.top)
    exclude.extend(obs.board_opponent.middle)
    exclude.extend(obs.board_opponent.bottom)
    exclude.extend(obs.known_discards_self)

    result = pe_evaluate_mc_candidates(
        top=top, mid=mid, bot=bot,
        dealt=dealt, exclude=exclude,
        turn=turn, sims=sims,
    )

    if not result.get("candidates"):
        return None

    best = result["candidates"][0]
    placements = [(c, p) for c, p in best["placements"]]
    discard = best.get("discard")

    # Map "Xj" back to original joker card names (X1/X2)
    jokers_in_dealt = [c for c in dealt if c.startswith("X")]
    used_jokers = set()
    new_placements = []
    for c, p in placements:
        if c == "Xj":
            for j in jokers_in_dealt:
                if j not in used_jokers:
                    c = j
                    used_jokers.add(j)
                    break
        new_placements.append((c, p))
    placements = new_placements

    if discard == "Xj":
        for j in jokers_in_dealt:
            if j not in used_jokers:
                discard = j
                used_jokers.add(j)
                break
        else:
            discard = None
    if discard in ("-", "Xj"):
        discard = None

    return Action(placements=placements, discard=discard)


# ─── Game play ──────────────────────────────────────────────────────

def play_game(deck, mcts, rollout_eval, policy_net, fl_solver, full_mcts=None,
              use_prob_engine=False, pe_mc_sims=0):
    """Play one game. Hero: prob_engine/full_mcts/MCTS+Rollout. Opp: BC greedy."""
    hand = Hand(deck=list(deck), btn=0)
    hero = 0

    # T0
    for seat in [hand.btn, 1 - hand.btn]:
        obs = hand.get_observation(seat)
        if seat == hero:
            if pe_mc_sims > 0:
                # T0: MC simulation (all 232 candidates × N sims)
                action = prob_engine_mc_select_t0(obs, sims=pe_mc_sims)
            elif use_prob_engine:
                # T0: use BC greedy (prob_engine inaccurate for sparse boards)
                va = get_initial_actions(obs.dealt_cards, obs.board_self)
                action = bc_greedy(policy_net, obs, actions=va)
            elif full_mcts is not None:
                _, action = full_mcts.select_action(obs)
            elif mcts is not None:
                _, action = mcts.select_action(obs)
            else:
                _, action = rollout_eval.select_action(obs)
        else:
            va = get_initial_actions(obs.dealt_cards, obs.board_self)
            action = bc_greedy(policy_net, obs, actions=va)
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
            cc = obs.board_self.card_count()
            if seat == hero:
                if cc == 11:
                    action = direct_eval_last_turn(obs)
                elif pe_mc_sims > 0 and turn_num == 1:
                    # T1: MC simulation (independence assumption most harmful here)
                    action = prob_engine_mc_select_t1plus(obs, turn=turn_num, sims=pe_mc_sims)
                elif pe_mc_sims > 0 or use_prob_engine:
                    action = prob_engine_select_action(obs, turn=turn_num)
                elif full_mcts is not None:
                    _, action = full_mcts.select_action(obs)
                else:
                    _, action = rollout_eval.select_action(obs)
            else:
                if cc == 11:
                    action = direct_eval_last_turn(obs)
                else:
                    action = bc_greedy(policy_net, obs)
            if action:
                hand.apply_action(seat, action)

    # Score normal round
    result = GameEngine.compute_result(hand)
    normal_score = result.raw_score[hero]

    # FL rounds
    fl_deck = list(ALL_CARDS)
    used = set()
    for seat in range(2):
        used.update(hand.boards[seat].all_cards())
        used.update(hand.discards[seat])
    fl_deck = [c for c in fl_deck if c not in used]
    random.shuffle(fl_deck)

    hero_fl_roy = opp_fl_roy = 0
    hero_fl_rounds = opp_fl_rounds = 0
    hero_fl_boards = opp_fl_boards = []
    fl_idx = 0

    if result.fl_entry[hero] and not result.busted[hero]:
        hero_fl_roy, hero_fl_rounds, hero_fl_boards, used_n = play_fl_chain(
            fl_solver, result.fl_card_count[hero], fl_deck, fl_idx)
        fl_idx += used_n
    if result.fl_entry[1-hero] and not result.busted[1-hero]:
        opp_fl_roy, opp_fl_rounds, opp_fl_boards, used_n = play_fl_chain(
            fl_solver, result.fl_card_count[1-hero], fl_deck, fl_idx)

    fl_delta = 0
    if hero_fl_rounds > 0 and opp_fl_rounds > 0:
        fl_delta = compute_fl_score(hero_fl_boards, opp_fl_boards, hero_fl_roy, opp_fl_roy)
    elif hero_fl_rounds > 0:
        for _, _, _, r in hero_fl_boards:
            fl_delta += 6 + r
    elif opp_fl_rounds > 0:
        for _, _, _, r in opp_fl_boards:
            fl_delta -= (6 + r)

    # FL breakdown
    fl_type = None
    if result.fl_entry[hero] and not result.busted[hero]:
        top = hand.boards[hero].top
        fl_cards = result.fl_card_count[hero]
        if fl_cards == 17:
            fl_type = 'trips'
        elif fl_cards == 16:
            fl_type = 'AA'
        elif fl_cards == 15:
            fl_type = 'KK'
        elif fl_cards == 14:
            fl_type = 'QQ'

    return {
        'busted': result.busted[hero],
        'fl_entry': result.fl_entry[hero] and not result.busted[hero],
        'fl_type': fl_type,
        'royalty': result.royalties[hero]['total'],
        'normal_score': normal_score,
        'fl_score': fl_delta,
        'total_score': normal_score + fl_delta,
        'hero_fl_rounds': hero_fl_rounds,
    }


# ─── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MCTS 2-ply Evaluation")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mcts-sims", type=int, default=400)
    parser.add_argument("--t1-rollouts", type=int, default=250)
    parser.add_argument("--bust-penalty", type=float, default=0.0)
    parser.add_argument("--fl-ev-scale", type=float, default=1.0)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--vn-top-k", type=int, default=12)
    parser.add_argument("--bust-penalty-rollout", type=float, default=0.0,
                        help="VN bust_prob penalty for Rollout evaluator action selection")
    parser.add_argument("--bc-model", default="ai/models/expectimax_bc_v3/bc_policy_best.pt")
    parser.add_argument("--bc-t1", default=None, help="Per-turn T1 BC model")
    parser.add_argument("--bc-t2", default=None, help="Per-turn T2 BC model")
    parser.add_argument("--bc-t3", default=None, help="Per-turn T3 BC model")
    parser.add_argument("--vn-model", default="ai/models/value_v3/value_best.pt")
    parser.add_argument("--full-mcts", action="store_true",
                        help="Use OFC_MCTS for all turns (not just T0)")
    parser.add_argument("--full-mcts-sims", type=int, default=1000,
                        help="Simulations for full MCTS")
    parser.add_argument("--full-mcts-depth", type=int, default=3,
                        help="Max search depth for full MCTS")
    parser.add_argument("--vn-greedy", action="store_true",
                        help="Use VN-greedy for T1+ instead of RolloutEvaluator")
    parser.add_argument("--rollout-vn-top-k", type=int, default=10,
                        help="VN prefilter top-k for RolloutEvaluator (default: 10)")
    parser.add_argument("--vn-truncate-depth", type=int, default=0,
                        help="VN-truncated rollout depth (0=disabled, 1=1-turn lookahead)")
    parser.add_argument("--vn-truncate-n", type=int, default=500,
                        help="Rollouts per candidate in VN-truncated mode")
    parser.add_argument("--vn-hybrid", action="store_true",
                        help="Hybrid mode: full playout + VN blend (default: truncate only)")
    parser.add_argument("--vn-v3", action="store_true",
                        help="Use ValueNetworkV3 (turn-conditioned) instead of ValueNetwork")
    parser.add_argument("--prob-engine", action="store_true",
                        help="Use Rust prob_engine for hero (all turns)")
    parser.add_argument("--pe-mc-sims", type=int, default=0,
                        help="Use prob_engine MC simulation for T0 (N sims per candidate)")
    parser.add_argument("--game-start", type=int, default=0,
                        help="Start game index (for sharding across VMs)")
    parser.add_argument("--game-end", type=int, default=0,
                        help="End game index exclusive (0=all games)")
    parser.add_argument("--output", default=None,
                        help="Output per-game results to JSONL file")
    args = parser.parse_args()

    print("=" * 60)
    print("  MCTS 2-ply Evaluation")
    print("=" * 60)

    # Load models (auto-detect input_dim for backward compat: 520 or 522)
    global _bc_input_dim
    ck = torch.load(args.bc_model, map_location='cpu', weights_only=True)
    bc_sd = ck.get('model_state_dict', ck)
    bc_dim = bc_sd.get('net.0.weight', bc_sd.get('input_proj.0.weight', torch.empty(0))).shape[-1]
    _bc_input_dim = bc_dim if bc_dim > 0 else 520
    policy = PolicyNetwork(input_dim=_bc_input_dim) if bc_dim > 0 else PolicyNetwork()
    policy.load_state_dict(bc_sd)
    policy.eval()

    vk = torch.load(args.vn_model, map_location='cpu', weights_only=True)
    vn_sd = vk.get('model_state_dict', vk)
    vn_dim = vn_sd.get('shared.0.weight', torch.empty(0)).shape[-1]
    if args.vn_v3:
        vn = ValueNetworkV3(input_dim=vn_dim) if vn_dim > 0 else ValueNetworkV3()
    else:
        vn = ValueNetwork(input_dim=vn_dim) if vn_dim > 0 else ValueNetwork()
    vn.load_state_dict(vn_sd)
    vn.eval()
    ns = vk.get('norm_stats', None)

    if args.mcts_sims > 0:
        mcts = MultiTurnMCTS(
            policy_net=policy, value_net=vn,
            config=MultiTurnConfig(
                num_simulations=args.mcts_sims,
                vn_top_k=args.vn_top_k,
                bust_penalty=args.bust_penalty,
                fl_ev_scale=args.fl_ev_scale,
                c_puct=args.c_puct),
            device='cpu', norm_stats=ns,
        )
    else:
        mcts = None

    rollout_t1 = RolloutEvaluator(
        policy_net=policy, n_rollouts=args.t1_rollouts, top_k=20,
        device='cpu', value_net=vn, vn_top_k=args.rollout_vn_top_k, norm_stats=ns,
    )
    if args.bust_penalty_rollout > 0:
        rollout_t1.bust_penalty = args.bust_penalty_rollout
    if args.vn_truncate_depth > 0:
        rollout_t1.vn_truncate_depth = args.vn_truncate_depth
        rollout_t1.vn_truncate_n = args.vn_truncate_n
        rollout_t1.vn_hybrid = args.vn_hybrid

    # Load per-turn BC models (T4 is always exhaustive enumeration)
    has_per_turn = False
    for turn_num, bc_path in [(1, args.bc_t1), (2, args.bc_t2), (3, args.bc_t3)]:
        if bc_path and Path(bc_path).exists():
            ptk = torch.load(bc_path, map_location='cpu', weights_only=True)
            pt_sd = ptk.get('model_state_dict', ptk)
            pt_dim = pt_sd.get('net.0.weight', pt_sd.get('input_proj.0.weight', torch.empty(0))).shape[-1]
            per_turn_model = PolicyNetwork(input_dim=pt_dim) if pt_dim > 0 else PolicyNetwork()
            per_turn_model.load_state_dict(pt_sd)
            per_turn_model.eval()
            rollout_t1.per_turn_bc[turn_num] = per_turn_model
            has_per_turn = True
            print(f"  Per-turn BC T{turn_num}: {bc_path}")
    if has_per_turn:
        rollout_t1.use_policy_playout = True
        print("  BC playout enabled (per-turn BC + T4 exhaustive)")

    # Full MCTS engine (T0: MCTS, T1+: rollout evaluator)
    full_mcts = None
    if args.full_mcts:
        full_mcts = OFC_MCTS(
            policy_net=policy, value_net=vn,
            config=OFCMCTSConfig(
                num_simulations=args.full_mcts_sims,
                max_search_depth=args.full_mcts_depth,
                bust_penalty=args.bust_penalty,
                fl_ev_scale=args.fl_ev_scale,
                c_puct=args.c_puct,
                vn_top_k=args.vn_top_k),
            device='cpu', norm_stats=ns,
            t1_evaluator=None if args.vn_greedy else rollout_t1,
        )

    fl_solver = RustFLSolver()

    if args.pe_mc_sims > 0:
        print(f"  Hero: prob_engine MC ({args.pe_mc_sims} sims) T0+T1 + prob_engine exact T2-T3 + direct_eval T4")
        print(f"  Opp: BC greedy")
    elif args.prob_engine:
        print(f"  Hero: Rust prob_engine (all turns)")
        print(f"  Opp: BC greedy")
    elif full_mcts is not None:
        print(f"  T0: OFC_MCTS ({args.full_mcts_sims} sims)")
        print(f"      bust_penalty={args.bust_penalty}, fl_ev_scale={args.fl_ev_scale}")
        print(f"      c_puct={args.c_puct}, vn_top_k={args.vn_top_k}")
        if args.vn_greedy:
            print(f"  T1+: VN-Greedy (bust_penalty={args.bust_penalty})")
        elif args.vn_truncate_depth > 0:
            mode = "Hybrid" if args.vn_hybrid else "Truncated"
            print(f"  T1+: VN-{mode} (d={args.vn_truncate_depth}, r={args.vn_truncate_n}, vn_top_k={args.rollout_vn_top_k})")
        else:
            print(f"  T1+: RolloutEvaluator (r={args.t1_rollouts}, vn_top_k={args.rollout_vn_top_k})")
    elif mcts is not None:
        print(f"  T0: MultiTurnMCTS ({args.mcts_sims} sims, 2-ply)")
        print(f"      bust_penalty={args.bust_penalty}, fl_ev_scale={args.fl_ev_scale}")
        print(f"      c_puct={args.c_puct}, vn_top_k={args.vn_top_k}")
    else:
        print(f"  T0: RolloutEvaluator (r={args.t1_rollouts}, vn_top_k=10)")
    if full_mcts is None and mcts is not None:
        print(f"  T1+: RolloutEvaluator (r={args.t1_rollouts}, vn_top_k=10)")
    if args.bust_penalty_rollout > 0:
        print(f"  Bust penalty (rollout): {args.bust_penalty_rollout}")
    # Sharding support
    game_start = args.game_start
    game_end = args.game_end if args.game_end > 0 else args.games
    n_games = game_end - game_start
    if game_start > 0 or game_end < args.games:
        print(f"  Games: {n_games} (shard [{game_start}, {game_end}) of {args.games}), Seed: {args.seed}")
    else:
        print(f"  Games: {args.games}, Seed: {args.seed}")
    print()

    # Generate ALL decks (full set for reproducibility), then slice
    rng = random.Random(args.seed)
    all_decks = []
    for _ in range(args.games):
        d = list(ALL_CARDS)
        rng.shuffle(d)
        all_decks.append(d)
    decks = all_decks[game_start:game_end]

    # Play games
    busts = 0
    fl_entries = 0
    fl_types = {'AA': 0, 'KK': 0, 'QQ': 0, 'trips': 0}
    royalties = []
    normal_scores = []
    total_scores = []
    fl_rounds_total = 0
    t0_times = []

    out_f = open(args.output, 'w') if args.output else None

    t_start = time.time()
    for i, deck in enumerate(decks):
        t_game = time.time()
        r = play_game(deck, mcts, rollout_t1, policy, fl_solver, full_mcts=full_mcts,
                      use_prob_engine=args.prob_engine, pe_mc_sims=args.pe_mc_sims)
        game_time = time.time() - t_game

        if r['busted']:
            busts += 1
        if r['fl_entry']:
            fl_entries += 1
            if r['fl_type']:
                fl_types[r['fl_type']] += 1
        royalties.append(r['royalty'])
        normal_scores.append(r['normal_score'])
        total_scores.append(r['total_score'])
        fl_rounds_total += r['hero_fl_rounds']

        if out_f:
            r['game_time'] = round(game_time, 2)
            out_f.write(json.dumps(r) + '\n')
            out_f.flush()

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t_start
            avg_score = np.mean(total_scores)
            bust_pct = busts / (i + 1) * 100
            fl_pct = fl_entries / (i + 1) * 100
            print(f"  [{i+1:3d}/{n_games}] {elapsed:5.0f}s  "
                  f"Score={avg_score:+.2f}  Bust={bust_pct:.1f}%  FL={fl_pct:.1f}%  "
                  f"({game_time:.1f}s/game)", flush=True)

    total_time = time.time() - t_start

    # Report
    ns_arr = np.array(normal_scores)
    ts_arr = np.array(total_scores)
    roy_arr = np.array(royalties)

    print()
    print("=" * 60)
    print(f"  Results ({n_games} hands, {total_time:.0f}s total)")
    print("=" * 60)
    print(f"  Bust rate:     {busts/n_games*100:.1f}%")
    print(f"  FL entry rate: {fl_entries/n_games*100:.1f}%")
    if fl_entries > 0:
        print(f"    AA:    {fl_types['AA']:3d} ({fl_types['AA']/fl_entries*100:.0f}%)")
        print(f"    KK:    {fl_types['KK']:3d} ({fl_types['KK']/fl_entries*100:.0f}%)")
        print(f"    QQ:    {fl_types['QQ']:3d} ({fl_types['QQ']/fl_entries*100:.0f}%)")
        print(f"    Trips: {fl_types['trips']:3d} ({fl_types['trips']/fl_entries*100:.0f}%)")
    print(f"  Avg royalty:   {roy_arr.mean():.2f}")
    print(f"  Normal score:  {ns_arr.mean():+.2f} +/- {ns_arr.std():.2f}")
    print(f"  FL bonus:      {(ts_arr - ns_arr).mean():+.2f}")
    print(f"  Total score:   {ts_arr.mean():+.2f} +/- {ts_arr.std():.2f}")
    print(f"  Win rate:      {(ts_arr > 0).sum()/n_games*100:.1f}%")
    print(f"  FL rounds:     {fl_rounds_total}")
    print(f"  Speed:         {total_time/n_games:.2f}s/hand")
    print("=" * 60)

    if out_f:
        out_f.close()
        print(f"  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
