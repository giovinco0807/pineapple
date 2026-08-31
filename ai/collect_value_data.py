"""
Value Network Data Collection

Collects (obs_520dim, final_score, turn) tuples from AllRollout games.
Hero uses RolloutEvaluator, opponent uses BC greedy.
FL chains are resolved to get accurate final scores.

Usage:
    python ai/collect_value_data.py --games 1000 --rollouts 50 --workers 8 --output data/value_data.npz

Output format (NPZ):
    obs:    float32 array (N, 520) — encoded state observations
    score:  float32 array (N,) — final hand score (including FL chain)
    turn:   int8 array (N,) — turn number (0-4)
"""

import sys
import time
import random
import argparse
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# Also add CWD in case the script is run from project root
import os
sys.path.insert(0, os.getcwd())

import torch
from ai.engine.game_engine import (
    GameEngine,
    Hand,
    check_fl_entry as engine_check_fl_entry,
    evaluate_board_with_joker_constraint,
)
from ai.engine.scoring import check_fl_stay_from_cards
from ai.engine.encoding import encode_state, ALL_CARDS
from ai.engine.action_space import get_initial_actions, get_turn_actions
from ai.models.networks import PolicyNetwork, ValueNetwork
from ai.mcts.rollout_evaluator import RolloutEvaluator
from ai.rust_solver_wrapper import RustFLSolver

# Global worker variables
_policy_net = None
_rollout_eval = None
_rollout_eval_light = None
_fl_solver = None

# Card string <-> tuple conversion for RustFLSolver
_RANK_MAP = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14}
_SUIT_MAP = {'s':0,'h':1,'d':2,'c':3}
_RANK_REV = {v:k for k,v in _RANK_MAP.items()}
_SUIT_REV = {0:'s',1:'h',2:'d',3:'c'}

def _card_str_to_tuple(s):
    if s in ('X1','X2'):
        return (0, 4)
    return (_RANK_MAP[s[:-1]], _SUIT_MAP[s[-1]])

def _card_tuple_to_str(rank, suit):
    if rank == 0 or suit == 4:
        return 'X1'
    return _RANK_REV[rank] + _SUIT_REV[suit]


def init_worker(model_path, n_rollouts, top_k, vn_model_path=None, vn_top_k=10):
    """Initialize models in each worker process."""
    global _policy_net, _rollout_eval, _rollout_eval_light, _fl_solver
    
    _policy_net = PolicyNetwork()
    ck = torch.load(model_path, map_location='cpu', weights_only=False)
    _policy_net.load_state_dict(
        ck['model_state_dict'] if 'model_state_dict' in ck else ck)
    _policy_net.eval()
    
    # Load Value Network if available
    _vn = None
    _ns = None
    if vn_model_path and Path(vn_model_path).exists():
        _vn = ValueNetwork()
        vk = torch.load(vn_model_path, map_location='cpu', weights_only=False)
        _vn.load_state_dict(vk['model_state_dict'] if 'model_state_dict' in vk else vk)
        _vn.eval()
        _ns = vk.get('norm_stats', None)
    
    _rollout_eval = RolloutEvaluator(
        _policy_net, n_rollouts=n_rollouts, top_k=top_k, device='cpu',
        value_net=_vn, vn_top_k=vn_top_k, norm_stats=_ns)
    _rollout_eval_light = RolloutEvaluator(
        _policy_net, n_rollouts=max(n_rollouts // 2, 10), top_k=top_k, device='cpu',
        value_net=_vn, vn_top_k=vn_top_k, norm_stats=_ns)
    
    # FL solver via RustFLSolver (JSON stdin/stdout, v2)
    _fl_solver = RustFLSolver()


def bc_greedy_select(obs, actions=None):
    """BC greedy action selection."""
    global _policy_net
    with torch.no_grad():
        state = torch.FloatTensor(encode_state(obs)).unsqueeze(0)
        if actions is None:
            actions = get_turn_actions(obs.dealt_cards, obs.board_self)
        if not actions:
            return None
        from ai.engine.action_space import create_action_mask, MAX_ACTIONS
        mask = create_action_mask(actions)
        mask_t = torch.BoolTensor(mask).unsqueeze(0)
        probs = _policy_net(state, mask_t)
        idx = torch.argmax(probs, dim=-1).item()
        return actions[idx] if idx < len(actions) else actions[0]


def direct_eval_last_turn(obs):
    """Direct evaluation for the last turn (11 cards placed)."""
    from ai.engine.action_space import get_turn_actions

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
        evaluated = evaluate_board_with_joker_constraint(
            board.top, board.middle, board.bottom
        )
        if evaluated["busted"]:
            score = -100
        else:
            score = int(evaluated["royalties"]["total"])
        if score > best_score:
            best_score = score
            best_act = act
    return best_act


def check_fl_entry(top_cards):
    """Check FL entry from top cards."""
    return engine_check_fl_entry(top_cards)


def solve_fl_round(fl_card_count, available_cards):
    """Solve one FL round with Rust solver via RustFLSolver."""
    cards = available_cards[:fl_card_count]
    card_tuples = [_card_str_to_tuple(c) for c in cards]
    
    try:
        result = _fl_solver.solve(card_tuples)
        if result is None:
            return None
        
        top = [_card_tuple_to_str(r, s) for r, s in result['top']]
        mid = [_card_tuple_to_str(r, s) for r, s in result['middle']]
        bot = [_card_tuple_to_str(r, s) for r, s in result['bottom']]
        evaluated = evaluate_board_with_joker_constraint(top, mid, bot)
        royalty = int(evaluated["royalties"]["total"])
        can_stay, next_fl_cards = check_fl_stay_from_cards(
            top,
            bot,
            fl_card_count,
            middle_cards=mid,
        )
        
        return {
            'royalty': royalty, 'top': top, 'mid': mid, 'bot': bot,
            'can_stay': can_stay, 'next_fl_cards': next_fl_cards
        }
    except Exception:
        return None


def play_one_game(game_idx):
    """Play one full game, return list of (obs_vector, final_score, turn)."""
    rng = random.Random(game_idx * 7919 + 42)
    deck = list(ALL_CARDS)
    rng.shuffle(deck)
    
    hand = Hand(deck=list(deck), btn=0)
    hero_seat = 0
    
    # Collect observations for hero at each turn
    hero_obs_list = []  # list of (obs_520dim, turn_number)
    
    # T0
    for seat in [hand.btn, 1 - hand.btn]:
        obs = hand.get_observation(seat)
        if seat == hero_seat:
            hero_obs_list.append((encode_state(obs), 0))
            _, action = _rollout_eval.select_action(obs)
        else:
            va = get_initial_actions(obs.dealt_cards, obs.board_self)
            action = bc_greedy_select(obs, actions=va)
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
                hero_obs_list.append((encode_state(obs), turn_num))
                if card_count == 11:
                    action = direct_eval_last_turn(obs)
                else:
                    _, action = _rollout_eval_light.select_action(obs)
            else:
                if card_count == 11:
                    action = direct_eval_last_turn(obs)
                else:
                    action = bc_greedy_select(obs)
            if action:
                hand.apply_action(seat, action)
    
    # Compute normal round result
    result = GameEngine.compute_result(hand)
    normal_score = result.raw_score[hero_seat]
    
    # FL chain resolution
    fl_score = 0.0
    if result.fl_entry[hero_seat] and not result.busted[hero_seat]:
        fl_card_count = result.fl_card_count[hero_seat]
        
        # Build remaining deck for FL
        used = set()
        for s in range(2):
            used.update(hand.boards[s].top)
            used.update(hand.boards[s].middle)
            used.update(hand.boards[s].bottom)
            used.update(hand.discards[s])
        remaining = [c for c in ALL_CARDS if c not in used]
        rng.shuffle(remaining)
        
        idx = 0
        current_fl_cards = fl_card_count
        while idx + current_fl_cards <= len(remaining):
            fl_available = remaining[idx:idx + current_fl_cards]
            idx += current_fl_cards
            
            sol = solve_fl_round(current_fl_cards, fl_available)
            if sol is None:
                break
            
            # FL vs opponent: simplified scoop scoring for data collection
            # (same as score_fl_vs_normal from eval_hybrid but simplified)
            fl_round_score = 6 + sol['royalty']  # conservative estimate
            fl_score += fl_round_score
            
            if sol['can_stay'] and sol['next_fl_cards'] > 0:
                current_fl_cards = sol['next_fl_cards']
                # New deck for stay
                stay_remaining = [c for c in ALL_CARDS 
                                  if c not in set(sol['top'] + sol['mid'] + sol['bot'])]
                rng.shuffle(stay_remaining)
                remaining = stay_remaining
                idx = 0
            else:
                break
    
    final_score = normal_score + fl_score
    
    # Build output: each observation gets the same final score + bust/FL labels
    busted = result.busted[hero_seat]
    fl_entered = result.fl_entry[hero_seat] and not busted

    records = []
    for obs_vec, turn in hero_obs_list:
        records.append((obs_vec, final_score, turn, busted, fl_entered))

    return records, busted, fl_entered


def main():
    parser = argparse.ArgumentParser(description="Collect Value Network training data")
    parser.add_argument('--games', type=int, default=1000)
    parser.add_argument('--rollouts', type=int, default=150)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--vn-top-k', type=int, default=10)
    parser.add_argument('--vn-model', default='ai/models/value_v1/value_best.pt')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--model', default='ai/models/selfplay_iter17/bc_policy_best.pt')
    parser.add_argument('--output', default='data/value_data_v3.npz')
    args = parser.parse_args()
    
    n_workers = args.workers or max(1, mp.cpu_count() - 2)
    
    print(f"Value Network Data Collection (VN Prefilter)")
    print(f"  Policy: {args.model}")
    print(f"  VN: {args.vn_model}")
    print(f"  Rollouts: {args.rollouts}, top-k: {args.top_k}, vn-top-k: {args.vn_top_k}")
    print(f"  Workers: {n_workers}")
    print(f"  Games: {args.games}")
    print(f"  Output: {args.output}\n")
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    t0 = time.time()
    all_obs = []
    all_scores = []
    all_turns = []
    all_busted = []
    all_fl = []
    completed = 0
    total_busts = 0
    total_fl = 0
    
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_worker,
        initargs=(args.model, args.rollouts, args.top_k, args.vn_model, args.vn_top_k),
    ) as pool:
        futures = {pool.submit(play_one_game, i): i for i in range(args.games)}
        for future in as_completed(futures):
            try:
                records, busted, fl_entered = future.result()
                for obs_vec, score, turn, b, fl in records:
                    all_obs.append(obs_vec)
                    all_scores.append(score)
                    all_turns.append(turn)
                    all_busted.append(b)
                    all_fl.append(fl)
                total_busts += int(busted)
                total_fl += int(fl_entered)
                completed += 1
                
                if completed % 50 == 0 or completed == args.games:
                    elapsed = time.time() - t0
                    rate = completed / elapsed * 60
                    bust_r = total_busts / max(completed, 1) * 100
                    fl_r = total_fl / max(completed, 1) * 100
                    print(f"  [{completed}/{args.games}] "
                          f"samples={len(all_obs)} "
                          f"bust={bust_r:.0f}% fl={fl_r:.0f}% "
                          f"{rate:.1f} g/min ({elapsed:.0f}s)",
                          flush=True)
            except Exception as e:
                completed += 1
                print(f"  [ERROR] Game {futures[future]}: {e}", flush=True)
    
    # Save as NPZ
    obs_array = np.array(all_obs, dtype=np.float32)
    score_array = np.array(all_scores, dtype=np.float32)
    turn_array = np.array(all_turns, dtype=np.int8)
    busted_array = np.array(all_busted, dtype=np.bool_)
    fl_array = np.array(all_fl, dtype=np.bool_)

    np.savez_compressed(
        args.output,
        obs=obs_array,
        score=score_array,
        turn=turn_array,
        busted=busted_array,
        fl_entry=fl_array,
    )
    
    elapsed = time.time() - t0
    print(f"\nDone! {completed} games, {len(all_obs)} samples")
    print(f"  Shape: obs={obs_array.shape}, score={score_array.shape}")
    print(f"  Score mean={score_array.mean():.2f}, std={score_array.std():.2f}")
    print(f"  Bust={total_busts/max(completed,1)*100:.1f}%, "
          f"FL={total_fl/max(completed,1)*100:.1f}%")
    print(f"  Saved to: {args.output}")
    print(f"  Time: {elapsed/60:.1f} min ({elapsed/max(completed,1):.1f}s/game)")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
