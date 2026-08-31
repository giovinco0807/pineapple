"""
Value Network Data Collection V2 - Rollout-Averaged Labels

Instead of using a single game outcome as the label (noisy), each
observation gets the rollout evaluator's average score for the chosen
action. Since the rollout evaluator already averages over N rollouts,
the labels are inherently less noisy.

This breaks the VN correlation ceiling caused by label noise:
  V1: label = 1 game outcome (high variance)
  V2: label = average of N rollouts from this position (low variance)

Usage:
    python ai/collect_value_data_v2.py --games 500 --rollouts 100 --output data/value_data_v5.npz
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
import os
sys.path.insert(0, os.getcwd())

import torch
from ai.engine.game_engine import (
    GameEngine,
    Hand,
    evaluate_board_with_joker_constraint,
    evaluate_hand,
    get_top_royalty,
    get_middle_royalty,
    get_bottom_royalty,
)
from ai.engine.encoding import encode_state, ALL_CARDS, Board, Observation
from ai.engine.action_space import get_initial_actions, get_turn_actions, create_action_mask
from ai.models.networks import PolicyNetwork, ValueNetwork
from ai.mcts.rollout_evaluator import RolloutEvaluator

# Global worker variables
_policy_net = None
_rollout_eval = None
_rollout_eval_light = None

_RANK_MAP = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14}


def init_worker(model_path, n_rollouts, top_k, vn_model_path, vn_top_k):
    global _policy_net, _rollout_eval, _rollout_eval_light

    _policy_net = PolicyNetwork()
    ck = torch.load(model_path, map_location='cpu', weights_only=False)
    _policy_net.load_state_dict(
        ck['model_state_dict'] if 'model_state_dict' in ck else ck)
    _policy_net.eval()

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
        _policy_net, n_rollouts=max(n_rollouts // 2, 20), top_k=top_k, device='cpu',
        value_net=_vn, vn_top_k=vn_top_k, norm_stats=_ns)


def bc_greedy_select(obs, actions=None):
    global _policy_net
    with torch.no_grad():
        state = torch.FloatTensor(encode_state(obs)).unsqueeze(0)
        if actions is None:
            actions = get_turn_actions(obs.dealt_cards, obs.board_self)
        if not actions:
            return None
        mask = create_action_mask(actions)
        mask_t = torch.BoolTensor(mask).unsqueeze(0)
        probs = _policy_net(state, mask_t)
        idx = torch.argmax(probs, dim=-1).item()
        return actions[idx] if idx < len(actions) else actions[0]


def direct_eval_last_turn(obs):
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


def play_one_game(game_idx):
    """Play one game, record rollout-averaged scores as labels.

    For each turn:
      - Record the observation (state vector)
      - Use rollout evaluator to get action + average score
      - Label = rollout average score (averaged over N rollouts, much less noisy)
    """
    rng = random.Random(game_idx * 7919 + 42)
    deck = list(ALL_CARDS)
    rng.shuffle(deck)

    hand = Hand(deck=list(deck), btn=0)
    hero_seat = 0

    records = []  # list of (obs_vec, rollout_score, turn)

    # T0: use rollout eval with scores
    for seat in [hand.btn, 1 - hand.btn]:
        obs = hand.get_observation(seat)
        if seat == hero_seat:
            obs_vec = encode_state(obs)
            best_idx, action, scores = _rollout_eval.select_action_with_scores(obs)
            rollout_score = scores[best_idx]
            records.append((obs_vec, rollout_score, 0))
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
                obs_vec = encode_state(obs)
                if card_count == 11:
                    # Last turn: use direct eval (deterministic, score = royalty)
                    action = direct_eval_last_turn(obs)
                    # For last turn, score is just royalty (no uncertainty)
                    if action:
                        test_board = obs.board_self.copy()
                        for c, p in action.placements:
                            getattr(test_board, p).append(c)
                        evaluated = evaluate_board_with_joker_constraint(
                            test_board.top,
                            test_board.middle,
                            test_board.bottom,
                        )
                        if evaluated["busted"]:
                            last_score = -6.0  # bust
                        else:
                            last_score = float(evaluated["royalties"]["total"])
                    else:
                        last_score = 0.0
                    records.append((obs_vec, last_score, turn_num))
                else:
                    best_idx, action, scores = _rollout_eval_light.select_action_with_scores(obs)
                    rollout_score = scores[best_idx]
                    records.append((obs_vec, rollout_score, turn_num))
            else:
                if card_count == 11:
                    action = direct_eval_last_turn(obs)
                else:
                    action = bc_greedy_select(obs)
            if action:
                hand.apply_action(seat, action)

    # Compute actual game result for bust/FL labels
    result = GameEngine.compute_result(hand)
    busted = result.busted[hero_seat]
    fl_entered = result.fl_entry[hero_seat] and not busted

    # Add bust/FL labels to records
    final_records = []
    for obs_vec, rollout_score, turn in records:
        final_records.append((obs_vec, rollout_score, turn, busted, fl_entered))

    return final_records, busted, fl_entered


def main():
    parser = argparse.ArgumentParser(description="Collect VN data V2 (rollout-averaged labels)")
    parser.add_argument('--games', type=int, default=500)
    parser.add_argument('--rollouts', type=int, default=100,
                        help='Rollouts per action evaluation (label quality)')
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--vn-top-k', type=int, default=10)
    parser.add_argument('--vn-model', default='ai/models/value_v1/value_best.pt')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--model', default='ai/models/selfplay_iter17/bc_policy_best.pt')
    parser.add_argument('--output', default='data/value_data_v5.npz')
    args = parser.parse_args()

    n_workers = args.workers or max(1, mp.cpu_count() - 2)

    print(f"Value Network Data Collection V2 (Rollout-Averaged Labels)")
    print(f"  Policy: {args.model}")
    print(f"  VN: {args.vn_model}")
    print(f"  Rollouts: {args.rollouts} (T0), {max(args.rollouts//2,20)} (T1+)")
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
        initargs=(args.model, args.rollouts, args.top_k,
                  args.vn_model, args.vn_top_k),
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

    obs_array = np.array(all_obs, dtype=np.float32)
    score_array = np.array(all_scores, dtype=np.float32)
    turn_array = np.array(all_turns, dtype=np.int8)
    busted_array = np.array(all_busted, dtype=np.bool_)
    fl_array = np.array(all_fl, dtype=np.bool_)

    np.savez_compressed(
        args.output,
        obs=obs_array, score=score_array, turn=turn_array,
        busted=busted_array, fl_entry=fl_array,
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
