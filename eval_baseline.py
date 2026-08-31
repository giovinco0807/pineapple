"""
OFC Pineapple - Comprehensive Model Evaluation

Evaluates models with the SAME game flow as self_play.py:
  - T0: RolloutEvaluator (N rollouts)
  - T1-3: RolloutEvaluator (lighter)
  - Last turn (card_count=11): Direct evaluation
  - Other turns: BC greedy

Compares:
  1. BC + RolloutEvaluator (self_play.py baseline)
  2. PPO model (RL env)
  3. Pure BC (no rollout)

Usage:
    python eval_baseline.py --games 50
    python eval_baseline.py --games 50 --ppo-model ai/models/ppo_5m/final.zip
"""
import sys
import random
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np

from ai.engine.encoding import Board, Observation, ALL_CARDS, encode_state
from ai.engine.action_space import (
    get_initial_actions, get_turn_actions, create_action_mask, MAX_ACTIONS
)
from ai.engine.game_engine import (
    GameEngine, Hand, evaluate_hand,
    get_top_royalty, get_middle_royalty, get_bottom_royalty,
    check_fl_entry,
)
from ai.models.networks import PolicyNetwork
from ai.mcts.rollout_evaluator import RolloutEvaluator


# ─── Helper: BC greedy selection ─────────────────────────────────────

def bc_select(policy_net, obs):
    """BC greedy action selection (same as self_play.py)."""
    valid_actions = get_turn_actions(obs.dealt_cards, obs.board_self)
    if not valid_actions:
        return None
    if len(valid_actions) == 1:
        return valid_actions[0]
    state_vec = encode_state(obs)
    state_t = torch.FloatTensor(state_vec).unsqueeze(0)
    mask = create_action_mask(valid_actions)
    mask_t = torch.BoolTensor(mask).unsqueeze(0)
    with torch.no_grad():
        probs = policy_net(state_t, mask_t).squeeze(0).numpy()
    return valid_actions[int(np.argmax(probs[:len(valid_actions)]))]


def direct_eval_last_turn(obs):
    """Direct evaluation for the last turn (card_count=11)."""
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
            fl_bonus = 10 if check_fl_entry(top) else 0
            score = roy + fl_bonus

        if score > best_score:
            best_score = score
            best_action = action

    return best_action


# ─── Play one hand (self_play.py flow) ───────────────────────────────

def play_hand_with_rollout(deck, policy_net, rollout_eval, rollout_eval_light):
    """Play one hand with the full self_play.py game flow."""
    hand = Hand(deck=list(deck), btn=0)

    # T0: Rollout evaluator for both seats
    for seat in [hand.btn, 1 - hand.btn]:
        obs = hand.get_observation(seat)
        _, action = rollout_eval.select_action(obs)
        if action:
            hand.apply_action(seat, action)

    # T1-4: Rollout (light) or direct eval for last turn
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
            if card_count == 11:
                action = direct_eval_last_turn(obs)
            else:
                _, action = rollout_eval_light.select_action(obs)
            if action:
                hand.apply_action(seat, action)

    return GameEngine.compute_result(hand)


def play_hand_bc_only(deck, policy_net):
    """Play one hand with pure BC (no rollout evaluator)."""
    hand = Hand(deck=list(deck), btn=0)

    # T0
    for seat in [hand.btn, 1 - hand.btn]:
        obs = hand.get_observation(seat)
        va = get_initial_actions(obs.dealt_cards, obs.board_self)
        if len(va) == 1:
            action = va[0]
        else:
            state_t = torch.FloatTensor(encode_state(obs)).unsqueeze(0)
            mask_t = torch.BoolTensor(create_action_mask(va)).unsqueeze(0)
            with torch.no_grad():
                probs = policy_net(state_t, mask_t).squeeze(0).numpy()
            action = va[int(np.argmax(probs[:len(va)]))]
        hand.apply_action(seat, action)

    # T1+
    for turn in range(1, 9):
        if hand.is_hand_complete():
            break
        hand.deal_next_turn()
        for seat in [hand.btn, 1 - hand.btn]:
            cards = hand.dealt_cards[seat]
            if not cards or hand.boards[seat].is_complete():
                continue
            obs = hand.get_observation(seat)
            action = bc_select(policy_net, obs)
            if action:
                hand.apply_action(seat, action)

    return GameEngine.compute_result(hand)


def play_hand_ppo(deck, ppo_model):
    """Play one hand with PPO model in RL env."""
    from ai.rl.ofc_selfplay_env import OFCSelfPlayEnv
    env = OFCSelfPlayEnv(opponent_mode="random", seed=None)
    env.hand = Hand(deck=list(deck), btn=0)
    env.current_seat = 0
    env._compute_valid_actions()
    obs = env._get_obs()

    done = False
    while not done:
        mask = env.action_masks()
        action, _ = ppo_model.predict(obs, action_masks=mask, deterministic=True)
        obs, reward, done, trunc, info = env.step(int(action))

    return info.get("hand_result", {})


# ─── Aggregate stats ─────────────────────────────────────────────────

class Stats:
    def __init__(self, label):
        self.label = label
        self.busts = 0
        self.fl_entries = 0
        self.total_roy = 0
        self.scores = []
        self.n = 0

    def add(self, busted_0, fl_0, roy_0, score_0):
        self.n += 1
        if busted_0: self.busts += 1
        if fl_0: self.fl_entries += 1
        self.total_roy += roy_0
        self.scores.append(score_0)

    def report(self):
        s = np.array(self.scores)
        print(f"\n  [{self.label}] ({self.n} hands)")
        print(f"    Bust rate:    {self.busts/self.n*100:.1f}%")
        print(f"    FL rate:      {self.fl_entries/self.n*100:.1f}%")
        print(f"    Avg royalty:  {self.total_roy/self.n:.2f}")
        print(f"    Avg score:    {s.mean():.2f} +/- {s.std():.2f}")
        print(f"    Win rate:     {(s>0).sum()/self.n*100:.1f}%")


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OFC Pineapple Comprehensive Evaluation")
    parser.add_argument("--games", type=int, default=50, help="Number of hands per model")
    parser.add_argument("--bc-model", type=str,
                        default="ai/models/selfplay_iter17/bc_policy_best.pt")
    parser.add_argument("--ppo-model", type=str, default=None,
                        help="Path to PPO model .zip (optional)")
    parser.add_argument("--rollouts", type=int, default=20, help="Rollouts per action")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K candidates")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("  OFC Pineapple - Comprehensive Model Evaluation")
    print("=" * 60)
    print(f"  Games:     {args.games}")
    print(f"  BC model:  {args.bc_model}")
    print(f"  PPO model: {args.ppo_model or 'None'}")
    print(f"  Rollouts:  {args.rollouts}")

    # Load BC model
    policy_net = PolicyNetwork()
    ck = torch.load(args.bc_model, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and "model_state_dict" in ck:
        policy_net.load_state_dict(ck["model_state_dict"])
    else:
        policy_net.load_state_dict(ck)
    policy_net.eval()

    # Create rollout evaluators
    rollout_eval = RolloutEvaluator(
        policy_net=policy_net, n_rollouts=args.rollouts,
        top_k=args.top_k, device="cpu"
    )
    rollout_eval_light = RolloutEvaluator(
        policy_net=policy_net, n_rollouts=max(args.rollouts // 2, 10),
        top_k=args.top_k, device="cpu"
    )

    # Generate decks (same for all models)
    rng = random.Random(args.seed)
    decks = []
    for _ in range(args.games):
        d = list(ALL_CARDS)
        rng.shuffle(d)
        decks.append(d)

    # ─── Eval 1: BC + RolloutEvaluator (self_play.py baseline) ───
    print(f"\n  Evaluating BC + RolloutEvaluator...", flush=True)
    stats_rollout = Stats("BC + RolloutEvaluator (baseline)")
    t0 = time.time()
    for i, deck in enumerate(decks):
        result = play_hand_with_rollout(deck, policy_net, rollout_eval, rollout_eval_light)
        stats_rollout.add(
            result.busted[0], result.fl_entry[0],
            result.royalties[0]["total"], result.raw_score[0]
        )
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"    [{i+1}/{args.games}] {elapsed:.0f}s", flush=True)
    stats_rollout.report()

    # ─── Eval 2: Pure BC (no rollout) ───
    print(f"\n  Evaluating Pure BC...", flush=True)
    stats_bc = Stats("Pure BC (no rollout)")
    for deck in decks:
        result = play_hand_bc_only(deck, policy_net)
        stats_bc.add(
            result.busted[0], result.fl_entry[0],
            result.royalties[0]["total"], result.raw_score[0]
        )
    stats_bc.report()

    # ─── Eval 3: PPO model (if provided) ───
    if args.ppo_model:
        from sb3_contrib import MaskablePPO
        print(f"\n  Evaluating PPO model...", flush=True)
        ppo = MaskablePPO.load(args.ppo_model)
        stats_ppo = Stats(f"PPO ({Path(args.ppo_model).stem})")
        for deck in decks:
            hr = play_hand_ppo(deck, ppo)
            stats_ppo.add(
                hr.get("busted", [False, False])[0],
                hr.get("fl_entry", [False, False])[0],
                hr.get("royalties", [0, 0])[0],
                hr.get("raw_score", [0, 0])[0],
            )
        stats_ppo.report()

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
