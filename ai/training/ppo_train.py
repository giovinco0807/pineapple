"""
PPO Training for OFC Pineapple - FL Rate Optimization

Trains PolicyNetwork using PPO with:
- Initial weights from Iter18 BC
- Fixed BC opponent
- Episode reward = raw_score + FL_EV bonus

Usage:
    python -m ai.training.ppo_train --iterations 50 --games 500
"""
import sys
import copy
import multiprocessing as mp
from functools import partial
import json
import time
import random
import argparse
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai.engine.encoding import (
    Observation, encode_state, ALL_CARDS, Board, STATE_DIM,
)
from ai.engine.action_space import (
    get_initial_actions, get_turn_actions, create_action_mask,
    MAX_ACTIONS, Action,
)
from ai.engine.game_engine import Hand, GameEngine, check_fl_entry
from ai.models.networks import PolicyNetwork
from ai.training.ppo_buffer import PPOBuffer, EpisodeRecord, StepRecord


# ── FL Expected Value ─────────────────────────────────────────────────
def load_fl_ev() -> dict:
    """Load FL EV from config with reward_mode support.
    
    reward_mode:
        'total':     R / (1 - stay_rate)          e.g. AA = 66.3
        'per_hand':  R                             e.g. AA = 23.80
        'net_total': (R - opp) / (1 - stay_rate)   e.g. AA = 52.4
    """
    config_path = Path(__file__).parent.parent / "config" / "fl_ev.json"
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        opp = cfg["opponent_avg_royalty"]
        mode = cfg.get("reward_mode", "net_total")
        fl_ev = {}
        for cards, s in cfg["fl_stats"].items():
            r = s["R"]
            stay = s["stay_rate"]
            if mode == "total":
                fl_ev[int(cards)] = r / (1 - stay)
            elif mode == "per_hand":
                fl_ev[int(cards)] = r
            else:  # net_total
                fl_ev[int(cards)] = (r - opp) / (1 - stay)
        print(f"FL EV mode={mode}: {fl_ev}")
        return fl_ev
    except Exception:
        return {14: 24.9, 15: 37.6, 16: 66.3, 17: 126.8}


FL_EV = load_fl_ev()


# ── Reward computation ───────────────────────────────────────────────
def compute_reward(result, seat: int) -> float:
    """Compute final episode reward (game score + FL_EV)."""
    score = float(result.raw_score[seat])
    if not result.busted[seat] and result.fl_entry[seat]:
        fl_cards = result.fl_card_count[seat]
        if fl_cards in FL_EV:
            score += FL_EV[fl_cards]
    return score


def compute_turn_reward(action: Action, board_after: 'Board') -> float:
    """Compute immediate reward for FL-shaping.
    
    Rewards:
        - Place A/K on top row: +5
        - Place Joker on top row: +3  
        - AA pair completed on top: +15
        - KK pair completed on top: +10
    """
    reward = 0.0

    for card, pos in action.placements:
        if pos != 'top':
            continue
        rank = card[0]
        # A/K placed on top
        if rank == 'A':
            reward += 5.0
        elif rank == 'K':
            reward += 5.0
        # Joker on top
        elif rank == 'X':  # Joker cards: X1, X2
            reward += 3.0

    # Check for completed pairs on top
    top = board_after.top
    if len(top) >= 2:
        ranks = [c[0] for c in top]
        if ranks.count('A') >= 2:
            reward += 15.0
        elif ranks.count('K') >= 2:
            reward += 10.0

    return reward


# ── Action selection helpers ─────────────────────────────────────────
# NOTE: All inference uses CPU to avoid CUDA NaN in single-sample forward.
# GPU is used only for batched PPO updates.

def select_action_stochastic(
    net: PolicyNetwork, obs: Observation, temperature: float = 0.8
) -> tuple:
    """Select action with stochastic policy (CPU inference).
    
    Returns (action_idx, action, log_prob, state_vec, mask) or None.
    """
    if obs.turn == 0:
        valid = get_initial_actions(obs.dealt_cards, obs.board_self)
    else:
        valid = get_turn_actions(obs.dealt_cards, obs.board_self)

    if not valid:
        return None
    if len(valid) == 1:
        return 0, valid[0], 0.0, None, None

    state_vec = encode_state(obs)
    state_t = torch.FloatTensor(state_vec).unsqueeze(0)
    mask = create_action_mask(valid)
    mask_t = torch.BoolTensor(mask).unsqueeze(0)

    with torch.no_grad():
        logits = net.net(state_t)
        logits = logits.masked_fill(~mask_t, float('-inf'))
        scaled = logits / max(temperature, 1e-8)

        valid_logits = scaled[0, :len(valid)]
        probs = F.softmax(valid_logits, dim=-1)
        probs = probs.clamp(min=1e-8)
        probs = probs / probs.sum()

        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)

    return (
        action_idx.item(),
        valid[action_idx.item()],
        log_prob.item(),
        state_vec,
        mask,
    )


def select_action_greedy(
    net: PolicyNetwork, obs: Observation
) -> tuple:
    """Greedy BC selection for opponent (CPU inference)."""
    if obs.turn == 0:
        valid = get_initial_actions(obs.dealt_cards, obs.board_self)
    else:
        valid = get_turn_actions(obs.dealt_cards, obs.board_self)

    if not valid:
        return None
    if len(valid) == 1:
        return 0, valid[0]

    state_vec = encode_state(obs)
    state_t = torch.FloatTensor(state_vec).unsqueeze(0)
    mask = create_action_mask(valid)
    mask_t = torch.BoolTensor(mask).unsqueeze(0)

    with torch.no_grad():
        probs = net(state_t, mask_t).squeeze(0).numpy()

    best = int(np.argmax(probs[:len(valid)]))
    return best, valid[best]


# ── Data collection ──────────────────────────────────────────────────
def _play_single_game(
    policy_net: PolicyNetwork,
    opponent_net: PolicyNetwork,
    game_idx: int,
    temperature: float,
) -> EpisodeRecord:
    """Play one game and return an EpisodeRecord with per-step rewards."""
    deck = list(ALL_CARDS)
    random.shuffle(deck)
    hand = Hand(deck=list(deck), btn=0)

    my_seat = game_idx % 2
    ep = EpisodeRecord()

    # T0
    for seat in [hand.btn, 1 - hand.btn]:
        obs = hand.get_observation(seat)
        if seat == my_seat:
            result = select_action_stochastic(policy_net, obs, temperature)
            if result is None:
                continue
            action_idx, action, log_prob, state_vec, mask = result
            hand.apply_action(seat, action)
            if state_vec is not None:
                step_reward = compute_turn_reward(action, hand.boards[seat])
                ep.steps.append(StepRecord(
                    state=state_vec, action_idx=action_idx,
                    log_prob=log_prob, valid_mask=mask,
                    n_valid=len(get_initial_actions(obs.dealt_cards, obs.board_self)),
                    reward=step_reward,
                ))
        else:
            res = select_action_greedy(opponent_net, obs)
            if res is not None:
                _, action = res
                hand.apply_action(seat, action)

    # T1-8
    for turn in range(1, 9):
        if hand.is_hand_complete():
            break
        hand.deal_next_turn()
        for seat in [hand.btn, 1 - hand.btn]:
            cards = hand.dealt_cards[seat]
            if not cards or hand.boards[seat].is_complete():
                continue
            obs = hand.get_observation(seat)
            if seat == my_seat:
                result = select_action_stochastic(policy_net, obs, temperature)
                if result is None:
                    continue
                action_idx, action, log_prob, state_vec, mask = result
                hand.apply_action(seat, action)
                if state_vec is not None:
                    step_reward = compute_turn_reward(action, hand.boards[seat])
                    ep.steps.append(StepRecord(
                        state=state_vec, action_idx=action_idx,
                        log_prob=log_prob, valid_mask=mask,
                        n_valid=len(get_turn_actions(obs.dealt_cards, obs.board_self)),
                        reward=step_reward,
                    ))
            else:
                res = select_action_greedy(opponent_net, obs)
                if res is not None:
                    _, action = res
                    hand.apply_action(seat, action)

    game_result = GameEngine.compute_result(hand)
    ep.reward = compute_reward(game_result, my_seat)
    return ep


def collect_episodes(
    policy_net: PolicyNetwork,
    opponent_net: PolicyNetwork,
    n_games: int,
    temperature: float = 0.8,
) -> PPOBuffer:
    """Play n_games sequentially (single process)."""
    buf = PPOBuffer()
    for g in range(n_games):
        ep = _play_single_game(policy_net, opponent_net, g, temperature)
        buf.episodes.append(ep)
    return buf


def _worker_collect(
    policy_state_dict: dict,
    opponent_state_dict: dict,
    game_indices: list,
    temperature: float,
) -> list:
    """Worker function for multiprocessing. Returns list of serialized episodes."""
    policy_net = PolicyNetwork()
    policy_net.load_state_dict(policy_state_dict)
    policy_net.eval()

    opponent_net = PolicyNetwork()
    opponent_net.load_state_dict(opponent_state_dict)
    opponent_net.eval()

    episodes = []
    for g in game_indices:
        ep = _play_single_game(policy_net, opponent_net, g, temperature)
        episodes.append(ep)
    return episodes


def collect_episodes_parallel(
    policy_net: PolicyNetwork,
    opponent_net: PolicyNetwork,
    n_games: int,
    n_workers: int = 4,
    temperature: float = 0.8,
) -> PPOBuffer:
    """Play n_games in parallel using multiprocessing."""
    if n_workers <= 1:
        return collect_episodes(policy_net, opponent_net, n_games, temperature)

    # Split game indices across workers
    all_indices = list(range(n_games))
    chunks = [[] for _ in range(n_workers)]
    for i, idx in enumerate(all_indices):
        chunks[i % n_workers].append(idx)
    chunks = [c for c in chunks if c]  # Remove empty

    policy_sd = policy_net.state_dict()
    opponent_sd = opponent_net.state_dict()

    with mp.Pool(n_workers) as pool:
        results = pool.starmap(
            _worker_collect,
            [(policy_sd, opponent_sd, chunk, temperature) for chunk in chunks],
        )

    buf = PPOBuffer()
    for episode_list in results:
        buf.episodes.extend(episode_list)
    return buf


# ── PPO Update ───────────────────────────────────────────────────────
def ppo_update(
    policy_net: PolicyNetwork,
    optimizer: torch.optim.Optimizer,
    buffer: PPOBuffer,
    device: str,
    clip_eps: float = 0.2,
    epochs: int = 4,
    mini_batch_size: int = 256,
    entropy_coef: float = 0.01,
) -> dict:
    """Perform PPO parameter update."""
    states, actions, old_log_probs, masks, advantages, returns = \
        buffer.compute_advantages()

    states = states.to(device)
    actions = actions.to(device)
    old_log_probs = old_log_probs.to(device)
    masks = masks.to(device)
    advantages = advantages.to(device)

    n = len(states)
    total_loss = 0.0
    total_pg = 0.0
    total_entropy = 0.0
    n_updates = 0

    for epoch in range(epochs):
        indices = torch.randperm(n)
        for start in range(0, n, mini_batch_size):
            end = min(start + mini_batch_size, n)
            idx = indices[start:end]

            batch_states = states[idx]
            batch_actions = actions[idx]
            batch_old_lp = old_log_probs[idx]
            batch_masks = masks[idx]
            batch_adv = advantages[idx]

            # Forward pass — use -1e9 instead of -inf to avoid NaN in entropy
            logits = policy_net.net(batch_states)
            logits = logits.masked_fill(~batch_masks, -1e9)
            log_probs = F.log_softmax(logits, dim=-1)
            probs = F.softmax(logits, dim=-1)

            # New log prob for chosen action
            new_log_probs = log_probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1)

            # PPO clipped objective
            ratio = torch.exp(new_log_probs - batch_old_lp)
            surr1 = ratio * batch_adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_adv
            pg_loss = -torch.min(surr1, surr2).mean()

            # Entropy bonus (only over valid actions)
            # Mask out invalid for entropy to avoid 0 * log(0)
            valid_probs = probs * batch_masks.float()
            valid_log_probs = log_probs * batch_masks.float()
            entropy = -(valid_probs * valid_log_probs).sum(dim=-1).mean()
            entropy_loss = -entropy_coef * entropy

            loss = pg_loss + entropy_loss

            # Skip NaN batches
            if torch.isnan(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            total_pg += pg_loss.item()
            total_entropy += entropy.item()
            n_updates += 1

    return {
        "loss": total_loss / max(n_updates, 1),
        "pg_loss": total_pg / max(n_updates, 1),
        "entropy": total_entropy / max(n_updates, 1),
        "n_updates": n_updates,
    }


# ── Quick FL Evaluation ──────────────────────────────────────────────
def quick_eval(net: PolicyNetwork, n_games: int = 100) -> dict:
    """Quick BC-vs-BC evaluation for FL stats."""
    fl_count = 0
    bust_count = 0
    total = 0
    score_total = 0.0
    fl_types = Counter()

    for g in range(n_games):
        deck = list(ALL_CARDS)
        random.shuffle(deck)
        hand = Hand(deck=list(deck), btn=0)

        for seat in [0, 1]:
            obs = hand.get_observation(seat)
            res = select_action_greedy(net, obs)
            if res:
                _, action = res
                hand.apply_action(seat, action)

        for _ in range(1, 9):
            if hand.is_hand_complete():
                break
            hand.deal_next_turn()
            for seat in [0, 1]:
                c = hand.dealt_cards[seat]
                if not c or hand.boards[seat].is_complete():
                    continue
                obs = hand.get_observation(seat)
                res = select_action_greedy(net, obs)
                if res:
                    _, action = res
                    hand.apply_action(seat, action)

        result = GameEngine.compute_result(hand)
        # Track seat 0
        total += 1
        if result.busted[0]:
            bust_count += 1
        else:
            score_total += result.raw_score[0]
            if result.fl_entry[0]:
                fl_count += 1
                top = result.boards[0].top
                fl, cards = check_fl_entry(top)
                if cards == 17:
                    fl_types["Trips"] += 1
                elif cards == 16:
                    fl_types["AA"] += 1
                elif cards == 15:
                    fl_types["KK"] += 1
                elif cards == 14:
                    fl_types["QQ"] += 1

    return {
        "fl_rate": fl_count / max(total, 1) * 100,
        "fl_count": fl_count,
        "bust_rate": bust_count / max(total, 1) * 100,
        "avg_score": score_total / max(total - bust_count, 1),
        "fl_types": dict(fl_types),
        "total": total,
    }


# ── Main Training Loop ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="PPO Training for FL optimization")
    parser.add_argument("--model", default="ai/models/selfplay_iter18/bc_policy_best.pt",
                        help="Initial policy model path")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--games", type=int, default=500,
                        help="Games per iteration")
    parser.add_argument("--epochs", type=int, default=4,
                        help="PPO epochs per iteration")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--mini-batch", type=int, default=256)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--save-dir", default="ai/models/ppo_fl")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers for data collection")
    args = parser.parse_args()

    if args.device == "auto":
        train_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        train_device = args.device

    # Load models on CPU (data collection + eval use CPU)
    print(f"Loading model: {args.model}")
    policy_net = PolicyNetwork()
    policy_net.load_state_dict(torch.load(
        args.model, map_location="cpu", weights_only=True
    ))
    policy_net.eval()

    # Fixed opponent (stays on CPU forever)
    opponent_net = PolicyNetwork()
    opponent_net.load_state_dict(torch.load(
        args.model, map_location="cpu", weights_only=True
    ))
    opponent_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.lr)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Train device: {train_device}")
    print(f"Config: games={args.games}, epochs={args.epochs}, "
          f"lr={args.lr}, \u03b5={args.clip_eps}, temp={args.temperature}")
    print()

    init_eval = quick_eval(policy_net, args.eval_games)
    print(f"[Init] FL={init_eval['fl_rate']:.1f}% "
          f"Bust={init_eval['bust_rate']:.1f}% "
          f"Score={init_eval['avg_score']:.2f} "
          f"Types={init_eval['fl_types']}")

    best_fl_rate = init_eval["fl_rate"]
    history = []

    for it in range(1, args.iterations + 1):
        t0 = time.time()

        # 1. Collect episodes (CPU, optionally parallel)
        policy_net.eval()
        if args.workers > 1:
            buf = collect_episodes_parallel(
                policy_net, opponent_net, args.games,
                args.workers, args.temperature
            )
        else:
            buf = collect_episodes(
                policy_net, opponent_net, args.games, args.temperature
            )
        stats = buf.stats()

        # 2. PPO update (GPU if available)
        policy_net.to(train_device)
        policy_net.train()
        update_stats = ppo_update(
            policy_net, optimizer, buf, train_device,
            clip_eps=args.clip_eps,
            epochs=args.epochs,
            mini_batch_size=args.mini_batch,
            entropy_coef=args.entropy_coef,
        )
        policy_net.cpu()
        policy_net.eval()

        # 3. Evaluate (CPU)
        eval_result = quick_eval(policy_net, args.eval_games)

        elapsed = time.time() - t0

        record = {
            "iteration": it,
            "fl_rate": eval_result["fl_rate"],
            "bust_rate": eval_result["bust_rate"],
            "avg_score": eval_result["avg_score"],
            "fl_types": eval_result["fl_types"],
            "reward_mean": stats["reward_mean"],
            "reward_std": stats["reward_std"],
            "step_reward_mean": stats["step_reward_mean"],
            "loss": update_stats["loss"],
            "pg_loss": update_stats["pg_loss"],
            "entropy": update_stats["entropy"],
            "steps": stats["steps"],
            "time": elapsed,
        }
        history.append(record)

        print(f"[{it:3d}/{args.iterations}] "
              f"FL={eval_result['fl_rate']:5.1f}% "
              f"Bust={eval_result['bust_rate']:4.1f}% "
              f"Score={eval_result['avg_score']:+6.2f} "
              f"R={stats['reward_mean']:+6.2f}\u00b1{stats['reward_std']:.1f} "
              f"Loss={update_stats['loss']:.4f} "
              f"Ent={update_stats['entropy']:.3f} "
              f"Types={eval_result['fl_types']} "
              f"({elapsed:.0f}s)")

        # Save best model
        if eval_result["fl_rate"] > best_fl_rate:
            best_fl_rate = eval_result["fl_rate"]
            torch.save(policy_net.state_dict(), save_dir / "ppo_best.pt")
            print(f"  \u2605 New best FL rate: {best_fl_rate:.1f}%")

        # Save periodic checkpoint
        if it % 10 == 0:
            torch.save(policy_net.state_dict(), save_dir / f"ppo_iter{it}.pt")
            with open(save_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

    # Save final
    torch.save(policy_net.state_dict(), save_dir / "ppo_final.pt")
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best FL rate: {best_fl_rate:.1f}%")
    print(f"Models saved to: {save_dir}")


if __name__ == "__main__":
    main()

