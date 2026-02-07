"""
OFC Pineapple - Self-Play Data Generation

Plays full hands using MCTS for both players.
Records trajectories for training.
"""
import sys
import random
import time
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.encoding import Board, Observation, encode_state, ALL_CARDS
from ai.engine.action_space import (
    get_initial_actions, get_turn_actions, create_action_mask, MAX_ACTIONS, Action
)
from ai.engine.game_engine import GameEngine, Hand, HandResult
from ai.mcts.mcts import MCTS, MCTSConfig
from ai.training.config import RewardConfig, compute_hand_reward, REWARD_CONFIG


@dataclass
class TrajectoryStep:
    """One decision point in a game."""
    state_vec: np.ndarray          # 490-dim encoded state
    action_probs: Dict[int, float] # MCTS visit-count distribution
    action_idx: int                # Chosen action index
    valid_mask: np.ndarray         # MAX_ACTIONS bool mask
    player: int
    turn: int
    reward: float = 0.0           # Filled after hand completion


def play_hand_with_mcts(
    mcts: MCTS,
) -> Tuple[List[TrajectoryStep], HandResult]:
    """
    Play one complete hand using MCTS for both players.

    Returns:
        trajectories: list of trajectory steps (both players)
        result: HandResult from scoring
    """
    deck = list(ALL_CARDS)
    random.shuffle(deck)

    hand = Hand(deck=deck, btn=random.randint(0, 1))
    trajectories: Dict[int, List[TrajectoryStep]] = {0: [], 1: []}

    # Turn 0: initial 5-card placement (both players)
    for seat in [hand.btn, 1 - hand.btn]:
        obs = hand.get_observation(seat)
        action_idx, action_probs, valid_actions = mcts.search(obs, {})

        # Record trajectory
        state_vec = encode_state(obs)
        mask = create_action_mask(valid_actions)

        trajectories[seat].append(TrajectoryStep(
            state_vec=state_vec,
            action_probs=action_probs,
            action_idx=action_idx,
            valid_mask=mask,
            player=seat,
            turn=0,
        ))

        # Apply action
        if valid_actions and action_idx < len(valid_actions):
            hand.apply_action(seat, valid_actions[action_idx])

    # Turns 1-8
    for turn_num in range(1, 9):
        if hand.is_hand_complete():
            break
        hand.deal_next_turn()

        for seat in [hand.btn, 1 - hand.btn]:
            cards = hand.dealt_cards[seat]
            if not cards:
                continue
            if hand.boards[seat].is_complete():
                continue

            obs = hand.get_observation(seat)
            action_idx, action_probs, valid_actions = mcts.search(obs, {})

            state_vec = encode_state(obs)
            mask = create_action_mask(valid_actions)

            trajectories[seat].append(TrajectoryStep(
                state_vec=state_vec,
                action_probs=action_probs,
                action_idx=action_idx,
                valid_mask=mask,
                player=seat,
                turn=turn_num,
            ))

            if valid_actions and action_idx < len(valid_actions):
                hand.apply_action(seat, valid_actions[action_idx])

    # Score the hand
    result = GameEngine.compute_result(hand)

    # Attach rewards to trajectories
    for seat in [0, 1]:
        hand_reward = compute_hand_reward(
            {
                "raw_score": result.raw_score,
                "fl_entry": result.fl_entry,
                "busted": result.busted,
            },
            seat,
        )
        # Normalize reward to roughly [-1, 1]
        reward = max(-1.0, min(1.0, hand_reward / 20.0))
        for step in trajectories[seat]:
            step.reward = reward

    all_steps = trajectories[0] + trajectories[1]
    return all_steps, result


def generate_self_play_data(
    policy_net: torch.nn.Module,
    value_net: torch.nn.Module,
    num_games: int = 200,
    mcts_config: MCTSConfig = MCTSConfig(),
    device: str = "cpu",
) -> List[TrajectoryStep]:
    """
    Generate training data from self-play.

    Returns:
        all_steps: list of TrajectoryStep from all games
    """
    mcts = MCTS(policy_net, value_net, mcts_config, device)
    all_steps: List[TrajectoryStep] = []

    total_busts = 0
    total_fl = 0
    total_score = [0.0, 0.0]
    start_time = time.time()

    for game_idx in range(num_games):
        try:
            steps, result = play_hand_with_mcts(mcts)
            all_steps.extend(steps)

            for seat in [0, 1]:
                if result.busted[seat]:
                    total_busts += 1
                if result.fl_entry[seat]:
                    total_fl += 1
                total_score[seat] += result.raw_score[seat]

        except Exception as e:
            print(f"  [WARN] Game {game_idx} failed: {e}")

        if (game_idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (game_idx + 1) / elapsed
            print(f"  {game_idx+1}/{num_games} games "
                  f"({len(all_steps)} steps, {rate:.1f} games/s)")

    elapsed = time.time() - start_time
    bust_rate = total_busts / max(num_games * 2, 1)
    fl_rate = total_fl / max(num_games * 2, 1)

    print(f"\n  Self-play done: {len(all_steps)} steps in {elapsed:.1f}s")
    print(f"  Bust: {bust_rate:.1%}, FL: {fl_rate:.1%}")
    print(f"  Avg score: P0={total_score[0]/max(num_games,1):.1f}, "
          f"P1={total_score[1]/max(num_games,1):.1f}")

    return all_steps
