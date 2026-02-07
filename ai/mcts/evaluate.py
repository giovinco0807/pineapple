"""
OFC Pineapple - Model Evaluation

Pit two models against each other to measure relative strength.
Uses pure policy (no MCTS) for speed.
"""
import sys
import random
from pathlib import Path
from typing import List, Optional

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.encoding import Observation, encode_state, ALL_CARDS
from ai.engine.action_space import (
    get_initial_actions, get_turn_actions, create_action_mask, Action
)
from ai.engine.game_engine import GameEngine, Hand, HandResult


def policy_select_action(
    policy_net: torch.nn.Module,
    obs: Observation,
    device: str = "cpu",
) -> tuple:
    """Select action using pure policy (no MCTS)."""
    if obs.turn == 0:
        valid_actions = get_initial_actions(obs.dealt_cards, obs.board_self)
    else:
        valid_actions = get_turn_actions(obs.dealt_cards, obs.board_self)

    if not valid_actions:
        return 0, []
    if len(valid_actions) == 1:
        return 0, valid_actions

    state_vec = encode_state(obs)
    state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(device)
    mask = create_action_mask(valid_actions)
    mask_tensor = torch.BoolTensor(mask).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = policy_net(state_tensor, mask_tensor).squeeze(0).cpu().numpy()

    # Select best action among valid actions
    best_idx = np.argmax(probs[:len(valid_actions)])
    return best_idx, valid_actions


def play_eval_hand(
    policy_a: torch.nn.Module,
    policy_b: torch.nn.Module,
    seat_a: int = 0,
    device: str = "cpu",
) -> HandResult:
    """
    Play one hand: model A as seat_a, model B as the other.

    Returns:
        HandResult
    """
    deck = list(ALL_CARDS)
    random.shuffle(deck)
    hand = Hand(deck=deck, btn=random.randint(0, 1))
    seat_b = 1 - seat_a

    policies = {seat_a: policy_a, seat_b: policy_b}

    # Turn 0
    for seat in [hand.btn, 1 - hand.btn]:
        obs = hand.get_observation(seat)
        action_idx, valid_actions = policy_select_action(
            policies[seat], obs, device
        )
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
            action_idx, valid_actions = policy_select_action(
                policies[seat], obs, device
            )
            if valid_actions and action_idx < len(valid_actions):
                hand.apply_action(seat, valid_actions[action_idx])

    return GameEngine.compute_result(hand)


def evaluate_models(
    policy_a: torch.nn.Module,
    value_a: torch.nn.Module,
    policy_b: torch.nn.Module,
    value_b: torch.nn.Module,
    num_games: int = 100,
    device: str = "cpu",
) -> float:
    """
    Evaluate model A vs model B.

    Plays num_games hands, alternating sides.
    Returns win rate of model A (per-hand wins / total).
    """
    policy_a.eval()
    policy_b.eval()

    wins_a = 0
    wins_b = 0
    draws = 0
    a_busts = 0
    b_busts = 0
    score_a = 0
    score_b = 0

    for game_idx in range(num_games):
        # Alternate who is P0 vs P1
        seat_a = game_idx % 2

        try:
            result = play_eval_hand(policy_a, policy_b, seat_a, device)
            seat_b = 1 - seat_a

            sa = result.raw_score[seat_a]
            sb = result.raw_score[seat_b]
            score_a += sa
            score_b += sb

            if sa > sb:
                wins_a += 1
            elif sb > sa:
                wins_b += 1
            else:
                draws += 1

            if result.busted[seat_a]:
                a_busts += 1
            if result.busted[seat_b]:
                b_busts += 1

        except Exception as e:
            print(f"  [WARN] Eval game {game_idx} failed: {e}")

    total = wins_a + wins_b + draws
    win_rate = wins_a / max(total, 1)

    print(f"    Model A: wins={wins_a}, avg={score_a/max(num_games,1):.1f}, busts={a_busts}")
    print(f"    Model B: wins={wins_b}, avg={score_b/max(num_games,1):.1f}, busts={b_busts}")
    print(f"    Draws: {draws}")

    return win_rate
