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


def play_eval_hand_with_deck(
    policy_a: torch.nn.Module,
    policy_b: torch.nn.Module,
    deck: list,
    seat_a: int = 0,
    device: str = "cpu",
) -> HandResult:
    """
    Play one hand with a specific deck: model A as seat_a, model B as other.
    """
    hand = Hand(deck=list(deck), btn=0)
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
    num_games: int = 200,
    device: str = "cpu",
) -> float:
    """
    Evaluate model A vs model B using deck pairing.

    For each game, generates a random deck and plays two hands:
      1. A=seat0, B=seat1
      2. A=seat1, B=seat0  (same deck)
    This cancels out luck from card distribution.

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
    total_hands = 0

    for game_idx in range(num_games):
        deck = list(ALL_CARDS)
        random.shuffle(deck)

        # Play two hands with same deck, swapping seats
        for seat_a_val in [0, 1]:
            try:
                result = play_eval_hand_with_deck(
                    policy_a, policy_b, deck, seat_a=seat_a_val, device=device
                )
                seat_b_val = 1 - seat_a_val

                sa = result.raw_score[seat_a_val]
                sb = result.raw_score[seat_b_val]
                score_a += sa
                score_b += sb
                total_hands += 1

                if sa > sb:
                    wins_a += 1
                elif sb > sa:
                    wins_b += 1
                else:
                    draws += 1

                if result.busted[seat_a_val]:
                    a_busts += 1
                if result.busted[seat_b_val]:
                    b_busts += 1

            except Exception as e:
                print(f"  [WARN] Eval game {game_idx} seat {seat_a_val} failed: {e}")

    total = wins_a + wins_b + draws
    win_rate = wins_a / max(total, 1)
    avg_margin = (score_a - score_b) / max(total_hands, 1)

    print(f"    Model A: wins={wins_a}, avg={score_a/max(total_hands,1):.1f}, busts={a_busts}")
    print(f"    Model B: wins={wins_b}, avg={score_b/max(total_hands,1):.1f}, busts={b_busts}")
    print(f"    Draws: {draws}, Avg margin: {avg_margin:+.2f}/hand")

    return win_rate
