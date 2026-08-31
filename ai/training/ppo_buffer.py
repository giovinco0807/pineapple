"""
PPO Experience Buffer for OFC Pineapple.

Stores per-turn experience with per-step rewards (intermediate + final).
Advantages are computed from discounted returns.
"""
from dataclasses import dataclass, field
from typing import List

import torch
import numpy as np


@dataclass
class StepRecord:
    """Single decision-point record."""
    state: np.ndarray          # (STATE_DIM,)
    action_idx: int            # index into valid_actions
    log_prob: float            # log π_old(a|s)
    valid_mask: np.ndarray     # (MAX_ACTIONS,) bool
    n_valid: int               # number of valid actions
    reward: float = 0.0        # per-step reward (intermediate)


@dataclass
class EpisodeRecord:
    """One complete hand for one player."""
    steps: List[StepRecord] = field(default_factory=list)
    reward: float = 0.0        # final episode reward (game score + FL_EV)


class PPOBuffer:
    """Collects episodes and flattens them for PPO training."""

    def __init__(self):
        self.episodes: List[EpisodeRecord] = []

    def new_episode(self) -> EpisodeRecord:
        ep = EpisodeRecord()
        self.episodes.append(ep)
        return ep

    def size(self) -> int:
        return sum(len(ep.steps) for ep in self.episodes)

    def compute_advantages(self, gamma: float = 0.99) -> tuple:
        """Flatten all episodes and compute advantages from discounted returns.

        Each step's return = discounted sum of future step rewards + final reward.
        
        Returns:
            states:     (N, STATE_DIM) float tensor
            actions:    (N,) long tensor
            old_log_probs: (N,) float tensor
            masks:      (N, MAX_ACTIONS) bool tensor
            advantages: (N,) float tensor
            returns:    (N,) float tensor
        """
        all_states = []
        all_actions = []
        all_log_probs = []
        all_masks = []
        all_returns = []

        for ep in self.episodes:
            if not ep.steps:
                continue

            n = len(ep.steps)
            returns = [0.0] * n

            # Discounted return: work backwards
            # Last step gets: step_reward + episode_reward
            # Earlier steps: step_reward + gamma * next_return
            returns[n - 1] = ep.steps[n - 1].reward + ep.reward
            for t in range(n - 2, -1, -1):
                returns[t] = ep.steps[t].reward + gamma * returns[t + 1]

            for step, ret in zip(ep.steps, returns):
                all_states.append(step.state)
                all_actions.append(step.action_idx)
                all_log_probs.append(step.log_prob)
                all_masks.append(step.valid_mask)
                all_returns.append(ret)

        if not all_states:
            raise ValueError("Empty buffer")

        states = torch.FloatTensor(np.array(all_states))
        actions = torch.LongTensor(all_actions)
        old_log_probs = torch.FloatTensor(all_log_probs)
        masks = torch.BoolTensor(np.array(all_masks))
        returns = torch.FloatTensor(all_returns)

        # Advantage = return - baseline (mean return)
        advantages = returns - returns.mean()
        std = advantages.std()
        if std > 1e-8:
            advantages = advantages / std

        return states, actions, old_log_probs, masks, advantages, returns

    def clear(self):
        self.episodes.clear()

    def stats(self) -> dict:
        """Return summary statistics."""
        rewards = [ep.reward for ep in self.episodes]
        step_rewards = [
            s.reward for ep in self.episodes for s in ep.steps
        ]
        return {
            "episodes": len(self.episodes),
            "steps": self.size(),
            "reward_mean": np.mean(rewards) if rewards else 0,
            "reward_std": np.std(rewards) if rewards else 0,
            "reward_min": np.min(rewards) if rewards else 0,
            "reward_max": np.max(rewards) if rewards else 0,
            "step_reward_mean": np.mean(step_rewards) if step_rewards else 0,
        }
