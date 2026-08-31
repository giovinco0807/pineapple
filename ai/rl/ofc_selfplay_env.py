"""
OFC Pineapple - Self-Play PPO Environment

Gymnasium environment for training via PPO with self-play.
Two players take turns placing cards; the agent plays both seats
and receives the reward from seat-0 perspective.

Key features:
  - Uses existing encoding (520-dim), action space (MAX_ACTIONS=250)
  - Action masking for MaskablePPO (sb3-contrib)
  - FL reward shaping (+15 entry, +30 stay)
  - Opponent can be: 'self' (same policy), 'model' (frozen copy), 'random'

Usage:
    from ai.rl.ofc_selfplay_env import OFCSelfPlayEnv
    env = OFCSelfPlayEnv()
    obs, info = env.reset()
"""
import copy
import random
from typing import Dict, List, Optional, Tuple, Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

from ai.engine.encoding import (
    Board, Observation, ALL_CARDS, encode_state, STATE_DIM
)
from ai.engine.action_space import (
    get_initial_actions, get_turn_actions, MAX_ACTIONS, Action
)
from ai.engine.game_engine import (
    GameEngine, Hand, HandResult,
    evaluate_hand, get_top_royalty, get_middle_royalty, get_bottom_royalty,
    check_fl_entry, evaluate_board_with_joker_constraint,
    evaluate_row_with_joker_constraint,
)


class OFCSelfPlayEnv(gym.Env):
    """
    Self-play Gymnasium environment for OFC Pineapple.

    The agent controls BOTH seats in alternation.
    Reward is given from seat-0's perspective at hand end.

    Observation: 520-dim float32 vector (from encoding.py)
    Action: Discrete(MAX_ACTIONS) — index into current valid_actions list
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        opponent_model=None,
        opponent_model_path: str = None,  # Path to BC checkpoint for lazy loading
        opponent_mode: str = "self",   # "self", "model", "random"
        render_mode: Optional[str] = None,
        fl_entry_bonus: float = 15.0,
        fl_stay_bonus: float = 30.0,
        bust_penalty: float = 10.0,
        fl_progress_scale: float = 1.0,  # multiplier for FL staged rewards
        fl_force_ratio: float = 0.0,     # fraction of episodes with forced FL T0
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.opponent_model = opponent_model
        self.opponent_model_path = opponent_model_path
        self.opponent_mode = opponent_mode

        # Lazy-load opponent model from path (for SubprocVecEnv compatibility)
        if opponent_model is None and opponent_model_path and opponent_mode == "model":
            self._load_opponent_model(opponent_model_path)

        # Reward shaping
        self.fl_entry_bonus = fl_entry_bonus
        self.fl_stay_bonus = fl_stay_bonus
        self.bust_penalty = bust_penalty
        self.fl_progress_scale = fl_progress_scale
        self.fl_force_ratio = fl_force_ratio

        # Spaces
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(MAX_ACTIONS)

        # Internal state
        self.hand: Optional[Hand] = None
        self.current_seat: int = 0
        self.valid_actions: List[Action] = []
        self._rng = random.Random(seed)
        self._prev_fl_progress: Dict[int, float] = {0: 0.0, 1: 0.0}
        self._fl_forced: bool = False

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset: deal a new hand, return observation for seat-0 Turn 0."""
        if seed is not None:
            self._rng = random.Random(seed)

        deck = list(ALL_CARDS)
        self._rng.shuffle(deck)
        self.hand = Hand(deck=deck, btn=0)
        self.current_seat = 0
        self._prev_fl_progress = {0: 0.0, 1: 0.0}
        self._fl_forced = self._rng.random() < self.fl_force_ratio
        self._compute_valid_actions()

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take one step: apply action for current_seat, then advance.

        If opponent_mode is "self", the opponent is also controlled by
        the agent — we return obs for the next seat that needs a decision.
        Only when the hand is fully complete do we compute reward.
        """
        assert self.hand is not None, "Must call reset() before step()"

        # Validate action
        if action < 0 or action >= len(self.valid_actions):
            action = 0

        acting_seat = self.current_seat
        chosen_action = self.valid_actions[action]
        self.hand.apply_action(acting_seat, chosen_action)

        # Compute FL progress shaping reward (delta-based)
        shaping_reward = self._fl_progress_reward(acting_seat)

        # If opponent_mode is not "self", play opponent automatically
        if self.opponent_mode != "self":
            opp_seat = 1 - acting_seat
            if not self.hand.placed[opp_seat]:
                self._play_opponent(opp_seat)

        # Check if both players placed this turn
        if self.hand.is_turn_complete():
            if self.hand.is_hand_complete():
                # Hand done → compute reward
                result = GameEngine.compute_result(self.hand)
                reward = self._compute_reward(result) + shaping_reward
                obs = self._get_obs()
                info = self._get_info()
                info["hand_result"] = {
                    "busted": result.busted,
                    "royalties": [r["total"] for r in result.royalties],
                    "fl_entry": result.fl_entry,
                    "raw_score": result.raw_score,
                }
                return obs, reward, True, False, info
            else:
                # Deal next turn
                self.hand.deal_next_turn()

        # Advance to next seat needing action
        if self.opponent_mode == "self":
            self.current_seat = 1 - self.current_seat
            if self.hand.placed[self.current_seat]:
                self.current_seat = 1 - self.current_seat
        else:
            self.current_seat = 0

        self._compute_valid_actions()
        obs = self._get_obs()
        info = self._get_info()
        return obs, shaping_reward, False, False, info

    def action_masks(self) -> np.ndarray:
        """Return boolean mask of valid actions for MaskablePPO."""
        mask = np.zeros(MAX_ACTIONS, dtype=bool)
        n_valid = len(self.valid_actions)
        mask[:n_valid] = True
        return mask

    # ─── Internal ─────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Build observation for current seat."""
        if self.hand is None:
            return np.zeros(STATE_DIM, dtype=np.float32)
        obs = self.hand.get_observation(self.current_seat)
        return encode_state(obs)

    def _get_info(self) -> Dict:
        return {
            "turn": self.hand.turn if self.hand else -1,
            "seat": self.current_seat,
            "n_valid_actions": len(self.valid_actions),
        }

    def _compute_valid_actions(self):
        """Enumerate valid actions for the current seat/turn."""
        if self.hand is None:
            self.valid_actions = []
            return

        obs = self.hand.get_observation(self.current_seat)
        board = self.hand.boards[self.current_seat]
        cards = self.hand.dealt_cards[self.current_seat]

        if self.hand.turn == 0:
            self.valid_actions = get_initial_actions(cards, board)
            # FL experience injection: restrict T0 to FL-seeking actions
            if self._fl_forced and self.current_seat == 0:
                fl_actions = [a for a in self.valid_actions
                              if self._has_high_on_top(a)]
                if fl_actions:
                    self.valid_actions = fl_actions
        else:
            self.valid_actions = get_turn_actions(cards, board)

        # Bust-prevention filter: remove actions that guarantee bust
        if len(self.valid_actions) > 1:
            safe = [a for a in self.valid_actions if not self._causes_bust(a, board)]
            if safe:
                self.valid_actions = safe

    @staticmethod
    def _has_high_on_top(action: Action) -> bool:
        """Check if action places at least one Q/K/A/Joker on top row."""
        for card, pos in action.placements:
            if pos == 'top':
                if card.startswith('X'):  # Joker
                    return True
                rank = card[0] if len(card) >= 2 else ''
                if rank in ('Q', 'K', 'A', 'T'):  # T for 10-char cards like 'Th'
                    # Only Q, K, A qualify for FL
                    if card[:-1] in ('Q', 'K', 'A'):
                        return True
        return False

    def _causes_bust(self, action: Action, board: Board) -> bool:
        """
        Quick bust check: does this action create an ordering violation?

        OFC rule: top <= middle <= bottom (by hand strength).
        We check the immediate state after placing; doesn't guarantee
        the final hand won't bust, but catches obvious violations.
        """
        test_board = board.copy()
        for card, pos in action.placements:
            getattr(test_board, pos).append(card)

        # Only check rows that are complete
        top_full = len(test_board.top) == 3
        mid_full = len(test_board.middle) == 5
        bot_full = len(test_board.bottom) == 5

        if top_full and mid_full and bot_full:
            return bool(evaluate_board_with_joker_constraint(
                test_board.top, test_board.middle, test_board.bottom
            )["busted"])

        if mid_full and bot_full:
            bot_val = evaluate_hand(test_board.bottom, 5)
            _middle, mid_val = evaluate_row_with_joker_constraint(
                test_board.middle, 5, bot_val
            )
            if mid_val > bot_val:
                return True

        if top_full and mid_full:
            mid_val = evaluate_hand(test_board.middle, 5)
            _top, top_val = evaluate_row_with_joker_constraint(
                test_board.top, 3, mid_val
            )
            if top_val > mid_val:
                return True

        if top_full and bot_full:
            bot_val = evaluate_hand(test_board.bottom, 5)
            _top, top_val = evaluate_row_with_joker_constraint(
                test_board.top, 3, bot_val
            )
            if top_val > bot_val:
                return True

        return False

    def _compute_reward(self, result: HandResult) -> float:
        """
        Compute reward from seat-0 perspective.

        Components:
          - Raw score (line wins/losses + royalty diff + scoop)
          - FL entry bonus
          - Bust penalty
        """
        reward = float(result.raw_score[0])

        # FL entry bonus for seat 0
        if result.fl_entry[0]:
            reward += self.fl_entry_bonus

        # Bust penalty
        if result.busted[0]:
            reward -= self.bust_penalty

        return reward

    def _fl_progress_reward(self, seat: int) -> float:
        """
        FL staged progress reward (delta-based shaping).

        Measures how close the top row is to FL qualification (QQ+ pair).
        Returns the CHANGE in progress since last step, so cumulative
        shaping doesn't distort the total reward.

        Progress levels:
          0.0 - Nothing FL-relevant on top
          1.0 - One Q/K/A/Joker on top (1 slot used)
          2.0 - Two high cards on top, no pair yet
          3.0 - K or A on top (higher FL value)
          5.0 - Pair formed (QQ/KK/AA) — FL nearly guaranteed
          8.0 - FL actually achieved (full 3 cards, QQ+)
        """
        if self.fl_progress_scale == 0 or self.hand is None:
            return 0.0

        board = self.hand.boards[seat]
        progress = self._compute_fl_progress(board.top)

        prev = self._prev_fl_progress.get(seat, 0.0)
        self._prev_fl_progress[seat] = progress
        delta = progress - prev

        # Only give reward for seat 0 (agent's perspective)
        if seat != 0:
            return 0.0

        return delta * self.fl_progress_scale

    @staticmethod
    def _compute_fl_progress(top_cards: List[str]) -> float:
        """
        Score FL progress for the top row.

        Returns a float indicating how close to FL entry.
        """
        if not top_cards:
            return 0.0

        ranks = []
        joker_count = 0
        for c in top_cards:
            if c.startswith('X'):
                joker_count += 1
            else:
                ranks.append(c[0])

        high_ranks = {'Q', 'K', 'A'}
        high_count = sum(1 for r in ranks if r in high_ranks) + joker_count

        # Check for pair
        from collections import Counter
        rank_counts = Counter(ranks)
        has_pair = False
        pair_rank = None
        for r, cnt in rank_counts.items():
            if r in high_ranks and cnt + joker_count >= 2:
                has_pair = True
                pair_rank = r
                break
        if not has_pair and joker_count >= 1:
            for r in ranks:
                if r in high_ranks:
                    has_pair = True
                    pair_rank = r
                    break

        # Full board (3 cards) with FL pair
        if len(top_cards) == 3 and has_pair:
            return 8.0

        # Pair formed but board not yet full
        if has_pair:
            rank_bonus = {'Q': 0, 'K': 0.5, 'A': 1.0}.get(pair_rank, 0)
            return 5.0 + rank_bonus

        # Single high card(s) on top, no pair yet
        if high_count >= 2:
            # Two high cards but no pair — good position
            has_ak = any(r in ('A', 'K') for r in ranks) or joker_count > 0
            return 3.0 if has_ak else 2.0
        elif high_count == 1:
            # One high card — early FL intent
            has_ak = any(r in ('A', 'K') for r in ranks) or joker_count > 0
            return 1.5 if has_ak else 1.0

        return 0.0

    def _load_opponent_model(self, path: str):
        """Load BC policy as opponent model."""
        from ai.models.networks import PolicyNetwork
        model = PolicyNetwork()
        ck = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(ck.get('model_state_dict', ck))
        model.eval()
        self.opponent_model = model

    def _play_opponent(self, seat: int):
        """Play the opponent's turn automatically."""
        obs_obj = self.hand.get_observation(seat)
        board = self.hand.boards[seat]
        cards = self.hand.dealt_cards[seat]

        if self.hand.turn == 0:
            valid_actions = get_initial_actions(cards, board)
        else:
            valid_actions = get_turn_actions(cards, board)

        if not valid_actions:
            return

        if self.opponent_mode == "random":
            action = self._rng.choice(valid_actions)
        elif self.opponent_mode == "model" and self.opponent_model is not None:
            action = self._model_select(obs_obj, valid_actions)
        else:
            action = self._rng.choice(valid_actions)

        self.hand.apply_action(seat, action)

    def _model_select(self, obs: Observation, valid_actions: List[Action]) -> Action:
        """Select action using the opponent model (greedy)."""
        state = encode_state(obs)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        mask = np.zeros(MAX_ACTIONS, dtype=bool)
        mask[:len(valid_actions)] = True
        mask_t = torch.BoolTensor(mask).unsqueeze(0)

        with torch.no_grad():
            probs = self.opponent_model(state_t, mask_t)
            idx = probs.argmax(dim=-1).item()

        if idx < len(valid_actions):
            return valid_actions[idx]
        return valid_actions[0]

    def render(self):
        if self.render_mode == "ansi" and self.hand is not None:
            lines = []
            for seat in [0, 1]:
                b = self.hand.boards[seat]
                lines.append(f"P{seat}: T={b.top} M={b.middle} B={b.bottom}")
            return "\n".join(lines)


class OFCSelfPlayVecWrapper(gym.Wrapper):
    """
    Wrapper that handles the self-play alternation more cleanly
    for PPO training. In self-play mode, the agent controls both
    seats but we always return a canonical observation.

    This wrapper ensures:
    - When seat=1, we negate the reward at hand end (since PPO sees
      the game from the "current player" perspective)
    - This enables symmetric self-play learning
    """

    def __init__(self, env: OFCSelfPlayEnv):
        super().__init__(env)
        self._current_sign = 1.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._current_sign = 1.0  # seat 0 starts
        return obs, info

    def step(self, action):
        # Track which seat we're playing
        seat_before = self.env.current_seat
        self._current_sign = 1.0 if seat_before == 0 else -1.0

        obs, reward, terminated, truncated, info = self.env.step(action)

        # At hand end, reward is from seat-0 perspective.
        # If agent was playing seat-1, negate for symmetric learning.
        if terminated:
            reward *= self._current_sign

        return obs, reward, terminated, truncated, info
