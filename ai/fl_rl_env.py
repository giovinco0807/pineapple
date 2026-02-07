"""
Fantasyland RL Environment

Gymnasium environment for training RL agents to solve Fantasyland placement.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add ai directory to path for fl_solver
sys.path.insert(0, str(Path(__file__).parent))

from game.card import Card, Deck, RANKS, SUITS
from game.board import Board, Row
from game.hand_evaluator import evaluate_5_card_hand, evaluate_3_card_hand
from game.royalty import get_top_royalty, get_middle_royalty, get_bottom_royalty, check_fantasyland_stay
from game.joker_optimizer import optimize_board
from fl_solver import solve_fantasyland_exhaustive


# Constants
NUM_RANKS = 13
NUM_SUITS = 4
JOKER_FEATURE = 1
CARD_FEATURES = NUM_RANKS + NUM_SUITS + JOKER_FEATURE  # 18 features per card
MAX_CARDS = 17  # Maximum cards in FL hand
POSITIONS = 4  # Top, Middle, Bottom, Discard


class FantasylandEnv(gym.Env):
    """
    Gymnasium environment for Fantasyland card placement.
    
    The agent receives 14-17 cards and must place them:
    - Top: 3 cards
    - Middle: 5 cards  
    - Bottom: 5 cards
    - Discard: remaining cards
    
    The goal is to maximize royalties while avoiding bust and staying in FL.
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        num_cards: int = 14,
        include_jokers: bool = True,
        fl_stay_bonus: float = 30.0,
        bust_penalty: float = -100.0,
    ):
        super().__init__()
        
        self.render_mode = render_mode
        self.num_cards = num_cards
        self.include_jokers = include_jokers
        self.fl_stay_bonus = fl_stay_bonus
        self.bust_penalty = bust_penalty
        
        # Observation space: flattened card features for sb3 compatibility
        # Each card: CARD_FEATURES + POSITIONS = 22 features
        # Total: MAX_CARDS * 22 = 374 features
        obs_size = MAX_CARDS * (CARD_FEATURES + POSITIONS)
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(obs_size,),
            dtype=np.float32
        )
        
        # Action space: which card (0-16) goes to which position (0-3)
        # Action = card_idx * 4 + position
        # Position: 0=Top, 1=Middle, 2=Bottom, 3=Discard
        self.action_space = spaces.Discrete(MAX_CARDS * POSITIONS)
        
        # Episode state
        self.hand: List[Card] = []
        self.top: List[int] = []      # Indices of cards placed in top
        self.middle: List[int] = []   # Indices of cards placed in middle
        self.bottom: List[int] = []   # Indices of cards placed in bottom
        self.discards: List[int] = [] # Indices of discarded cards
        self.placed: set = set()      # Set of placed card indices
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment with a new FL hand."""
        super().reset(seed=seed)
        
        # Deal new hand
        deck = Deck(include_jokers=self.include_jokers)
        deck.shuffle()
        self.hand = deck.deal(self.num_cards)
        
        # Reset placement state
        self.top = []
        self.middle = []
        self.bottom = []
        self.discards = []
        self.placed = set()
        
        # Pre-compute optimal placement using solver (fast action masking)
        self._compute_optimal_placement()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def _compute_optimal_placement(self):
        """Use solver to find optimal placement at episode start."""
        solutions = solve_fantasyland_exhaustive(self.hand, max_solutions=1)
        
        if solutions:
            best = solutions[0]
            # Map cards back to indices
            self.optimal_top = self._cards_to_indices(best.top)
            self.optimal_middle = self._cards_to_indices(best.middle)
            self.optimal_bottom = self._cards_to_indices(best.bottom)
            self.optimal_discards = self._cards_to_indices(best.discards)
        else:
            # Fallback: no valid solution found
            self.optimal_top = []
            self.optimal_middle = []
            self.optimal_bottom = []
            self.optimal_discards = []
    
    def _cards_to_indices(self, cards: List[Card]) -> List[int]:
        """Convert cards to their indices in self.hand."""
        indices = []
        for card in cards:
            for i, h in enumerate(self.hand):
                if h == card and i not in indices:
                    indices.append(i)
                    break
        return indices
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Take a step by placing a card.
        
        Args:
            action: card_idx * 4 + position
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        card_idx = action // POSITIONS
        position = action % POSITIONS
        
        # BUG FIX #3: マスクされたアクションが選ばれていないか確認
        masks = self.action_masks()
        if not masks[action]:
            print(f"[DEBUG] WARNING: 禁止アクション {action} が選択された！")
            print(f"[DEBUG] カード {card_idx} → 位置 {position}")
            print(f"[DEBUG] 有効なアクション数: {masks.sum()}")
            print(f"[DEBUG] 現在の配置: top={len(self.top)}, mid={len(self.middle)}, bot={len(self.bottom)}")
        
        # Validate action
        if not self._is_valid_action(action):
            # Invalid action - small penalty and skip
            return self._get_observation(), -1.0, False, False, self._get_info()
        
        # Place the card
        self.placed.add(card_idx)
        
        if position == 0:  # Top
            self.top.append(card_idx)
        elif position == 1:  # Middle
            self.middle.append(card_idx)
        elif position == 2:  # Bottom
            self.bottom.append(card_idx)
        else:  # Discard
            self.discards.append(card_idx)
        
        # Check if episode is done
        terminated = self._is_placement_complete()
        
        # Calculate reward only at end
        if terminated:
            reward = self._calculate_final_reward()
            # バーストした場合のデバッグログ
            if reward == self.bust_penalty:
                print(f"[DEBUG] BUST DETECTED!")
                print(f"[DEBUG] Top: {[str(self.hand[i]) for i in self.top]}")
                print(f"[DEBUG] Middle: {[str(self.hand[i]) for i in self.middle]}")
                print(f"[DEBUG] Bottom: {[str(self.hand[i]) for i in self.bottom]}")
        else:
            reward = 0.0  # Intermediate steps get no reward
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, False, info
    
    def _is_placement_complete(self) -> bool:
        """Check if all cards are placed."""
        return (
            len(self.top) == 3 and
            len(self.middle) == 5 and
            len(self.bottom) == 5
        )
    
    def _is_valid_action(self, action: int) -> bool:
        """Check if action is valid."""
        card_idx = action // POSITIONS
        position = action % POSITIONS
        
        # Card must exist and not be placed
        if card_idx >= len(self.hand) or card_idx in self.placed:
            return False
        
        # Check position capacity
        if position == 0 and len(self.top) >= 3:
            return False
        if position == 1 and len(self.middle) >= 5:
            return False
        if position == 2 and len(self.bottom) >= 5:
            return False
        if position == 3:
            # Discard only allowed if not all 13 positions filled
            total_placed = len(self.top) + len(self.middle) + len(self.bottom)
            remaining_cards = len(self.hand) - len(self.placed)
            positions_left = 13 - total_placed
            if remaining_cards <= positions_left:
                return False  # Must place in board, can't discard
        
        return True
    
    def action_masks(self) -> np.ndarray:
        """
        Return mask of valid actions using pre-computed optimal placement.
        Only allows actions that follow the solver's optimal solution.
        This is O(1) instead of O(n!) - much faster!
        """
        mask = np.zeros(MAX_CARDS * POSITIONS, dtype=bool)
        
        # Find the next cards to place based on optimal solution
        for card_idx in self.optimal_top:
            if card_idx not in self.placed and len(self.top) < 3:
                action = card_idx * POSITIONS + 0  # Top
                if self._is_valid_action(action):
                    mask[action] = True
        
        for card_idx in self.optimal_middle:
            if card_idx not in self.placed and len(self.middle) < 5:
                action = card_idx * POSITIONS + 1  # Middle
                if self._is_valid_action(action):
                    mask[action] = True
        
        for card_idx in self.optimal_bottom:
            if card_idx not in self.placed and len(self.bottom) < 5:
                action = card_idx * POSITIONS + 2  # Bottom
                if self._is_valid_action(action):
                    mask[action] = True
        
        for card_idx in self.optimal_discards:
            if card_idx not in self.placed:
                action = card_idx * POSITIONS + 3  # Discard
                if self._is_valid_action(action):
                    mask[action] = True
        
        # Fallback: if no optimal actions (shouldn't happen), allow any valid
        if not mask.any():
            for action in range(MAX_CARDS * POSITIONS):
                if self._is_valid_action(action):
                    mask[action] = True
        
        return mask
    
    def _can_complete_without_bust(self, action: int) -> bool:
        """
        Check if taking this action can lead to a valid (non-busting) final state.
        Does exhaustive simulation of greedy placement after this action.
        """
        card_idx = action // POSITIONS
        position = action % POSITIONS
        
        # Simulate the action
        sim_top = list(self.top)
        sim_middle = list(self.middle)
        sim_bottom = list(self.bottom)
        sim_discards = list(self.discards)
        sim_placed = set(self.placed)
        
        if position == 0:
            sim_top.append(card_idx)
        elif position == 1:
            sim_middle.append(card_idx)
        elif position == 2:
            sim_bottom.append(card_idx)
        else:
            sim_discards.append(card_idx)
        
        sim_placed.add(card_idx)
        
        # 残りのカードを取得
        remaining = [i for i in range(len(self.hand)) if i not in sim_placed]
        
        # グリーディに残りのカードを配置してバーストしないか確認
        return self._try_all_completions(sim_top, sim_middle, sim_bottom, sim_discards, remaining)
    
    def _try_all_completions(self, top, middle, bottom, discards, remaining) -> bool:
        """
        Try all possible completions and check if any doesn't bust.
        Returns True if at least one valid completion exists.
        """
        # ベースケース：全て配置完了
        if len(top) == 3 and len(middle) == 5 and len(bottom) == 5:
            top_cards = [self.hand[i] for i in top]
            mid_cards = [self.hand[i] for i in middle]
            bot_cards = [self.hand[i] for i in bottom]
            _, _, _, is_bust = optimize_board(top_cards, mid_cards, bot_cards)
            return not is_bust
        
        # BUG FIX #1: カードが残っていないのにボードが未完成 → バースト確定
        if not remaining:
            return False  # 未完成なのでバースト
        
        # 次のカードを各位置に試す
        card_idx = remaining[0]
        new_remaining = remaining[1:]
        
        # BUG FIX #2: 全ての順列を試す（固定順序ではなく）
        # 利用可能な全位置
        available_positions = []
        
        if len(top) < 3:
            available_positions.append('top')
        if len(middle) < 5:
            available_positions.append('middle')
        if len(bottom) < 5:
            available_positions.append('bottom')
        
        # ディスカードも考慮
        total_placed = len(top) + len(middle) + len(bottom)
        remaining_cards = len(remaining)
        positions_left = 13 - total_placed
        if remaining_cards > positions_left:
            available_positions.append('discard')
        
        # 全ての位置を試す
        for pos_name in available_positions:
            if pos_name == 'top':
                new_top = list(top) + [card_idx]
                new_middle = middle
                new_bottom = bottom
                new_discards = discards
            elif pos_name == 'middle':
                new_top = top
                new_middle = list(middle) + [card_idx]
                new_bottom = bottom
                new_discards = discards
            elif pos_name == 'bottom':
                new_top = top
                new_middle = middle
                new_bottom = list(bottom) + [card_idx]
                new_discards = discards
            else:
                new_top = top
                new_middle = middle
                new_bottom = bottom
                new_discards = list(discards) + [card_idx]
            
            if self._try_all_completions(new_top, new_middle, new_bottom, new_discards, new_remaining):
                return True
        
        return False
    
    def _calculate_final_reward(self) -> float:
        """Calculate reward at episode end with shaped rewards."""
        # Get actual cards
        top_cards = [self.hand[i] for i in self.top]
        mid_cards = [self.hand[i] for i in self.middle]
        bot_cards = [self.hand[i] for i in self.bottom]
        
        # Optimize for jokers
        opt_top, opt_mid, opt_bot, is_bust = optimize_board(top_cards, mid_cards, bot_cards)
        
        # PRIORITY 1: バーストしない
        if is_bust or opt_top is None:
            return self.bust_penalty
        
        # Calculate royalties
        top_roy = get_top_royalty(opt_top)
        mid_roy = get_middle_royalty(opt_mid)
        bot_roy = get_bottom_royalty(opt_bot)
        total_royalties = top_roy + mid_roy + bot_roy
        
        # Check FL stay
        can_stay = check_fantasyland_stay(opt_top, opt_bot)
        
        if can_stay:
            # FL Stay達成! 大きなボーナス
            fl_bonus = self.fl_stay_bonus
        else:
            # FL Stayを目指す途中の部分報酬（shaped reward）
            fl_bonus = self._calculate_shaped_fl_bonus(opt_top, opt_bot)
        
        return total_royalties + fl_bonus
    
    def _calculate_shaped_fl_bonus(self, top_cards, bot_cards) -> float:
        """FL Stay を目指す途中の部分報酬を計算"""
        bonus = 0.0
        
        # ボトムのハンド強度をチェック
        # ロイヤルストレートフラッシュ > ストフラ > クワッズ
        from game.hand_evaluator import evaluate_5_card_hand, evaluate_3_card_hand, HandRank, HandRank3
        
        bot_rank, _ = evaluate_5_card_hand(bot_cards)
        
        # ボトムの報酬（クワッズ以上でFL Stay可能）
        if bot_rank == HandRank.ROYAL_FLUSH:
            bonus += 15.0  # Royal Flush!
        elif bot_rank == HandRank.STRAIGHT_FLUSH:
            bonus += 15.0  # SF on bottom - close to FL Stay!
        elif bot_rank == HandRank.FOUR_OF_A_KIND:
            bonus += 12.0  # Quads on bottom - FL Stay possible!
        elif bot_rank == HandRank.FULL_HOUSE:
            bonus += 5.0   # Full House - good but not FL Stay
        elif bot_rank == HandRank.FLUSH:
            bonus += 3.0   # Flush - decent
        elif bot_rank == HandRank.STRAIGHT:
            bonus += 2.0   # Straight - decent
        
        # トップのハンド強度をチェック（トリップスでFL Stay可能）
        top_rank, _ = evaluate_3_card_hand(top_cards)
        
        if top_rank == HandRank3.THREE_OF_A_KIND:
            bonus += 12.0  # Trips on top - FL Stay possible!
        elif top_rank == HandRank3.PAIR:
            bonus += 2.0   # Pair on top - getting there
        
        return bonus
    
    def _get_observation(self) -> np.ndarray:
        """Build flattened observation."""
        # Card features
        cards_obs = np.zeros((MAX_CARDS, CARD_FEATURES + POSITIONS), dtype=np.float32)
        
        for i, card in enumerate(self.hand):
            # Encode card
            cards_obs[i, :CARD_FEATURES] = self._encode_card(card)
            
            # Encode placement status
            if i in self.top:
                cards_obs[i, CARD_FEATURES + 0] = 1.0
            elif i in self.middle:
                cards_obs[i, CARD_FEATURES + 1] = 1.0
            elif i in self.bottom:
                cards_obs[i, CARD_FEATURES + 2] = 1.0
            elif i in self.discards:
                cards_obs[i, CARD_FEATURES + 3] = 1.0
        
        # Flatten to 1D array
        return cards_obs.flatten()
    
    def _encode_card(self, card: Card) -> np.ndarray:
        """Encode a single card as feature vector."""
        features = np.zeros(CARD_FEATURES, dtype=np.float32)
        
        if card.is_joker:
            features[-1] = 1.0  # Joker flag
        else:
            # One-hot rank
            rank_idx = RANKS.index(card.rank)
            features[rank_idx] = 1.0
            
            # One-hot suit
            suit_idx = SUITS.index(card.suit)
            features[NUM_RANKS + suit_idx] = 1.0
        
        return features
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info dict."""
        return {
            "n_cards": len(self.hand),
            "placed": len(self.placed),
            "top_count": len(self.top),
            "mid_count": len(self.middle),
            "bot_count": len(self.bottom),
            "discard_count": len(self.discards),
        }
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            print(self._render_ansi())
        elif self.render_mode == "ansi":
            return self._render_ansi()
    
    def _render_ansi(self) -> str:
        """Render as ASCII string."""
        lines = []
        lines.append("=" * 40)
        lines.append(f"FL Hand: {len(self.hand)} cards")
        
        def cards_str(indices):
            return " ".join(str(self.hand[i]) for i in indices)
        
        lines.append(f"Top ({len(self.top)}/3): {cards_str(self.top)}")
        lines.append(f"Mid ({len(self.middle)}/5): {cards_str(self.middle)}")
        lines.append(f"Bot ({len(self.bottom)}/5): {cards_str(self.bottom)}")
        lines.append(f"Discard: {cards_str(self.discards)}")
        
        unplaced = [i for i in range(len(self.hand)) if i not in self.placed]
        lines.append(f"Unplaced: {cards_str(unplaced)}")
        lines.append("=" * 40)
        
        return "\n".join(lines)


# Register the environment
def register_fl_env():
    """Register the FL environment with gymnasium."""
    try:
        gym.register(
            id="Fantasyland-v0",
            entry_point="ai.fl_rl_env:FantasylandEnv",
        )
    except gym.error.Error:
        pass  # Already registered


if __name__ == "__main__":
    # Test the environment
    env = FantasylandEnv(render_mode="human")
    obs, info = env.reset(seed=42)
    
    print("Initial observation:")
    print(f"  obs shape: {obs.shape}")
    
    # Random play test
    done = False
    total_reward = 0
    steps = 0
    
    while not done:
        mask = env.action_masks()
        valid_actions = np.where(mask)[0]
        
        if len(valid_actions) == 0:
            print("No valid actions!")
            break
        
        action = np.random.choice(valid_actions)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
        
        env.render()
    
    print(f"\nEpisode finished in {steps} steps")
    print(f"Total reward: {total_reward}")
