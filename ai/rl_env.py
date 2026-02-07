"""
Gymnasium environment for OFC Pineapple.

This environment allows training RL agents to play OFC Pineapple.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any, List

from game.card import Card, Deck, RANKS, SUITS
from game.board import Board, Row
from game.game_state import GameState, GamePhase, PlayerState
from game.scoring import calculate_round_score


class OFCPineappleEnv(gym.Env):
    """
    Gymnasium environment for OFC Pineapple.
    
    Observation Space:
        - Board state for both players (cards placed)
        - Cards in hand
        - Game phase
    
    Action Space:
        - Discrete: (card_index * 3 + row_index)
        - card_index: 0-16 (max 17 cards in Fantasyland)
        - row_index: 0=top, 1=middle, 2=bottom
    
    Rewards:
        - Score difference at round end
        - Bonus for Fantasyland entry
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    # Constants for encoding
    NUM_RANKS = 13  # 2-A
    NUM_SUITS = 4   # s, h, d, c
    MAX_HAND_SIZE = 17  # Max FL cards
    CARD_FEATURES = NUM_RANKS + NUM_SUITS + 2  # rank one-hot + suit one-hot + joker + empty
    
    def __init__(self, render_mode: Optional[str] = None, opponent_policy: str = "random"):
        super().__init__()
        
        self.render_mode = render_mode
        self.opponent_policy = opponent_policy
        
        # Game state
        self.game = GameState()
        self.current_player = "player"  # Which player the agent controls
        
        # Action space: card_index (0-16) * 3 + row_index (0-2)
        self.action_space = spaces.Discrete(self.MAX_HAND_SIZE * 3)
        
        # Observation space
        # Board: 13 cards * CARD_FEATURES for each player
        # Hand: MAX_HAND_SIZE * CARD_FEATURES
        # Phase: 3 (one-hot for initial, pineapple, fantasyland)
        # In fantasyland: 1
        obs_size = (
            13 * self.CARD_FEATURES * 2 +  # Both boards
            self.MAX_HAND_SIZE * self.CARD_FEATURES +  # Hand
            4 +  # Phase (one-hot)
            1    # In fantasyland flag
        )
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_size,), dtype=np.float32
        )
        
        # For tracking
        self._action_mask = np.zeros(self.action_space.n, dtype=np.int8)
        
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment for new episode."""
        super().reset(seed=seed)
        
        # Reset game
        self.game = GameState()
        self.game.start_new_round()
        
        # Process any AI turns if AI goes first
        self._process_opponent_if_needed()
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Integer action (card_index * 3 + row_index)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        reward = 0.0
        terminated = False
        truncated = False
        
        # Decode action
        card_idx = action // 3
        row_idx = action % 3
        row = [Row.TOP, Row.MIDDLE, Row.BOTTOM][row_idx]
        
        # Get player state
        player = self.game.players[self.current_player]
        
        # Validate action
        if not self._is_valid_action(action):
            # Invalid action penalty
            reward = -10.0
            obs = self._get_observation()
            info = self._get_info()
            return obs, reward, False, False, info
        
        # Place the card
        card = player.hand[card_idx]
        success = self.game.place_cards(self.current_player, [(card, row)])
        
        if not success:
            reward = -10.0
            obs = self._get_observation()
            info = self._get_info()
            return obs, reward, False, False, info
        
        # Check if player has placed required cards for current phase
        if self._check_placement_complete():
            # In pineapple phase, auto-discard remaining card
            if self.game.phase == GamePhase.PINEAPPLE and len(player.hand) == 1:
                self.game.discard_card(self.current_player, player.hand[0])
            
            # Check phase completion and advance
            self.game.check_phase_complete()
            
            # Process opponent
            self._process_opponent_if_needed()
            
            # Check for round end
            if self.game.phase == GamePhase.SCORING:
                self.game.calculate_scores()
                reward = self._calculate_reward()
                self.game.finalize_round()
                terminated = True
            elif self.game.phase == GamePhase.ROUND_END:
                reward = self._calculate_reward()
                terminated = True
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _check_placement_complete(self) -> bool:
        """Check if current placement phase is complete for agent."""
        player = self.game.players[self.current_player]
        
        if self.game.phase == GamePhase.INITIAL_DEAL:
            if player.in_fantasyland:
                # FL: all 13 cards placed (board complete)
                return player.board.is_complete()
            else:
                # Normal: 5 cards placed (hand empty)
                return len(player.hand) == 0
        
        elif self.game.phase == GamePhase.PINEAPPLE:
            # Pineapple: 2 cards placed (1 remaining for discard)
            return len(player.hand) == 1
        
        return False
    
    def _process_opponent_if_needed(self):
        """Process opponent's turn using selected policy."""
        opponent_id = "ai" if self.current_player == "player" else "player"
        opponent = self.game.players[opponent_id]
        
        # Keep processing until opponent has no cards or phase changes
        for _ in range(20):  # Safety limit
            if len(opponent.hand) == 0:
                break
            
            if self.opponent_policy == "random":
                self._random_opponent_play(opponent_id)
            
            # Check phase transitions
            self.game.check_phase_complete()
            
            # If it's scoring phase, stop
            if self.game.phase in [GamePhase.SCORING, GamePhase.ROUND_END]:
                break
    
    def _random_opponent_play(self, opponent_id: str):
        """Random policy for opponent."""
        import random
        opponent = self.game.players[opponent_id]
        
        if not opponent.hand:
            return
        
        if opponent.in_fantasyland:
            # Place all cards
            placements = []
            cards = opponent.hand.copy()
            random.shuffle(cards)
            
            top_count = 0
            mid_count = 0
            bot_count = 0
            
            for card in cards:
                if len(placements) >= 13:
                    break
                if bot_count < 5:
                    placements.append((card, Row.BOTTOM))
                    bot_count += 1
                elif mid_count < 5:
                    placements.append((card, Row.MIDDLE))
                    mid_count += 1
                elif top_count < 3:
                    placements.append((card, Row.TOP))
                    top_count += 1
            
            self.game.place_cards(opponent_id, placements)
            
        elif self.game.phase == GamePhase.INITIAL_DEAL:
            # Place 5 cards randomly
            placements = []
            for card in opponent.hand.copy():
                available = self._get_available_rows(opponent.board, placements)
                if available:
                    row = random.choice(available)
                    placements.append((card, row))
            self.game.place_cards(opponent_id, placements)
            
        elif self.game.phase == GamePhase.PINEAPPLE:
            # Place 2, discard 1
            cards = opponent.hand.copy()
            random.shuffle(cards)
            discard = cards.pop()
            
            placements = []
            for card in cards:
                available = self._get_available_rows(opponent.board, placements)
                if available:
                    row = random.choice(available)
                    placements.append((card, row))
            
            self.game.place_cards(opponent_id, placements)
            self.game.discard_card(opponent_id, discard)
    
    def _get_available_rows(self, board: Board, pending: List) -> List[Row]:
        """Get rows with available space."""
        rows = []
        for row in [Row.TOP, Row.MIDDLE, Row.BOTTOM]:
            current = board.row_count(row) + sum(1 for _, r in pending if r == row)
            max_count = 3 if row == Row.TOP else 5
            if current < max_count:
                rows.append(row)
        return rows
    
    def _get_observation(self) -> np.ndarray:
        """Build observation vector."""
        obs = []
        
        # Encode agent's board
        agent_board = self.game.players[self.current_player].board
        obs.extend(self._encode_board(agent_board))
        
        # Encode opponent's board
        opponent_id = "ai" if self.current_player == "player" else "player"
        opp_board = self.game.players[opponent_id].board
        obs.extend(self._encode_board(opp_board))
        
        # Encode hand
        hand = self.game.players[self.current_player].hand
        obs.extend(self._encode_hand(hand))
        
        # Encode phase
        phase_one_hot = [0, 0, 0, 0]
        if self.game.phase == GamePhase.INITIAL_DEAL:
            phase_one_hot[0] = 1
        elif self.game.phase == GamePhase.PINEAPPLE:
            phase_one_hot[1] = 1
        elif self.game.phase == GamePhase.SCORING:
            phase_one_hot[2] = 1
        else:
            phase_one_hot[3] = 1
        obs.extend(phase_one_hot)
        
        # Fantasyland flag
        in_fl = 1.0 if self.game.players[self.current_player].in_fantasyland else 0.0
        obs.append(in_fl)
        
        return np.array(obs, dtype=np.float32)
    
    def _encode_card(self, card: Optional[Card]) -> List[float]:
        """Encode a single card as feature vector."""
        features = [0.0] * self.CARD_FEATURES
        
        if card is None:
            features[-1] = 1.0  # Empty slot
        elif card.is_joker:
            features[-2] = 1.0  # Joker flag
        else:
            # Rank one-hot
            rank_idx = RANKS.index(card.rank)
            features[rank_idx] = 1.0
            # Suit one-hot
            suit_idx = SUITS.index(card.suit)
            features[self.NUM_RANKS + suit_idx] = 1.0
        
        return features
    
    def _encode_board(self, board: Board) -> List[float]:
        """Encode a board (13 cards)."""
        features = []
        
        # Top (3 cards)
        for i in range(3):
            card = board.top[i] if i < len(board.top) else None
            features.extend(self._encode_card(card))
        
        # Middle (5 cards)
        for i in range(5):
            card = board.middle[i] if i < len(board.middle) else None
            features.extend(self._encode_card(card))
        
        # Bottom (5 cards)
        for i in range(5):
            card = board.bottom[i] if i < len(board.bottom) else None
            features.extend(self._encode_card(card))
        
        return features
    
    def _encode_hand(self, hand: List[Card]) -> List[float]:
        """Encode hand (up to MAX_HAND_SIZE cards)."""
        features = []
        
        for i in range(self.MAX_HAND_SIZE):
            card = hand[i] if i < len(hand) else None
            features.extend(self._encode_card(card))
        
        return features
    
    def _is_valid_action(self, action: int) -> bool:
        """Check if action is valid in current state."""
        card_idx = action // 3
        row_idx = action % 3
        row = [Row.TOP, Row.MIDDLE, Row.BOTTOM][row_idx]
        
        player = self.game.players[self.current_player]
        
        # Check card exists
        if card_idx >= len(player.hand):
            return False
        
        # Check row has space
        board = player.board
        current = board.row_count(row)
        max_count = 3 if row == Row.TOP else 5
        
        return current < max_count
    
    def action_masks(self) -> np.ndarray:
        """Return mask of valid actions (for masked PPO)."""
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        
        player = self.game.players[self.current_player]
        
        for card_idx in range(len(player.hand)):
            for row_idx, row in enumerate([Row.TOP, Row.MIDDLE, Row.BOTTOM]):
                if player.board.row_count(row) < (3 if row == Row.TOP else 5):
                    action = card_idx * 3 + row_idx
                    mask[action] = 1
        
        return mask
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward at end of round.
        
        Reward priorities:
        1. FL Stay (継続) - highest priority
        2. FL Entry (進入) - second priority  
        3. Royalties / Score - base reward
        4. Bust - large penalty
        """
        if self.game.last_scores is None:
            return 0.0
        
        scores = self.game.last_scores
        player = self.game.players[self.current_player]
        
        # Check if player busted
        is_bust = scores["player_bust"] if self.current_player == "player" else scores["ai_bust"]
        
        if is_bust:
            # Large penalty for bust (-20 base + lose all royalty potential)
            return -30.0
        
        # Base reward: score difference
        if self.current_player == "player":
            reward = float(scores["player_net"])
        else:
            reward = float(scores["ai_net"])
        
        # FL Stay bonus (最優先 - 継続)
        # Was in FL and stayed in FL
        was_in_fl = player.board.is_complete() and player.in_fantasyland
        if was_in_fl:
            # Check if conditions to stay were met
            if player.board.get_fantasyland_stay():
                reward += 25.0  # Large bonus for staying
            else:
                reward -= 10.0  # Penalty for failing to stay when in FL
        
        # FL Entry bonus (次に優先 - 進入)
        # Newly entered FL this round
        elif player.in_fantasyland and player.fantasyland_cards > 0:
            # Scale by FL cards: QQ(14)=+10, KK(15)=+12, AA(16)=+14, Trips(17)=+16
            fl_entry_bonus = 8.0 + (player.fantasyland_cards - 13) * 2.0
            reward += fl_entry_bonus
        
        return reward
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info dict for debugging."""
        player = self.game.players[self.current_player]
        
        return {
            "phase": self.game.phase.value,
            "hand_size": len(player.hand),
            "board_complete": player.board.is_complete(),
            "in_fantasyland": player.in_fantasyland,
            "valid_actions": int(self.action_masks().sum()),
        }
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            print(self._render_ansi())
    
    def _render_ansi(self) -> str:
        """Render as ASCII string."""
        lines = []
        lines.append(f"Phase: {self.game.phase.value}")
        lines.append(f"Round: {self.game.round_number}")
        
        for pid in ["player", "ai"]:
            player = self.game.players[pid]
            lines.append(f"\n{player.name} (Stack: {player.stack}):")
            lines.append(f"  Top: {[str(c) for c in player.board.top]}")
            lines.append(f"  Mid: {[str(c) for c in player.board.middle]}")
            lines.append(f"  Bot: {[str(c) for c in player.board.bottom]}")
            if pid == self.current_player:
                lines.append(f"  Hand: {[str(c) for c in player.hand]}")
        
        return "\n".join(lines)


# Register the environment
def register_env():
    """Register the environment with gymnasium."""
    gym.register(
        id="OFCPineapple-v0",
        entry_point="ai.rl_env:OFCPineappleEnv",
    )
