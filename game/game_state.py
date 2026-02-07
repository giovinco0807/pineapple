"""Game state management for OFC Pineapple."""
from typing import List, Optional, Dict, Tuple
from enum import Enum
from dataclasses import dataclass, field
from .card import Card, Deck
from .board import Board, Row


class GamePhase(Enum):
    """Current phase of the game."""
    WAITING = "waiting"           # Waiting to start
    INITIAL_DEAL = "initial"      # First 5 cards dealt
    PINEAPPLE = "pineapple"       # 3 cards dealt, place 2, discard 1
    SCORING = "scoring"           # All cards placed, calculating scores
    ROUND_END = "round_end"       # Round completed
    SESSION_END = "session_end"   # Session ended


@dataclass
class PlayerState:
    """State for a single player."""
    name: str
    is_ai: bool = False
    stack: int = 200
    board: Board = field(default_factory=Board)
    hand: List[Card] = field(default_factory=list)  # Cards in hand waiting to be placed
    in_fantasyland: bool = False
    fantasyland_cards: int = 0  # Number of cards to deal in fantasyland
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'is_ai': self.is_ai,
            'stack': self.stack,
            'board': self.board.to_dict(),
            'hand': [c.to_dict() for c in self.hand],
            'in_fantasyland': self.in_fantasyland,
            'fantasyland_cards': self.fantasyland_cards
        }


class GameState:
    """
    Manages the overall game state for OFC Pineapple.
    
    Game flow:
    1. Initial deal: Each player gets 5 cards, places all
    2. Pineapple rounds (x4): Each player gets 3 cards, places 2, discards 1
    3. Scoring: Compare hands, calculate points
    4. Check Fantasyland, rotate dealer, next round
    """
    
    INITIAL_STACK = 200
    
    def __init__(self, player_name: str = "Player"):
        self.players: Dict[str, PlayerState] = {
            'player': PlayerState(name=player_name, is_ai=False),
            'ai': PlayerState(name="AI", is_ai=True)
        }
        self.deck = Deck()
        self.phase = GamePhase.WAITING
        self.dealer = 'player'  # Who is dealer this round
        self.pineapple_round = 0  # Current pineapple round (1-4)
        self.round_number = 0
        self.discards: List[Card] = []
        self.last_scores: Optional[Dict] = None
        self.session_can_end = False
    
    def start_new_round(self):
        """Start a new round."""
        self.round_number += 1
        self.pineapple_round = 0
        self.discards = []
        self.last_scores = None
        
        # Clear boards
        for p in self.players.values():
            p.board.clear()
            p.hand = []
        
        # Reset and shuffle deck
        self.deck.reset()
        self.deck.shuffle()
        
        # Handle Fantasyland
        player_fl = self.players['player'].in_fantasyland
        ai_fl = self.players['ai'].in_fantasyland
        
        if player_fl or ai_fl:
            self._deal_fantasyland()
        else:
            self._deal_initial()
    
    def _deal_initial(self):
        """Deal initial 5 cards to each player."""
        self.phase = GamePhase.INITIAL_DEAL
        
        # Deal 5 cards to each player
        self.players['player'].hand = self.deck.deal(5)
        self.players['ai'].hand = self.deck.deal(5)
    
    def _deal_fantasyland(self):
        """Deal cards for Fantasyland players.
        
        Important: FL player doesn't get cards until non-FL player completes their board.
        Initially deal only to non-FL player.
        """
        self.phase = GamePhase.INITIAL_DEAL
        
        for pid, player in self.players.items():
            if player.in_fantasyland:
                # FL player waits - don't deal yet
                player.hand = []
            else:
                # Normal initial deal
                player.hand = self.deck.deal(5)
    
    def _deal_fantasyland_cards(self):
        """Deal cards to Fantasyland player after non-FL player is done."""
        for pid, player in self.players.items():
            if player.in_fantasyland and len(player.hand) == 0:
                player.hand = self.deck.deal(player.fantasyland_cards)
    
    def _deal_pineapple(self):
        """Deal 3 cards for pineapple round."""
        self.pineapple_round += 1
        self.phase = GamePhase.PINEAPPLE
        
        for pid, player in self.players.items():
            if player.in_fantasyland:
                # Fantasyland player doesn't participate in pineapple
                continue
            player.hand = self.deck.deal(3)
    
    def place_cards(self, player_id: str, placements: List[Tuple[Card, Row]]) -> bool:
        """
        Place cards on the board.
        
        Args:
            player_id: 'player' or 'ai'
            placements: List of (card, row) tuples
        
        Returns:
            True if successful
        """
        player = self.players.get(player_id)
        if not player:
            return False
        
        # Validate cards are in hand
        placement_cards = [c for c, r in placements]
        for card in placement_cards:
            if card not in player.hand:
                return False
        
        # Place cards
        for card, row in placements:
            if not player.board.place_card(row, card):
                return False
            player.hand.remove(card)
        
        return True
    
    def discard_card(self, player_id: str, card: Card) -> bool:
        """
        Discard a card (pineapple phase).
        
        Returns:
            True if successful
        """
        player = self.players.get(player_id)
        if not player:
            return False
        
        if card not in player.hand:
            return False
        
        player.hand.remove(card)
        self.discards.append(card)
        return True
    
    def _is_non_fl_player_complete(self) -> bool:
        """Check if non-Fantasyland player has completed their board."""
        for player in self.players.values():
            if not player.in_fantasyland:
                return player.board.is_complete()
        return True  # All players are in FL
    
    def _both_in_fantasyland(self) -> bool:
        """Check if both players are in Fantasyland."""
        return all(p.in_fantasyland for p in self.players.values())
    
    def _any_in_fantasyland(self) -> bool:
        """Check if any player is in Fantasyland."""
        return any(p.in_fantasyland for p in self.players.values())
    
    def check_phase_complete(self) -> bool:
        """
        Check if current phase is complete and advance if so.
        
        Returns:
            True if phase advanced
        """
        if self.phase == GamePhase.INITIAL_DEAL:
            any_fl = self._any_in_fantasyland()
            both_fl = self._both_in_fantasyland()
            
            if any_fl and not both_fl:
                # Mixed FL/non-FL game
                non_fl_done = all(
                    p.in_fantasyland or len(p.hand) == 0
                    for p in self.players.values()
                )
                
                if non_fl_done and not self._is_non_fl_player_complete():
                    # Non-FL player finished initial, but board not complete
                    # Continue to pineapple for non-FL player
                    self._deal_pineapple()
                    return True
                elif non_fl_done and self._is_non_fl_player_complete():
                    # Non-FL player's board is complete, now FL player can place
                    fl_has_cards = any(
                        p.in_fantasyland and len(p.hand) > 0
                        for p in self.players.values()
                    )
                    if not fl_has_cards:
                        # Deal cards to FL player now
                        self._deal_fantasyland_cards()
                        return True
                    else:
                        # FL player has cards, check if done
                        if all(p.board.is_complete() for p in self.players.values()):
                            self.phase = GamePhase.SCORING
                            return True
                return False
            
            elif both_fl:
                # Both in FL - they place simultaneously
                all_placed = all(len(p.hand) == 0 for p in self.players.values())
                if all_placed:
                    self.phase = GamePhase.SCORING
                    return True
                return False
            
            else:
                # Normal game (no FL)
                all_placed = all(len(p.hand) == 0 for p in self.players.values())
                if all_placed:
                    self._deal_pineapple()
                    return True
                return False
        
        elif self.phase == GamePhase.PINEAPPLE:
            any_fl = self._any_in_fantasyland()
            both_fl = self._both_in_fantasyland()
            
            # Check if non-FL hands are empty
            non_fl_done = all(
                p.in_fantasyland or len(p.hand) == 0
                for p in self.players.values()
            )
            
            if not non_fl_done:
                return False
            
            if any_fl and not both_fl:
                # Mixed game - check if non-FL player is complete
                if self._is_non_fl_player_complete():
                    # Non-FL done, deal to FL player
                    fl_has_cards = any(
                        p.in_fantasyland and len(p.hand) > 0
                        for p in self.players.values()
                    )
                    if not fl_has_cards:
                        self._deal_fantasyland_cards()
                        self.phase = GamePhase.INITIAL_DEAL  # FL placement phase
                        return True
                elif self.pineapple_round < 4:
                    self._deal_pineapple()
                    return True
            else:
                # Normal game
                if all(p.board.is_complete() for p in self.players.values()):
                    self.phase = GamePhase.SCORING
                    return True
                elif self.pineapple_round < 4:
                    self._deal_pineapple()
                    return True
        
        return False
    
    def calculate_scores(self) -> Dict:
        """
        Calculate round scores.
        
        Returns:
            Dict with scoring details
        """
        from .scoring import calculate_round_score
        
        player_board = self.players['player'].board
        ai_board = self.players['ai'].board
        player_stack = self.players['player'].stack
        ai_stack = self.players['ai'].stack
        
        result = calculate_round_score(player_board, ai_board, player_stack, ai_stack)
        
        # Apply scores
        self.players['player'].stack += result['player_net']
        self.players['ai'].stack += result['ai_net']
        
        self.last_scores = result
        return result
    
    def finalize_round(self):
        """Finalize round, check Fantasyland, rotate dealer."""
        # Check Fantasyland entry/stay for both players
        for pid, player in self.players.items():
            if player.board.is_bust():
                player.in_fantasyland = False
                player.fantasyland_cards = 0
            elif player.in_fantasyland:
                # Check stay
                if player.board.get_fantasyland_stay():
                    player.fantasyland_cards = 14  # Standard FL cards for stay
                else:
                    player.in_fantasyland = False
                    player.fantasyland_cards = 0
            else:
                # Check entry
                fl_cards = player.board.get_fantasyland_entry()
                if fl_cards > 0:
                    player.in_fantasyland = True
                    player.fantasyland_cards = fl_cards
        
        # Check if session can end
        has_score_change = self.last_scores and self.last_scores['player_net'] != 0
        any_fantasyland = any(p.in_fantasyland for p in self.players.values())
        self.session_can_end = has_score_change and not any_fantasyland
        
        # Rotate dealer
        self.dealer = 'ai' if self.dealer == 'player' else 'player'
        
        self.phase = GamePhase.ROUND_END
    
    def can_end_session(self) -> bool:
        """Check if session can be ended."""
        return self.session_can_end
    
    def end_session(self):
        """End the session and reset stacks."""
        self.phase = GamePhase.SESSION_END
        # Reset stacks to initial value
        for player_state in self.players.values():
            player_state.stack = 200
    
    def get_state_for_player(self, player_id: str, hide_opponent: bool = True) -> dict:
        """
        Get game state from a player's perspective.
        
        Args:
            player_id: 'player' or 'ai'
            hide_opponent: Whether to hide opponent's hand
        """
        opponent_id = 'ai' if player_id == 'player' else 'player'
        
        state = {
            'phase': self.phase.value,
            'round_number': self.round_number,
            'pineapple_round': self.pineapple_round,
            'dealer': self.dealer,
            'player': self.players[player_id].to_dict(),
            'session_can_end': self.session_can_end,
            'last_scores': self.last_scores
        }
        
        opponent_state = self.players[opponent_id].to_dict()
        if hide_opponent and self.phase not in [GamePhase.SCORING, GamePhase.ROUND_END]:
            # Hide opponent's hand
            opponent_state['hand'] = [{'hidden': True} for _ in opponent_state['hand']]
        state['opponent'] = opponent_state
        
        return state
    
    def to_dict(self) -> dict:
        """Full state serialization."""
        return {
            'phase': self.phase.value,
            'round_number': self.round_number,
            'pineapple_round': self.pineapple_round,
            'dealer': self.dealer,
            'players': {pid: p.to_dict() for pid, p in self.players.items()},
            'discards': [c.to_dict() for c in self.discards],
            'session_can_end': self.session_can_end,
            'last_scores': self.last_scores
        }
