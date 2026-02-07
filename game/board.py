"""Player board for OFC Pineapple."""
from typing import List, Optional, Dict, Tuple
from enum import Enum
from .card import Card, has_joker
from .hand_evaluator import (
    evaluate_3_card_hand, evaluate_5_card_hand,
    compare_hands_3, compare_hands_5,
    HandRank, HandRank3
)
from .royalty import (
    get_top_royalty, get_middle_royalty, get_bottom_royalty,
    check_fantasyland_entry, check_fantasyland_stay
)


class Row(Enum):
    """Board rows."""
    TOP = "top"
    MIDDLE = "middle"
    BOTTOM = "bottom"


class Board:
    """
    Represents a player's OFC board with three rows:
    - Top: 3 cards
    - Middle: 5 cards
    - Bottom: 5 cards
    
    Supports jokers (wild cards) with automatic optimization.
    """
    
    MAX_TOP = 3
    MAX_MIDDLE = 5
    MAX_BOTTOM = 5
    
    def __init__(self):
        self.top: List[Card] = []
        self.middle: List[Card] = []
        self.bottom: List[Card] = []
        # Cache for optimized hands (cleared when board changes)
        self._optimized_cache: Optional[Tuple] = None
    
    def _clear_cache(self):
        """Clear the optimization cache."""
        self._optimized_cache = None
    
    def place_card(self, row: Row, card: Card) -> bool:
        """
        Place a card in a row.
        
        Returns:
            True if successful, False if row is full
        """
        self._clear_cache()
        if row == Row.TOP:
            if len(self.top) >= self.MAX_TOP:
                return False
            self.top.append(card)
        elif row == Row.MIDDLE:
            if len(self.middle) >= self.MAX_MIDDLE:
                return False
            self.middle.append(card)
        elif row == Row.BOTTOM:
            if len(self.bottom) >= self.MAX_BOTTOM:
                return False
            self.bottom.append(card)
        return True
    
    def remove_card(self, row: Row, card: Card) -> bool:
        """
        Remove a card from a row.
        
        Returns:
            True if successful, False if card not found
        """
        self._clear_cache()
        target = self._get_row(row)
        if card in target:
            target.remove(card)
            return True
        return False
    
    def _get_row(self, row: Row) -> List[Card]:
        """Get the card list for a row."""
        if row == Row.TOP:
            return self.top
        elif row == Row.MIDDLE:
            return self.middle
        else:
            return self.bottom
    
    def row_count(self, row: Row) -> int:
        """Get number of cards in a row."""
        return len(self._get_row(row))
    
    def row_max(self, row: Row) -> int:
        """Get maximum cards for a row."""
        if row == Row.TOP:
            return self.MAX_TOP
        return self.MAX_MIDDLE  # Same for middle and bottom
    
    def row_available(self, row: Row) -> int:
        """Get number of available slots in a row."""
        return self.row_max(row) - self.row_count(row)
    
    def is_row_complete(self, row: Row) -> bool:
        """Check if a row is complete."""
        return self.row_count(row) == self.row_max(row)
    
    def is_complete(self) -> bool:
        """Check if all rows are complete (13 cards total)."""
        return (len(self.top) == self.MAX_TOP and
                len(self.middle) == self.MAX_MIDDLE and
                len(self.bottom) == self.MAX_BOTTOM)
    
    def total_cards(self) -> int:
        """Get total number of cards on the board."""
        return len(self.top) + len(self.middle) + len(self.bottom)
    
    def has_jokers(self) -> bool:
        """Check if board has any jokers."""
        return (has_joker(self.top) or 
                has_joker(self.middle) or 
                has_joker(self.bottom))
    
    def get_optimized_hands(self) -> Tuple[List[Card], List[Card], List[Card], bool]:
        """
        Get optimized hands with jokers resolved.
        
        Returns:
            (opt_top, opt_middle, opt_bottom, is_bust)
        """
        if not self.is_complete():
            return (self.top, self.middle, self.bottom, False)
        
        # Return cached result if available
        if self._optimized_cache is not None:
            return self._optimized_cache
        
        # No jokers - just check regular bust
        if not self.has_jokers():
            is_bust = self._check_bust_no_jokers()
            self._optimized_cache = (self.top, self.middle, self.bottom, is_bust)
            return self._optimized_cache
        
        # Has jokers - use optimizer
        from .joker_optimizer import optimize_board
        opt_top, opt_middle, opt_bottom, is_bust = optimize_board(
            self.top, self.middle, self.bottom
        )
        
        # If optimization failed, fall back to original cards
        if opt_top is None:
            opt_top = self.top
        if opt_middle is None:
            opt_middle = self.middle
        if opt_bottom is None:
            opt_bottom = self.bottom
        
        self._optimized_cache = (opt_top, opt_middle, opt_bottom, is_bust)
        return self._optimized_cache
    
    def _check_bust_no_jokers(self) -> bool:
        """Check bust for a board without jokers."""
        # Compare middle vs bottom (5-card hands)
        if compare_hands_5(self.bottom, self.middle) < 0:
            return True
        
        # Compare top vs middle using proper 3 vs 5 comparison
        from .joker_optimizer import compare_3_vs_5, hand_strength_3, hand_strength_5
        top_strength = hand_strength_3(self.top)
        mid_strength = hand_strength_5(self.middle)
        
        if compare_3_vs_5(top_strength, mid_strength) > 0:
            return True
        
        return False
    
    def is_bust(self) -> bool:
        """
        Check if the board is bust (foul).
        
        With jokers, the board is bust only if no joker substitution
        can avoid the bust.
        """
        if not self.is_complete():
            return False
        
        _, _, _, is_bust = self.get_optimized_hands()
        return is_bust
    
    def get_royalties(self) -> Dict[str, int]:
        """
        Get royalty points for each row.
        Uses optimized hands for joker boards.
        
        Returns:
            Dict with 'top', 'middle', 'bottom', 'total' keys
        """
        if not self.is_complete():
            return {'top': 0, 'middle': 0, 'bottom': 0, 'total': 0}
        
        opt_top, opt_middle, opt_bottom, is_bust = self.get_optimized_hands()
        
        if is_bust:
            return {'top': 0, 'middle': 0, 'bottom': 0, 'total': 0}
        
        top_r = get_top_royalty(opt_top)
        mid_r = get_middle_royalty(opt_middle)
        bot_r = get_bottom_royalty(opt_bottom)
        
        return {
            'top': top_r,
            'middle': mid_r,
            'bottom': bot_r,
            'total': top_r + mid_r + bot_r
        }
    
    def get_fantasyland_entry(self) -> int:
        """
        Check Fantasyland entry qualification.
        Uses optimized hands.
        
        Returns:
            Number of cards for Fantasyland (0 if not qualified)
        """
        if not self.is_complete():
            return 0
        
        opt_top, _, _, is_bust = self.get_optimized_hands()
        if is_bust:
            return 0
        return check_fantasyland_entry(opt_top)
    
    def get_fantasyland_stay(self) -> bool:
        """
        Check if can stay in Fantasyland.
        Uses optimized hands.
        
        Returns:
            True if qualified to stay
        """
        if not self.is_complete():
            return False
        
        opt_top, _, opt_bottom, is_bust = self.get_optimized_hands()
        if is_bust:
            return False
        return check_fantasyland_stay(opt_top, opt_bottom)
    
    def get_all_cards(self) -> List[Card]:
        """Get all cards on the board (original, including jokers)."""
        return self.top + self.middle + self.bottom
    
    def clear(self):
        """Remove all cards from the board."""
        self._clear_cache()
        self.top = []
        self.middle = []
        self.bottom = []
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            'top': [c.to_dict() for c in self.top],
            'middle': [c.to_dict() for c in self.middle],
            'bottom': [c.to_dict() for c in self.bottom],
            'is_complete': self.is_complete(),
            'has_jokers': self.has_jokers()
        }
        
        if self.is_complete():
            opt_top, opt_middle, opt_bottom, is_bust = self.get_optimized_hands()
            result['is_bust'] = is_bust
            result['optimized'] = {
                'top': [c.to_dict() for c in opt_top],
                'middle': [c.to_dict() for c in opt_middle],
                'bottom': [c.to_dict() for c in opt_bottom]
            }
            if not is_bust:
                result['royalties'] = self.get_royalties()
        
        return result
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Board':
        """Create from dictionary."""
        board = cls()
        board.top = [Card.from_dict(c) for c in data.get('top', [])]
        board.middle = [Card.from_dict(c) for c in data.get('middle', [])]
        board.bottom = [Card.from_dict(c) for c in data.get('bottom', [])]
        return board
    
    def __repr__(self) -> str:
        top_str = ' '.join(str(c) for c in self.top) or '---'
        mid_str = ' '.join(str(c) for c in self.middle) or '-----'
        bot_str = ' '.join(str(c) for c in self.bottom) or '-----'
        return f"Board(top=[{top_str}], middle=[{mid_str}], bottom=[{bot_str}])"
