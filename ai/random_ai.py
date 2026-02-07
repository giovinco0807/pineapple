"""Random AI for OFC Pineapple - for testing purposes."""
import random
from typing import List, Tuple
from game.card import Card
from game.board import Board, Row


class RandomAI:
    """
    Simple AI that makes random valid moves.
    Used for testing the game before implementing a strategic AI.
    """
    
    def __init__(self):
        self.name = "Random AI"
    
    def place_initial_cards(self, hand: List[Card], board: Board) -> List[Tuple[Card, Row]]:
        """
        Place all 5 initial cards randomly.
        
        Returns:
            List of (card, row) tuples
        """
        placements = []
        cards = hand.copy()
        random.shuffle(cards)
        
        # Distribute cards to rows
        # Try to put stronger cards at bottom
        for card in cards:
            available_rows = self._get_available_rows(board, placements)
            if available_rows:
                row = random.choice(available_rows)
                placements.append((card, row))
        
        return placements
    
    def place_pineapple_cards(
        self, 
        hand: List[Card], 
        board: Board
    ) -> Tuple[List[Tuple[Card, Row]], Card]:
        """
        Place 2 cards and discard 1 in pineapple phase.
        
        Returns:
            (placements, discard_card)
        """
        cards = hand.copy()
        random.shuffle(cards)
        
        # Pick one to discard
        discard = cards.pop()
        
        # Place the other two
        placements = []
        for card in cards:
            available_rows = self._get_available_rows(board, placements)
            if available_rows:
                row = random.choice(available_rows)
                placements.append((card, row))
        
        return (placements, discard)
    
    def place_fantasyland_cards(
        self, 
        hand: List[Card], 
        board: Board
    ) -> List[Tuple[Card, Row]]:
        """
        Place all fantasyland cards at once.
        
        Returns:
            List of (card, row) tuples
        """
        placements = []
        cards = hand.copy()
        random.shuffle(cards)
        
        top_count = 0
        middle_count = 0
        bottom_count = 0
        
        for card in cards:
            # Simple distribution: fill from bottom
            if bottom_count < 5:
                placements.append((card, Row.BOTTOM))
                bottom_count += 1
            elif middle_count < 5:
                placements.append((card, Row.MIDDLE))
                middle_count += 1
            elif top_count < 3:
                placements.append((card, Row.TOP))
                top_count += 1
        
        return placements
    
    def _get_available_rows(
        self, 
        board: Board, 
        pending_placements: List[Tuple[Card, Row]]
    ) -> List[Row]:
        """Get rows that have available slots."""
        # Count current + pending
        top_count = board.row_count(Row.TOP) + sum(1 for _, r in pending_placements if r == Row.TOP)
        mid_count = board.row_count(Row.MIDDLE) + sum(1 for _, r in pending_placements if r == Row.MIDDLE)
        bot_count = board.row_count(Row.BOTTOM) + sum(1 for _, r in pending_placements if r == Row.BOTTOM)
        
        available = []
        if top_count < 3:
            available.append(Row.TOP)
        if mid_count < 5:
            available.append(Row.MIDDLE)
        if bot_count < 5:
            available.append(Row.BOTTOM)
        
        return available
