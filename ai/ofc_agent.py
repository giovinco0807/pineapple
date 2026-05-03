import os
from typing import List, Tuple

from game.card import Card
from game.board import Board, Row
from ai.random_ai import RandomAI

class OFCAgent:
    """
    Agent that delegates to trained ML models when possible, 
    falling back to heuristics or random play otherwise.
    """
    
    def __init__(self, t1_model_path="ai/models/t1_placement_net_v1.pt"):
        self.random_ai = RandomAI()
        
        try:
            # We import here to avoid torch dependency if not needed
            from ai.t1_agent import T1Agent
            if os.path.exists(t1_model_path):
                self.t1_agent = T1Agent(t1_model_path)
                print(f"Loaded T1Agent from {t1_model_path}")
            else:
                print(f"T1Agent model not found at {t1_model_path}, using random.")
                self.t1_agent = None
        except Exception as e:
            print(f"Failed to load T1Agent, falling back to Random AI: {e}")
            self.t1_agent = None
            
    def place_initial_cards(self, hand: List[Card], board: Board) -> List[Tuple[Card, Row]]:
        # T0 logic (TODO: add T0 agent integration here)
        return self.random_ai.place_initial_cards(hand, board)
        
    def place_pineapple_cards(
        self, 
        hand: List[Card], 
        board: Board
    ) -> Tuple[List[Tuple[Card, Row]], Card]:
        
        # Determine if it's T1 (board has exactly 5 cards)
        board_count = board.row_count(Row.TOP) + board.row_count(Row.MIDDLE) + board.row_count(Row.BOTTOM)
        
        if board_count == 5 and self.t1_agent is not None:
            # Use T1 Agent
            top_strs = [self._card_to_str(c) for c in board.top]
            mid_strs = [self._card_to_str(c) for c in board.middle]
            bot_strs = [self._card_to_str(c) for c in board.bottom]
            hand_strs = [self._card_to_str(c) for c in hand]
            
            placements_str, _ = self.t1_agent.get_action(top_strs, mid_strs, bot_strs, hand_strs)
            
            placements = []
            discard = None
            for c_str, r_str in placements_str:
                matched_card = next((c for c in hand if self._card_to_str(c) == c_str), None)
                if r_str == "Discard":
                    discard = matched_card
                else:
                    placements.append((matched_card, self._str_to_row(r_str)))
                    
            if discard is not None and len(placements) == 2:
                return placements, discard
            else:
                print("T1Agent returned invalid action. Falling back to random.")
        
        # T2, T3, T4 logic
        return self.random_ai.place_pineapple_cards(hand, board)
        
    def place_fantasyland_cards(self, hand: List[Card], board: Board) -> List[Tuple[Card, Row]]:
        # FL logic (TODO: use FL solver here)
        return self.random_ai.place_fantasyland_cards(hand, board)
        
    def _card_to_str(self, c: Card) -> str:
        if c.is_joker:
            return "X1"  # Or X2, depending on how jokers are mapped
        return str(c.rank) + c.suit
        
    def _str_to_row(self, r: str) -> Row:
        if r == "Top": return Row.TOP
        if r == "Mid": return Row.MIDDLE
        return Row.BOTTOM
