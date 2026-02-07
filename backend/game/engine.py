"""
OFC Pineapple Game Engine
Handles deck, dealing, scoring, FL integration
"""

import random
from typing import Optional
from .models import Board, Session, HandState, PlayerState
import uuid
import sys
from pathlib import Path

# Add parent for game module access
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from game.hand_evaluator import evaluate_3_card_hand, evaluate_5_card_hand, HandRank, HandRank3
from game.royalty import calculate_royalty
from game.scoring import compare_hands


class Deck:
    """54-card deck with 2 jokers."""
    
    RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    SUITS = ['h', 'd', 'c', 's']  # hearts, diamonds, clubs, spades
    
    def __init__(self, include_jokers: bool = True):
        self.cards = []
        for suit in self.SUITS:
            for rank in self.RANKS:
                self.cards.append(f"{rank}{suit}")
        if include_jokers:
            self.cards.append("X1")  # Joker 1
            self.cards.append("X2")  # Joker 2
    
    def shuffle(self):
        random.shuffle(self.cards)
    
    def deal(self, count: int) -> list[str]:
        dealt = self.cards[:count]
        self.cards = self.cards[count:]
        return dealt


class GameEngine:
    """Manages a single game session."""
    
    def __init__(self, room_id: str, player_ids: list[str]):
        self.session = Session(
            session_id=str(uuid.uuid4()),
            room_id=room_id,
            status="active",
            chips=[200, 200],
            btn_seat=random.randint(0, 1)
        )
        self.player_ids = player_ids
    
    def start_hand(self) -> HandState:
        """Start a new hand."""
        deck = Deck(include_jokers=True)
        deck.shuffle()
        
        hand = HandState(
            hand_id=str(uuid.uuid4()),
            session_id=self.session.session_id,
            hand_number=self.session.hands_played + 1,
            status="dealing",
            turn=0,
            btn=self.session.btn_seat,
            current_player=self.session.btn_seat,  # BTN acts first
            deck=deck.cards,
            players=[
                PlayerState(player_id=self.player_ids[0], seat=0),
                PlayerState(player_id=self.player_ids[1], seat=1)
            ]
        )
        
        # Deal initial 5 cards to each player
        hand.dealt_cards = {
            0: hand.deck[:5],
            1: hand.deck[5:10]
        }
        hand.deck = hand.deck[10:]
        hand.status = "playing"
        
        self.session.current_hand = hand
        return hand
    
    def deal_turn(self, hand: HandState) -> dict:
        """Deal 3 cards to each player for regular turn."""
        if hand.turn >= 8:
            return {}
        
        hand.turn += 1
        
        # Deal 3 cards to each
        hand.dealt_cards = {
            0: hand.deck[:3],
            1: hand.deck[3:6]
        }
        hand.deck = hand.deck[6:]
        hand.current_player = hand.btn
        
        return hand.dealt_cards
    
    def apply_placement(self, hand: HandState, seat: int, 
                       placements: list[tuple[str, str]], 
                       discard: Optional[str] = None) -> bool:
        """Apply player's card placements."""
        player = hand.players[seat]
        
        for card, position in placements:
            if not player.board.place_card(card, position):
                return False
        
        if discard:
            hand.discards[seat].append(discard)
        
        return True
    
    def is_turn_complete(self, hand: HandState) -> bool:
        """Check if both players have placed cards for current turn."""
        if hand.turn == 0:
            # Initial turn: each player needs 5 cards placed
            p0_placed = sum(len(row) for row in [hand.players[0].board.top, 
                                                   hand.players[0].board.middle,
                                                   hand.players[0].board.bottom])
            p1_placed = sum(len(row) for row in [hand.players[1].board.top,
                                                   hand.players[1].board.middle,
                                                   hand.players[1].board.bottom])
            return p0_placed >= 5 and p1_placed >= 5
        else:
            # Regular turn tracking would need additional state
            return True  # Simplified for now
    
    def is_hand_complete(self, hand: HandState) -> bool:
        """Check if hand is complete (13 cards each)."""
        return (hand.players[0].board.is_complete() and 
                hand.players[1].board.is_complete())
    
    def evaluate_board(self, board: Board) -> dict:
        """Evaluate a complete board."""
        # Check bust (Top <= Middle <= Bottom)
        top_rank = evaluate_3_card_hand(board.top)
        mid_rank = evaluate_5_card_hand(board.middle)
        bot_rank = evaluate_5_card_hand(board.bottom)
        
        is_bust = not (top_rank <= mid_rank <= bot_rank)  # Simplified comparison
        
        # Calculate royalties
        royalties = {
            "top": calculate_royalty("top", board.top) if not is_bust else 0,
            "middle": calculate_royalty("middle", board.middle) if not is_bust else 0,
            "bottom": calculate_royalty("bottom", board.bottom) if not is_bust else 0,
        }
        royalties["total"] = sum(royalties.values())
        
        return {
            "is_bust": is_bust,
            "royalties": royalties,
            "hand_ranks": {
                "top": str(top_rank),
                "middle": str(mid_rank),
                "bottom": str(bot_rank)
            }
        }
    
    def score_hand(self, hand: HandState) -> dict:
        """Calculate final scores for the hand."""
        eval0 = self.evaluate_board(hand.players[0].board)
        eval1 = self.evaluate_board(hand.players[1].board)
        
        # Line results
        line_results = [0, 0, 0]  # P0 perspective
        
        if eval0["is_bust"] and not eval1["is_bust"]:
            line_results = [-1, -1, -1]
        elif eval1["is_bust"] and not eval0["is_bust"]:
            line_results = [1, 1, 1]
        elif not eval0["is_bust"] and not eval1["is_bust"]:
            # Compare each row
            # Simplified - actual comparison needs proper hand ranking
            pass
        
        # Scoop
        scoop = all(r == 1 for r in line_results) or all(r == -1 for r in line_results)
        scoop_bonus = 3 if scoop else 0
        
        # Raw score
        line_total = sum(line_results)
        royalty_diff = eval0["royalties"]["total"] - eval1["royalties"]["total"]
        raw_score_p0 = line_total + (scoop_bonus if line_total > 0 else -scoop_bonus) + royalty_diff
        
        # Chip cap
        actual_score = min(abs(raw_score_p0), self.session.chips[1 if raw_score_p0 > 0 else 0])
        if raw_score_p0 < 0:
            actual_score = -actual_score
        
        # Update chips
        self.session.chips[0] += actual_score
        self.session.chips[1] -= actual_score
        
        return {
            "boards": [
                {"top": hand.players[0].board.top, 
                 "middle": hand.players[0].board.middle,
                 "bottom": hand.players[0].board.bottom},
                {"top": hand.players[1].board.top,
                 "middle": hand.players[1].board.middle,
                 "bottom": hand.players[1].board.bottom}
            ],
            "busted": [eval0["is_bust"], eval1["is_bust"]],
            "royalties": [eval0["royalties"], eval1["royalties"]],
            "line_results": line_results,
            "scoop": scoop,
            "raw_score": [raw_score_p0, -raw_score_p0],
            "actual_score": [actual_score, -actual_score],
            "chips": self.session.chips.copy()
        }
    
    def check_fl_entry(self, board: Board) -> tuple[bool, int]:
        """Check if player qualifies for Fantasyland entry."""
        if len(board.top) < 3:
            return False, 0
        
        # Check top row for pairs QQ+, or trips
        top_cards = board.top
        ranks = [c[0] if c not in ["X1", "X2"] else "JK" for c in top_cards]
        
        # Count ranks
        rank_counts = {}
        jokers = 0
        for r in ranks:
            if r == "JK":
                jokers += 1
            else:
                rank_counts[r] = rank_counts.get(r, 0) + 1
        
        # Check for trips
        for r, count in rank_counts.items():
            if count + jokers >= 3:
                return True, 17
        
        # Check for pairs QQ+
        high_pairs = {'Q': 14, 'K': 15, 'A': 16}
        for r, count in rank_counts.items():
            if r in high_pairs and count + jokers >= 2:
                return True, high_pairs[r]
        
        return False, 0
    
    def check_session_end(self) -> Optional[dict]:
        """Check if session should end."""
        # Bankrupt
        if self.session.chips[0] <= 0:
            return {"winner": 1, "reason": "bankrupt"}
        if self.session.chips[1] <= 0:
            return {"winner": 0, "reason": "bankrupt"}
        
        # Chip lead (both not in FL)
        if abs(self.session.chips[0] - self.session.chips[1]) >= 40:
            winner = 0 if self.session.chips[0] > self.session.chips[1] else 1
            return {"winner": winner, "reason": "chip_lead"}
        
        return None
