"""
OFC Pineapple Game Data Models (Pydantic)
"""

from pydantic import BaseModel
from typing import Optional
from enum import Enum


class Position(str, Enum):
    TOP = "top"
    MIDDLE = "middle"
    BOTTOM = "bottom"


class Board(BaseModel):
    top: list[str] = []       # max 3 cards
    middle: list[str] = []    # max 5 cards  
    bottom: list[str] = []    # max 5 cards
    
    def is_complete(self) -> bool:
        return len(self.top) == 3 and len(self.middle) == 5 and len(self.bottom) == 5
    
    def can_place(self, position: str) -> bool:
        if position == "top":
            return len(self.top) < 3
        elif position == "middle":
            return len(self.middle) < 5
        elif position == "bottom":
            return len(self.bottom) < 5
        return False
    
    def place_card(self, card: str, position: str) -> bool:
        if not self.can_place(position):
            return False
        if position == "top":
            self.top.append(card)
        elif position == "middle":
            self.middle.append(card)
        elif position == "bottom":
            self.bottom.append(card)
        return True


class PlayerState(BaseModel):
    player_id: str
    seat: int                      # 0 or 1
    board: Board = Board()
    is_fantasyland: bool = False
    fl_card_count: int = 0         # 14-17
    connected: bool = True
    

class Placement(BaseModel):
    card: str
    position: str  # "top", "middle", "bottom"


class Action(BaseModel):
    placements: list[tuple[str, str]]  # [(card, position), ...]
    discard: Optional[str] = None      # discard card (None for initial turn)


class HandState(BaseModel):
    hand_id: str
    session_id: str
    hand_number: int
    status: str = "dealing"  # "dealing" | "playing" | "fantasyland" | "scoring" | "finished"
    turn: int = 0            # 0-8 (0=initial 5 cards, 1-8=3 cards)
    btn: int = 0             # 0 or 1 (BTN player)
    current_player: int = 0  # whose turn
    
    players: list[PlayerState] = []
    deck: list[str] = []     # remaining deck (server only)
    dealt_cards: dict = {}   # {seat: [cards]}
    discards: list[list[str]] = [[], []]  # each player's discards


class Session(BaseModel):
    session_id: str
    room_id: str
    status: str = "waiting"  # "waiting" | "active" | "finished"
    
    chips: list[int] = [200, 200]
    hands_played: int = 0
    btn_seat: int = 0        # current BTN (0 or 1)
    
    current_hand: Optional[HandState] = None
    
    winner: Optional[int] = None
    finish_reason: Optional[str] = None  # "chip_lead" | "bankrupt"


class TurnLog(BaseModel):
    hand_id: str
    turn: int
    player: int
    
    board_self: Board
    board_opponent: Board
    dealt_cards: list[str]
    known_discards: list[str]
    
    action_placements: list[tuple[str, str]]
    action_discard: Optional[str]
    
    think_time_ms: int
    timestamp: str
