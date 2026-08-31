"""OFC Pineapple - Backend Game State"""
import uuid
import random
from typing import Dict, Optional, List


RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['h', 'd', 'c', 's']


def create_deck(include_jokers: bool = True) -> List[str]:
    deck = [f"{r}{s}" for s in SUITS for r in RANKS]
    if include_jokers:
        deck.extend(["X1", "X2"])
    random.shuffle(deck)
    return deck


class GameState:
    def __init__(self, room_id: str, players: List[str]):
        self.session_id = str(uuid.uuid4())
        self.room_id = room_id
        self.players = players
        self.chips = [200, 200]
        self.btn = random.randint(0, 1)
        self.hands_played = 0

        # Current hand state
        self.hand_id: Optional[str] = None
        self.deck: List[str] = []
        self.turn = 0
        self.current_player = 0
        self.boards = [
            {"top": [], "middle": [], "bottom": []},
            {"top": [], "middle": [], "bottom": []}
        ]
        self.dealt_cards: Dict[int, List[str]] = {}
        self.discards = [[], []]
        self.placed_this_turn = [False, False]
        self.turn_start_times = [0.0, 0.0]

        # FL state
        self.is_fantasyland = [False, False]
        self.fl_card_count = [0, 0]

    def start_hand(self) -> dict:
        """Start a new hand. Handle Fantasyland mode."""
        self.hand_id = str(uuid.uuid4())
        self.deck = create_deck(include_jokers=True)
        self.turn = 0
        self.current_player = 1 - self.btn
        self.boards = [
            {"top": [], "middle": [], "bottom": []},
            {"top": [], "middle": [], "bottom": []}
        ]
        self.dealt_cards = {}
        self.discards = [[], []]
        self.placed_this_turn = [False, False]
        self.hands_played += 1

        # Check FL mode - deal more cards to FL player
        card_idx = 0
        for seat in [0, 1]:
            if self.is_fantasyland[seat] and self.fl_card_count[seat] > 0:
                fl_cards = self.fl_card_count[seat]
                self.dealt_cards[seat] = self.deck[card_idx:card_idx + fl_cards]
                card_idx += fl_cards
                print(f"[DEBUG] FL: Seat {seat} dealt {fl_cards} cards")
            else:
                self.dealt_cards[seat] = self.deck[card_idx:card_idx + 5]
                card_idx += 5
        self.deck = self.deck[card_idx:]

        return {
            "hand_id": self.hand_id,
            "hand_number": self.hands_played,
            "btn": self.btn,
            "chips": self.chips.copy(),
            "is_fantasyland": self.is_fantasyland.copy(),
            "fl_card_count": self.fl_card_count.copy()
        }

    def deal_turn(self) -> Dict[int, List[str]]:
        """Deal 3 cards to each player for regular turn. Skip FL players."""
        self.turn += 1
        self.placed_this_turn = [False, False]
        self.current_player = 1 - self.btn

        for seat in [0, 1]:
            if self.is_fantasyland[seat] and self.is_board_complete(seat):
                self.dealt_cards[seat] = []
                self.placed_this_turn[seat] = True
            else:
                self.dealt_cards[seat] = self.deck[:3]
                self.deck = self.deck[3:]

        return self.dealt_cards

    def is_board_complete(self, seat: int) -> bool:
        board = self.boards[seat]
        return len(board["top"]) == 3 and len(board["middle"]) == 5 and len(board["bottom"]) == 5

    def apply_placement(self, seat: int, placements: List[List[str]],
                       discard: Optional[str] = None) -> bool:
        """Apply player's card placement."""
        board = self.boards[seat]

        for card, position in placements:
            if position == "top" and len(board["top"]) < 3:
                board["top"].append(card)
            elif position == "middle" and len(board["middle"]) < 5:
                board["middle"].append(card)
            elif position == "bottom" and len(board["bottom"]) < 5:
                board["bottom"].append(card)
            else:
                return False

        if discard:
            self.discards[seat].append(discard)

        self.placed_this_turn[seat] = True
        return True

    def is_turn_complete(self) -> bool:
        return all(self.placed_this_turn)

    def is_hand_complete(self) -> bool:
        for board in self.boards:
            if len(board["top"]) != 3 or len(board["middle"]) != 5 or len(board["bottom"]) != 5:
                return False
        return True

    def next_btn(self):
        self.btn = 1 - self.btn
