"""Card and Deck classes for OFC Pineapple."""
import random
from dataclasses import dataclass, field
from typing import List, Optional, Set

# Ranks: 2-10, J, Q, K, A
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
RANK_VALUES = {r: i for i, r in enumerate(RANKS)}

# Suits
SUITS = ['s', 'h', 'd', 'c']  # spades, hearts, diamonds, clubs
SUIT_SYMBOLS = {'s': '♠', 'h': '♥', 'd': '♦', 'c': '♣'}

# Joker constants
JOKER_RANK = 'X'
JOKER_SUIT = 'j'


@dataclass
class Card:
    """Represents a playing card (including jokers)."""
    rank: str
    suit: str
    
    def __post_init__(self):
        # Allow joker
        if self.is_joker:
            return
        if self.rank not in RANKS:
            raise ValueError(f"Invalid rank: {self.rank}")
        if self.suit not in SUITS:
            raise ValueError(f"Invalid suit: {self.suit}")
    
    @property
    def is_joker(self) -> bool:
        """Check if this card is a joker."""
        return self.rank == JOKER_RANK and self.suit == JOKER_SUIT
    
    @property
    def rank_value(self) -> int:
        """Get numeric value of rank (0-12). Returns -1 for joker."""
        if self.is_joker:
            return -1
        return RANK_VALUES[self.rank]
    
    @property
    def suit_symbol(self) -> str:
        """Get suit symbol for display."""
        if self.is_joker:
            return '🃏'
        return SUIT_SYMBOLS[self.suit]
    
    def __str__(self) -> str:
        if self.is_joker:
            return '🃏'
        return f"{self.rank}{self.suit_symbol}"
    
    def __repr__(self) -> str:
        if self.is_joker:
            return "Card.joker()"
        return f"Card('{self.rank}', '{self.suit}')"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Card):
            return False
        return self.rank == other.rank and self.suit == other.suit
    
    def __hash__(self) -> int:
        return hash((self.rank, self.suit))
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'rank': self.rank,
            'suit': self.suit,
            'display': str(self),
            'is_joker': self.is_joker
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Card':
        """Create from dictionary."""
        if data.get('is_joker'):
            return cls.joker()
        return cls(data['rank'], data['suit'])
    
    @classmethod
    def from_string(cls, s: str) -> 'Card':
        """Create from string like 'As' or 'Th'. 'Xj' for joker."""
        if s in ('Xj', 'joker', '🃏'):
            return cls.joker()
        if len(s) != 2:
            raise ValueError(f"Invalid card string: {s}")
        return cls(s[0], s[1])
    
    @classmethod
    def joker(cls) -> 'Card':
        """Create a joker card."""
        return cls(JOKER_RANK, JOKER_SUIT)


def get_all_standard_cards() -> List[Card]:
    """Get all 52 standard cards (no jokers)."""
    return [Card(rank, suit) for suit in SUITS for rank in RANKS]


class Deck:
    """Represents a deck of 54 cards (52 + 2 jokers)."""
    
    def __init__(self, include_jokers: bool = True):
        self.include_jokers = include_jokers
        self.cards: List[Card] = []
        self.reset()
    
    def reset(self):
        """Reset deck to full 54 cards (or 52 without jokers)."""
        self.cards = get_all_standard_cards()
        if self.include_jokers:
            self.cards.append(Card.joker())
            self.cards.append(Card.joker())
    
    def shuffle(self):
        """Shuffle the deck."""
        random.shuffle(self.cards)
    
    def deal(self, count: int = 1) -> List[Card]:
        """Deal cards from the deck."""
        if count > len(self.cards):
            raise ValueError(f"Cannot deal {count} cards, only {len(self.cards)} remaining")
        dealt = self.cards[:count]
        self.cards = self.cards[count:]
        return dealt
    
    def deal_one(self) -> Optional[Card]:
        """Deal a single card."""
        if not self.cards:
            return None
        return self.cards.pop(0)
    
    def remove(self, cards: List[Card]):
        """Remove specific cards from the deck."""
        for card in cards:
            if card in self.cards:
                self.cards.remove(card)
    
    def __len__(self) -> int:
        return len(self.cards)
    
    def __repr__(self) -> str:
        return f"Deck({len(self.cards)} cards)"


def has_joker(cards: List[Card]) -> bool:
    """Check if any card in the list is a joker."""
    return any(c.is_joker for c in cards)


def count_jokers(cards: List[Card]) -> int:
    """Count the number of jokers in the list."""
    return sum(1 for c in cards if c.is_joker)


def get_non_joker_cards(cards: List[Card]) -> List[Card]:
    """Get all non-joker cards from the list."""
    return [c for c in cards if not c.is_joker]
