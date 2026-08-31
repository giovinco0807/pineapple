"""
OFC Pineapple CFR - Lightweight Game State

Immutable-style game state for CFR traversal.
Supports both full OFC Pineapple (5 turns, 2 players) and
simplified toy variants for convergence testing.

Key design:
- State is copyable and lightweight for tree traversal
- Tracks both players' boards, decks, FL status
- Provides legal action enumeration delegated to action_space module
- Terminal utility uses exact scoring from game_engine
"""
import copy
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import IntEnum

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.encoding import Board, ALL_CARDS
from ai.engine.action_space import Action, get_initial_actions, get_turn_actions
from ai.engine.game_engine import (
    evaluate_hand, hand_category, check_fl_entry,
    get_top_royalty, get_middle_royalty, get_bottom_royalty,
    evaluate_board_with_joker_constraint,
)
from ai.engine.turn_order import action_order


class NodeType(IntEnum):
    """Type of game tree node."""
    PLAYER_0 = 0    # Player 0 decision
    PLAYER_1 = 1    # Player 1 decision
    CHANCE = 2       # Card deal
    TERMINAL = 3     # Game over


# FL chain EV values (from backward induction / FL solver stats)
FL_CHAIN_EV = {14: 14.0, 15: 27.9, 16: 52.4, 17: 104.5}


@dataclass
class OFCState:
    """
    Complete game state for CFR traversal.

    Designed to be cheaply copyable for tree expansion.
    All mutations return new states (functional style).
    """
    # Boards for both players
    boards: Tuple[Board, Board] = field(default_factory=lambda: (Board(), Board()))

    # Remaining deck (cards not yet dealt to anyone)
    deck: List[str] = field(default_factory=list)

    # Cards currently in each player's hand (to be placed)
    hands: Tuple[List[str], List[str]] = field(default_factory=lambda: ([], []))

    # Known discards per player
    discards: Tuple[List[str], List[str]] = field(default_factory=lambda: ([], []))

    # Current turn (0 = initial 5-card placement, 1-4 = pineapple turns)
    turn: int = 0

    # Which players have placed this turn
    placed: Tuple[bool, bool] = (False, False)

    # Button position (0 or 1)
    btn: int = 0

    # FL status
    is_fl: Tuple[bool, bool] = (False, False)
    fl_card_count: Tuple[int, int] = (0, 0)

    def copy(self) -> "OFCState":
        """Deep copy the state."""
        return OFCState(
            boards=(self.boards[0].copy(), self.boards[1].copy()),
            deck=list(self.deck),
            hands=(list(self.hands[0]), list(self.hands[1])),
            discards=(list(self.discards[0]), list(self.discards[1])),
            turn=self.turn,
            placed=self.placed,
            btn=self.btn,
            is_fl=self.is_fl,
            fl_card_count=self.fl_card_count,
        )

    @property
    def node_type(self) -> NodeType:
        """Determine the current node type."""
        # Check if game is over
        if self.boards[0].is_complete() and self.boards[1].is_complete():
            return NodeType.TERMINAL

        # Check if we need to deal cards
        if self._needs_deal():
            return NodeType.CHANCE

        # Determine which player acts
        return NodeType(self._acting_player())

    def _needs_deal(self) -> bool:
        """Check if cards need to be dealt."""
        # If both have placed this turn, advance
        if self.placed[0] and self.placed[1]:
            return True
        # If either player needs cards but doesn't have them
        for seat in range(2):
            if not self.placed[seat] and not self.boards[seat].is_complete():
                if not self.hands[seat]:
                    return True
        return False

    def _acting_player(self) -> int:
        """Determine which player acts next (non-button first)."""
        order = action_order(self.btn)
        for seat in order:
            if not self.placed[seat] and self.hands[seat] and not self.boards[seat].is_complete():
                return seat
        # Fallback (shouldn't reach here normally)
        return self.btn

    def get_legal_actions(self, player: int) -> List[Action]:
        """Get legal actions for the specified player."""
        if not self.hands[player]:
            return []

        board = self.boards[player]
        if board.is_complete():
            return []

        if self.turn == 0 and len(self.hands[player]) == 5:
            return get_initial_actions(self.hands[player], board)
        elif len(self.hands[player]) == 3:
            return get_turn_actions(self.hands[player], board)
        else:
            return []

    def apply_action(self, player: int, action: Action) -> "OFCState":
        """Apply action and return NEW state (immutable pattern)."""
        new_state = self.copy()

        # Apply placements
        for card, pos in action.placements:
            getattr(new_state.boards[player], pos).append(card)

        # Apply discard
        if action.discard:
            new_discards = list(new_state.discards[player])
            new_discards.append(action.discard)
            if player == 0:
                new_state.discards = (new_discards, list(new_state.discards[1]))
            else:
                new_state.discards = (list(new_state.discards[0]), new_discards)

        # Clear hand
        if player == 0:
            new_state.hands = ([], list(new_state.hands[1]))
        else:
            new_state.hands = (list(new_state.hands[0]), [])

        # Mark as placed
        if player == 0:
            new_state.placed = (True, new_state.placed[1])
        else:
            new_state.placed = (new_state.placed[0], True)

        return new_state

    def deal_cards(self) -> "OFCState":
        """Deal cards for the next phase. Returns new state.

        Turn 0: deal 5 cards to each player
        Turn 1-4: deal 3 cards to each player
        """
        new_state = self.copy()

        if new_state.turn == 0 and not new_state.hands[0] and not new_state.hands[1]:
            # Initial deal: 5 cards each
            new_state.hands = (
                new_state.deck[:5],
                new_state.deck[5:10],
            )
            new_state.deck = new_state.deck[10:]
        else:
            # Advance to next turn
            new_state.turn += 1
            new_state.placed = (False, False)

            # Deal 3 cards each (only to non-complete boards)
            idx = 0
            hands = [[], []]
            for seat in range(2):
                if not new_state.boards[seat].is_complete():
                    hands[seat] = new_state.deck[idx:idx + 3]
                    idx += 3
            new_state.hands = (hands[0], hands[1])
            new_state.deck = new_state.deck[idx:]

        return new_state

    def terminal_utility(self, player: int) -> float:
        """
        Compute utility for the specified player at a terminal state.

        Includes:
        - Line comparison (+1/-1 per line won/lost)
        - Scoop bonus (+3)
        - Royalties
        - FL entry EV bonus
        """
        b0, b1 = self.boards
        busted = [False, False]
        royalties = [0, 0]
        hand_vals = [{}, {}]
        board_evals = []

        for seat, board in enumerate([b0, b1]):
            evaluated = evaluate_board_with_joker_constraint(
                board.top, board.middle, board.bottom
            )
            board_evals.append(evaluated)
            hand_vals[seat] = dict(evaluated["values"])
            busted[seat] = bool(evaluated["busted"])
            royalties[seat] = int(evaluated["royalties"]["total"])

        # Score calculation (from P0 perspective)
        if busted[0] and busted[1]:
            p0_score = 0.0
        elif busted[0]:
            p0_score = float(-6 - royalties[1])
        elif busted[1]:
            p0_score = float(6 + royalties[0])
        else:
            line_results = [0, 0, 0]
            for i, line in enumerate(["top", "middle", "bottom"]):
                if hand_vals[0][line] > hand_vals[1][line]:
                    line_results[i] = 1
                elif hand_vals[0][line] < hand_vals[1][line]:
                    line_results[i] = -1

            line_total = sum(line_results)
            scoop = abs(line_total) == 3
            scoop_bonus = 3 if scoop else 0

            p0_score = float(
                line_total +
                (scoop_bonus if line_total > 0 else -scoop_bonus if line_total < 0 else 0) +
                royalties[0] - royalties[1]
            )

        # FL entry EV bonus (normal round only -- not when already in FL)
        for seat in range(2):
            if not busted[seat] and not self.is_fl[seat]:
                fl = bool(board_evals[seat]["fl_entry"])
                fl_cards = int(board_evals[seat]["fl_card_count"])
                if fl:
                    ev = FL_CHAIN_EV.get(fl_cards, 0)
                    if seat == 0:
                        p0_score += ev
                    else:
                        p0_score -= ev

        return p0_score if player == 0 else -p0_score

    # ─── Info Set Key Generation ──────────────────────────────────────

    def info_set_key(self, player: int) -> str:
        """
        Generate information set key for the acting player.

        An information set groups all states that look identical
        from the player's perspective:
        - Own board (visible)
        - Opponent's board (visible in OFC)
        - Own hand cards (private)
        - Own discards (private)
        - Turn number
        - Button position

        NOTE: In OFC, opponent's board IS visible, but opponent's
        hand and discards are NOT visible. This is a key difference
        from NLH where hole cards are completely hidden.
        """
        my_board = self.boards[player]
        opp_board = self.boards[1 - player]

        parts = [
            f"t{self.turn}",
            f"b{1 if player == self.btn else 0}",
            # My board state
            f"mt{_sort_key(my_board.top)}",
            f"mm{_sort_key(my_board.middle)}",
            f"mb{_sort_key(my_board.bottom)}",
            # Opponent board (visible in OFC!)
            f"ot{_sort_key(opp_board.top)}",
            f"om{_sort_key(opp_board.middle)}",
            f"ob{_sort_key(opp_board.bottom)}",
            # My hand (private)
            f"h{_sort_key(self.hands[player])}",
            # My discards (private)
            f"d{_sort_key(self.discards[player])}",
        ]
        return "|".join(parts)


def _sort_key(cards: List[str]) -> str:
    """Create a canonical string key from a card list."""
    return ",".join(sorted(cards)) if cards else "_"


# ─── Game Factory ──────────────────────────────────────────────────

def create_initial_state(
    deck: Optional[List[str]] = None,
    btn: int = 0,
) -> OFCState:
    """Create a new game state with shuffled deck."""
    if deck is None:
        deck = list(ALL_CARDS)
        random.shuffle(deck)

    state = OFCState(
        deck=deck,
        btn=btn,
    )
    # Deal initial hands
    state = state.deal_cards()
    return state


# ─── Toy Game (Simplified OFC for convergence testing) ───────────

def create_toy_state(n_cards: int = 6, btn: int = 0) -> OFCState:
    """
    Create a simplified OFC game for convergence testing.

    Toy game: Each player gets 3 cards, places into 1-row board.
    No pineapple turns, just initial placement.
    Used to verify CFR convergence on a tractable game.

    Args:
        n_cards: Cards per player (3 for minimal game)
        btn: Button position
    """
    # Use a minimal deck (just high cards for simplicity)
    mini_deck = [f"{r}{s}" for r in "AKQJT" for s in "hdcs"]
    random.shuffle(mini_deck)

    state = OFCState(
        deck=mini_deck[n_cards * 2:],
        hands=(mini_deck[:n_cards], mini_deck[n_cards:n_cards * 2]),
        btn=btn,
    )
    return state
