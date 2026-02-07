"""
OFC Pineapple - Headless Game Engine for AI Training

No WebSocket dependency. Used for self-play and evaluation.
Includes full scoring with royalties, bust detection, FL entry.
"""
import copy
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .encoding import Board, Observation, ALL_CARDS
from .action_space import Action

# Rank values for hand evaluation
RANK_VALUES = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
               '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}


@dataclass
class HandResult:
    """Result of a completed hand."""
    boards: List[Board]
    busted: List[bool]
    royalties: List[Dict[str, int]]
    hand_names: List[Dict[str, str]]
    line_results: List[int]   # +1/-1/0 from P0 perspective
    scoop: bool
    raw_score: List[int]      # [p0_score, p1_score]
    fl_entry: List[bool]
    fl_card_count: List[int]


class Hand:
    """A single hand of OFC Pineapple."""

    def __init__(self, deck: List[str], btn: int = 0):
        self.deck = list(deck)
        self.btn = btn
        self.boards = [Board(), Board()]
        self.dealt_cards: List[List[str]] = [[], []]
        self.discards: List[List[str]] = [[], []]
        self.turn = 0
        self.placed = [False, False]

        # Deal initial 5 cards each
        self.dealt_cards[0] = self.deck[:5]
        self.dealt_cards[1] = self.deck[5:10]
        self.deck = self.deck[10:]

    def get_observation(self, seat: int) -> Observation:
        """Build observation for a specific player."""
        return Observation(
            board_self=self.boards[seat].copy(),
            board_opponent=self.boards[1 - seat].copy(),
            dealt_cards=list(self.dealt_cards[seat]),
            known_discards_self=list(self.discards[seat]),
            turn=self.turn,
            is_btn=(seat == self.btn),
        )

    def apply_action(self, seat: int, action: Action):
        """Apply an action for a player."""
        for card, pos in action.placements:
            getattr(self.boards[seat], pos).append(card)
        if action.discard:
            self.discards[seat].append(action.discard)
        self.placed[seat] = True

    def is_turn_complete(self) -> bool:
        return self.placed[0] and self.placed[1]

    def deal_next_turn(self):
        """Deal 3 cards to each player for the next turn."""
        self.turn += 1
        self.placed = [False, False]
        self.dealt_cards[0] = self.deck[:3]
        self.dealt_cards[1] = self.deck[3:6]
        self.deck = self.deck[6:]

    def is_hand_complete(self) -> bool:
        return self.boards[0].is_complete() and self.boards[1].is_complete()


class Session:
    """A session of multiple hands."""

    def __init__(self, chips: Optional[List[int]] = None, max_hands: int = 20):
        self.chips = chips or [200, 200]
        self.btn = 0
        self.hand_count = 0
        self.max_hands = max_hands

    def new_hand(self) -> Hand:
        deck = list(ALL_CARDS)
        random.shuffle(deck)
        hand = Hand(deck=deck, btn=self.btn)
        self.hand_count += 1
        return hand

    def apply_result(self, result: HandResult):
        self.chips[0] += result.raw_score[0]
        self.chips[1] += result.raw_score[1]
        self.btn = 1 - self.btn

    def is_finished(self) -> bool:
        return (self.hand_count >= self.max_hands
                or self.chips[0] <= 0
                or self.chips[1] <= 0
                or self.chips[0] >= 400
                or self.chips[1] >= 400)


class GameEngine:
    """Headless game engine — no WebSocket, pure logic."""

    @staticmethod
    def compute_result(hand: Hand) -> HandResult:
        """Score a completed hand."""
        busted = [False, False]
        royalties = [
            {"top": 0, "middle": 0, "bottom": 0, "total": 0},
            {"top": 0, "middle": 0, "bottom": 0, "total": 0},
        ]
        hand_names = [{}, {}]
        fl_entry = [False, False]
        fl_card_count = [0, 0]

        hand_values = [{}, {}]
        for seat in [0, 1]:
            board = hand.boards[seat]
            top_val = evaluate_hand(board.top, 3)
            mid_val = evaluate_hand(board.middle, 5)
            bot_val = evaluate_hand(board.bottom, 5)
            hand_values[seat] = {"top": top_val, "middle": mid_val, "bottom": bot_val}

            # Bust check
            if top_val > mid_val or mid_val > bot_val:
                busted[seat] = True
            else:
                royalties[seat]["top"] = get_top_royalty(board.top)
                royalties[seat]["middle"] = get_middle_royalty(board.middle)
                royalties[seat]["bottom"] = get_bottom_royalty(board.bottom)
                royalties[seat]["total"] = (
                    royalties[seat]["top"] + royalties[seat]["middle"] + royalties[seat]["bottom"]
                )
                fl, cards = check_fl_entry(board.top)
                fl_entry[seat] = fl
                fl_card_count[seat] = cards

            hand_names[seat] = {
                "top": get_hand_name(board.top, 3),
                "middle": get_hand_name(board.middle, 5),
                "bottom": get_hand_name(board.bottom, 5),
            }

        # Line results (P0 perspective)
        line_results = [0, 0, 0]
        if busted[0] and busted[1]:
            pass
        elif busted[0]:
            line_results = [-1, -1, -1]
        elif busted[1]:
            line_results = [1, 1, 1]
        else:
            for i, line in enumerate(["top", "middle", "bottom"]):
                if hand_values[0][line] > hand_values[1][line]:
                    line_results[i] = 1
                elif hand_values[0][line] < hand_values[1][line]:
                    line_results[i] = -1

        # Score
        line_total = sum(line_results)
        scoop = abs(line_total) == 3
        scoop_bonus = 3 if scoop else 0

        if busted[0] and not busted[1]:
            raw_score = [-6 - royalties[1]["total"], 6 + royalties[1]["total"]]
        elif busted[1] and not busted[0]:
            raw_score = [6 + royalties[0]["total"], -6 - royalties[0]["total"]]
        elif busted[0] and busted[1]:
            raw_score = [0, 0]
        else:
            p0 = line_total + (scoop_bonus if line_total > 0 else -scoop_bonus if line_total < 0 else 0)
            p0 += royalties[0]["total"] - royalties[1]["total"]
            raw_score = [p0, -p0]

        return HandResult(
            boards=[b.copy() for b in hand.boards],
            busted=busted,
            royalties=royalties,
            hand_names=hand_names,
            line_results=line_results,
            scoop=scoop,
            raw_score=raw_score,
            fl_entry=fl_entry,
            fl_card_count=fl_card_count,
        )


# ============================================================
# Hand evaluation
# ============================================================

def evaluate_hand(cards: List[str], expected_count: int) -> int:
    """Evaluate hand strength. Higher = stronger."""
    if len(cards) != expected_count:
        return 0

    ranks = []
    suits = []
    jokers = 0
    for c in cards:
        if c in ("X1", "X2"):
            jokers += 1
        else:
            ranks.append(RANK_VALUES.get(c[0], 0))
            suits.append(c[1])

    rank_counts = Counter(ranks)
    best_count = max(rank_counts.values()) if rank_counts else 0

    if expected_count == 3:
        # 3-card hand
        if best_count + jokers >= 3:
            r = max(r for r, c in rank_counts.items() if c + jokers >= 3) if rank_counts else 14
            return 4000 + r
        if best_count + jokers >= 2:
            pair_ranks = [r for r, c in rank_counts.items() if c + jokers >= 2]
            return 2000 + max(pair_ranks)
        return max(ranks) if ranks else 14

    # 5-card hand
    suit_counts = Counter(suits)
    is_flush = (max(suit_counts.values()) + jokers >= 5) if suit_counts else jokers >= 5

    sorted_ranks = sorted(ranks, reverse=True)
    is_straight = check_straight(sorted_ranks, jokers)

    # Straight flush / Royal flush
    if is_flush and is_straight:
        return 9000 + (max(ranks) if ranks else 14)
    # Four of a kind
    if best_count + jokers >= 4:
        return 8000 + max(r for r, c in rank_counts.items() if c >= 2)
    # Full house: need trips + pair (exactly 2 distinct ranks, or joker helps)
    if best_count >= 3 and len(rank_counts) == 2:
        # Natural trips + natural pair (e.g. AAA KK)
        return 7000 + (max(ranks) if ranks else 14)
    if best_count >= 3 and len(rank_counts) == 1 and jokers >= 2:
        # Trips + jokers making a pair (e.g. AAA + 2 jokers)
        return 7000 + (max(ranks) if ranks else 14)
    if best_count >= 2 and jokers >= 1 and len(rank_counts) == 2:
        # Pair + joker -> trips, with another pair (e.g. AA KK + joker)
        return 7000 + (max(ranks) if ranks else 14)
    if is_flush:
        return 6000 + max(ranks)
    if is_straight:
        return 5000 + max(ranks)
    if best_count + jokers >= 3:
        return 4000 + max(r for r, c in rank_counts.items() if c >= 2)
    pairs = [r for r, c in rank_counts.items() if c >= 2]
    if len(pairs) >= 2:
        return 3000 + max(pairs)
    if best_count >= 2 or jokers >= 1:
        return 2000 + (max(pairs) if pairs else max(ranks))
    return sorted_ranks[0] if sorted_ranks else 0


def check_straight(sorted_ranks: List[int], jokers: int = 0) -> bool:
    """Check if ranks can form a straight (with jokers)."""
    if len(sorted_ranks) + jokers < 5:
        return False
    unique = sorted(set(sorted_ranks), reverse=True)
    # Regular straights
    for high in range(14, 4, -1):
        needed = set(range(high, high - 5, -1))
        missing = needed - set(unique)
        if len(missing) <= jokers:
            return True
    # Wheel (A-2-3-4-5)
    wheel = {14, 2, 3, 4, 5}
    if len(wheel - set(unique)) <= jokers:
        return True
    return False


# ============================================================
# Royalties
# ============================================================

def get_top_royalty(cards: List[str]) -> int:
    """Top row: 66=1, 77=2, ..., AA=9. Trips: 222=10, ..., AAA=22."""
    ranks = []
    jokers = 0
    for c in cards:
        if c in ("X1", "X2"):
            jokers += 1
        else:
            ranks.append(RANK_VALUES.get(c[0], 0))
    rank_counts = Counter(ranks)

    best = 0
    for r in sorted(rank_counts.keys(), reverse=True):
        count = rank_counts[r]
        if count + jokers >= 3:
            return 10 + (r - 2)
        if count + jokers >= 2 and r >= 6:
            best = max(best, r - 5)
    return best


def get_middle_royalty(cards: List[str]) -> int:
    """Middle row royalties."""
    val = evaluate_hand(cards, 5)
    if val >= 9014: return 50   # Royal flush (A-high SF)
    if val >= 9000: return 30   # Straight flush
    if val >= 8000: return 20   # Four of a kind
    if val >= 7000: return 12   # Full house
    if val >= 6000: return 8    # Flush
    if val >= 5000: return 4    # Straight
    if val >= 4000: return 2    # Three of a kind
    return 0


def get_bottom_royalty(cards: List[str]) -> int:
    """Bottom row royalties."""
    val = evaluate_hand(cards, 5)
    if val >= 9014: return 25   # Royal flush (A-high SF)
    if val >= 9000: return 15   # Straight flush
    if val >= 8000: return 10   # Four of a kind
    if val >= 7000: return 6    # Full house
    if val >= 6000: return 4    # Flush
    if val >= 5000: return 2    # Straight
    return 0


def check_fl_entry(top_cards: List[str]) -> Tuple[bool, int]:
    """Check if top row qualifies for Fantasyland entry."""
    ranks = []
    jokers = 0
    for c in top_cards:
        if c in ("X1", "X2"):
            jokers += 1
        else:
            ranks.append(RANK_VALUES.get(c[0], 0))
    rank_counts = Counter(ranks)

    for r in sorted(rank_counts.keys(), reverse=True):
        count = rank_counts[r]
        if count + jokers >= 3:
            return True, 17  # Trips
        if count + jokers >= 2:
            if r == 14:    return True, 16  # AA
            elif r == 13:  return True, 15  # KK
            elif r == 12:  return True, 14  # QQ
    return False, 0


def get_hand_name(cards: List[str], expected_count: int) -> str:
    """Human-readable hand name (Japanese)."""
    val = evaluate_hand(cards, expected_count)
    if val == 0:
        return "---"
    if expected_count == 3:
        if val >= 4000:
            return "スリーカード"
        elif val >= 2000:
            rank = val - 2000
            names = {2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',
                     10:'T',11:'J',12:'Q',13:'K',14:'A'}
            return f"{names.get(rank,'?')}のペア"
        return "ハイカード"
    else:
        if val >= 9014: return "ロイヤルフラッシュ"
        if val >= 9000: return "ストレートフラッシュ"
        if val >= 8000: return "フォーカード"
        if val >= 7000: return "フルハウス"
        if val >= 6000: return "フラッシュ"
        if val >= 5000: return "ストレート"
        if val >= 4000: return "スリーカード"
        if val >= 3000: return "ツーペア"
        if val >= 2000: return "ワンペア"
        return "ハイカード"
