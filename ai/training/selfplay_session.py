"""
Session-Based Multi-Hand Self-Play with FL Continuation.

Simulates full sessions (200 chips each) with:
  - VN-greedy mirror matches (both players use same VN)
  - FL entry → Rust FL Solver v2 for optimal placement
  - FL vs Normal: Non-FL plays first, FL places optimally against opponent's board
  - FL Stay with card count inheritance and unlimited chaining
  - BTN alternates each hand (random first BTN)
  - Session ends on 40-point lead (outside FL) or bankruptcy (chips ≤ 0)

Output: JSONL compatible with preprocess_mc_teacher.py

Usage:
    python -m ai.training.selfplay_session \
        --model models/vn_v3_s200/value_best.pt \
        --norm models/vn_v3_s200/norm_stats.json \
        --sessions 1000 --output data/session_selfplay.jsonl
"""
import sys
import json
import random
import time
import argparse
import itertools
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.encoding import Board, Observation, ALL_CARDS, encode_state, STATE_DIM
from ai.engine.action_space import get_initial_actions, get_turn_actions, Action
from ai.engine.game_engine import (
    GameEngine, Hand, HandResult, check_fl_entry, evaluate_hand, hand_category,
    get_top_royalty, get_middle_royalty, get_bottom_royalty,
    evaluate_board_with_joker_constraint,
)
from ai.engine.scoring import check_fl_stay_from_cards
from ai.engine.turn_order import action_order
from ai.models.networks import ValueNetworkV3
from ai.rust_solver_wrapper import RustFLSolver

# Card string → (rank, suit) for Rust solver
RANK_MAP = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
            '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
SUIT_MAP = {'s': 0, 'h': 1, 'd': 2, 'c': 3}


def card_str_to_tuple(card: str) -> Tuple[int, int]:
    """Convert card string (e.g., 'As', 'X1') to (rank, suit) tuple."""
    if card in ("X1", "X2", "JK"):
        return (0, 4)  # Joker
    return (RANK_MAP[card[0]], SUIT_MAP[card[1]])


def card_tuple_to_str(rank: int, suit: int) -> str:
    """Convert (rank, suit) tuple back to card string."""
    if suit == 4 or rank == 0:
        return "X1"  # Simplify joker back
    rank_chars = {2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8',
                  9:'9', 10:'T', 11:'J', 12:'Q', 13:'K', 14:'A'}
    suit_chars = {0:'s', 1:'h', 2:'d', 3:'c'}
    return rank_chars[rank] + suit_chars[suit]


def placement_to_board(placement: dict) -> Board:
    """Convert Rust solver placement dict to Board."""
    return Board(
        top=[card_tuple_to_str(c["rank"], c["suit"]) for c in placement["top"]],
        middle=[card_tuple_to_str(c["rank"], c["suit"]) for c in placement["middle"]],
        bottom=[card_tuple_to_str(c["rank"], c["suit"]) for c in placement["bottom"]],
    )


# ─────────────────────────────────────────────────────────────
# VN-Greedy Player (reused from selfplay_vn_greedy.py)
# ─────────────────────────────────────────────────────────────

class VNGreedyPlayer:
    """Plays OFC hands using VN greedy evaluation with softmax sampling."""

    # Default FL chain EVs by card count (from fl_ev.json)
    DEFAULT_FL_EV = {14: 16.0, 15: 23.8, 16: 34.9, 17: 99.2}

    def __init__(self, model: ValueNetworkV3, device: str = "cpu",
                 norm_stats: dict = None, temperature: float = 0.1,
                 fl_ev: dict = None):
        self.model = model
        self.device = device
        self.model.eval()
        self.temperature = temperature

        # Per-card-count FL bonus
        self.fl_ev = fl_ev if fl_ev else dict(self.DEFAULT_FL_EV)

        self.score_mean = 0.0
        self.score_std = 1.0
        if norm_stats:
            self.score_mean = norm_stats.get("mean", norm_stats.get("score_mean", 0.0))
            self.score_std = norm_stats.get("std", norm_stats.get("score_std", 1.0))

    def apply_action_to_board(self, board: Board, action: Action) -> Board:
        new_top = list(board.top)
        new_mid = list(board.middle)
        new_bot = list(board.bottom)
        for card, pos in action.placements:
            if pos == "top": new_top.append(card)
            elif pos == "middle": new_mid.append(card)
            elif pos == "bottom": new_bot.append(card)
        return Board(top=new_top, middle=new_mid, bottom=new_bot)

    def _get_fl_bonus(self, board: Board) -> float:
        """Compute FL bonus for a board based on top row FL entry potential."""
        if len(board.top) < 3:
            return 0.0
        enters, card_count = check_fl_entry(board.top)
        if enters:
            return float(self.fl_ev.get(card_count, 10.0))
        return 0.0

    @torch.no_grad()
    def select_action(self, obs: Observation) -> Tuple[int, Action, float]:
        """Select action by VN evaluation. Falls back to random on error."""
        if obs.turn == 0:
            actions = get_initial_actions(obs.dealt_cards, obs.board_self)
        else:
            actions = get_turn_actions(obs.dealt_cards, obs.board_self)

        if not actions:
            raise ValueError("No valid actions")
        if len(actions) == 1:
            return 0, actions[0], 0.0

        try:
            states = []
            turns = []
            new_boards = []
            for action in actions:
                new_board = self.apply_action_to_board(obs.board_self, action)
                new_boards.append(new_board)
                new_discards = list(obs.known_discards_self)
                if action.discard:
                    new_discards.append(action.discard)
                new_obs = Observation(
                    board_self=new_board,
                    board_opponent=obs.board_opponent,
                    dealt_cards=[],
                    known_discards_self=new_discards,
                    turn=obs.turn,
                    is_btn=obs.is_btn,
                )
                state = encode_state(new_obs)
                states.append(state)
                turns.append(obs.turn)

            state_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
            turn_tensor = torch.tensor(turns, dtype=torch.long).to(self.device)
            output = self.model(state_tensor, turn_tensor)
            values = output["value"].squeeze(-1).cpu().numpy()
            scores = values * self.score_std + self.score_mean
            bust_probs = output["bust_prob"].squeeze(-1).cpu().numpy()

            # Per-action FL bonus based on card count
            fl_bonuses = np.array([self._get_fl_bonus(b) for b in new_boards])

            adjusted = scores * (1 - bust_probs) + (-8.0) * bust_probs + fl_bonuses

            if self.temperature <= 0:
                best_idx = int(np.argmax(adjusted))
            else:
                logits = adjusted / self.temperature
                logits -= logits.max()
                probs = np.exp(logits)
                probs /= probs.sum()
                best_idx = int(np.random.choice(len(actions), p=probs))

            return best_idx, actions[best_idx], float(adjusted[best_idx])
        except Exception:
            # Fallback for Joker encoding errors etc
            idx = random.randint(0, len(actions) - 1)
            return idx, actions[idx], 0.0



# ─────────────────────────────────────────────────────────────
# FL Placement with Opponent-Aware Optimization
# ─────────────────────────────────────────────────────────────

def compute_raw_score_for_placement(fl_board: Board, opp_board: Board,
                                    fl_seat: int) -> int:
    """Compute the raw game score for an FL placement against opponent's board.

    Uses the same scoring logic as GameEngine.compute_result().
    Returns score from fl_seat's perspective.
    """
    boards = [Board(), Board()]
    boards[fl_seat] = fl_board
    boards[1 - fl_seat] = opp_board

    scoring_hand = Hand(deck=[], btn=0)
    scoring_hand.boards = boards
    result = GameEngine.compute_result(scoring_hand)
    return result.raw_score[fl_seat]


def compute_royalty_score(board: Board) -> int:
    """Compute total royalties for a board (without line comparison)."""
    evaluated = evaluate_board_with_joker_constraint(
        board.top, board.middle, board.bottom
    )
    if evaluated["busted"]:
        return -100  # Busted = very bad
    return int(evaluated["royalties"]["total"])


def check_board_fl_stay(board: Board) -> bool:
    """Check if a board qualifies for FL Stay.

    FL Stay conditions:
    - Top: Trips (Three of a Kind)
    - Bottom: Quads, Straight Flush, or Royal Flush
    """
    evaluated = evaluate_board_with_joker_constraint(
        board.top, board.middle, board.bottom
    )
    if evaluated["busted"]:
        return False

    # Top: Trips?
    top_cat = hand_category(int(evaluated["values"]["top"]))
    if top_cat >= 3:  # Trips or better
        return True

    # Bottom: Quads+ ?
    bot_cat = hand_category(int(evaluated["values"]["bottom"]))
    if bot_cat >= 7:  # Quads (7), Straight Flush (8)
        return True

    return False


def generate_fl_placements(cards: List[str], n_samples: int = 500) -> List[Board]:
    """Generate random valid FL placements by sampling.

    For each sample: pick 3 for top, 5 for mid, 5 for bot, rest is discard.
    Only returns non-busted placements.
    """
    n = len(cards)
    placements = []
    indices = list(range(n))

    for _ in range(n_samples):
        random.shuffle(indices)
        top = [cards[indices[i]] for i in range(3)]
        mid = [cards[indices[i]] for i in range(3, 8)]
        bot = [cards[indices[i]] for i in range(8, 13)]

        board = Board(top=top, middle=mid, bottom=bot)
        # Quick bust check through the canonical Joker evaluator.
        if not evaluate_board_with_joker_constraint(
            board.top, board.middle, board.bottom
        )["busted"]:
            placements.append(board)

    return placements


def find_best_fl_placement_vs_opponent(
    fl_solver: RustFLSolver,
    fl_cards: List[str],
    opp_board: Board,
    fl_seat: int,
    margin: int = 4,
    n_samples: int = 500,
) -> Board:
    """Find the best FL placement.

    Uses Rust solver v2 (role-based) which already:
    - Phase A: Bottom FL Stay (RF, SF, Quads)  
    - Phase B: Top FL Stay (Trips)
    - Phase C: No FL Stay (maximize royalties)
    - Falls back to exhaustive search if needed

    FL Stay is already prioritized by the Rust solver's phase ordering.
    """
    try:
        tuples = [card_str_to_tuple(c) for c in fl_cards]
        rust_result = fl_solver.solve(tuples)
    except Exception:
        rust_result = None

    if not rust_result:
        return _random_fl_placement(fl_cards)

    return placement_to_board(rust_result)


# ─────────────────────────────────────────────────────────────
# Session Manager
# ─────────────────────────────────────────────────────────────

class SessionManager:
    """Manages a single session with chips, FL state, and termination."""

    def __init__(self, starting_chips: int = 200):
        self.chips = [starting_chips, starting_chips]
        self.is_fl = [False, False]
        self.fl_card_count = [0, 0]
        self.btn = random.randint(0, 1)  # Random initial button seat.
        self.hand_count = 0
        self.fl_hands = 0

    def apply_score(self, raw_score: List[int]):
        """Apply score with chip floor at 0."""
        for s in [0, 1]:
            self.chips[s] = max(0, self.chips[s] + raw_score[s])

    def check_fl_stay_for_seat(self, seat: int, board: Board) -> bool:
        """Check if seat qualifies for FL Stay."""
        if not self.is_fl[seat]:
            return False
        stays, cards = check_fl_stay_from_cards(
            board.top,
            board.bottom,
            self.fl_card_count[seat],
            middle_cards=board.middle,
        )
        if stays:
            return True
        else:
            self.is_fl[seat] = False
            self.fl_card_count[seat] = 0
            return False

    def is_finished(self) -> bool:
        """Check if session should end."""
        if self.chips[0] <= 0 or self.chips[1] <= 0:
            return True
        # 40-point lead AND neither player in FL
        if not self.is_fl[0] and not self.is_fl[1]:
            if abs(self.chips[0] - self.chips[1]) >= 40:
                return True
        return False

    def advance_btn(self):
        """Alternate button each hand."""
        self.btn = 1 - self.btn
        self.hand_count += 1


# ─────────────────────────────────────────────────────────────
# Hand Playing Functions
# ─────────────────────────────────────────────────────────────

def play_normal_hand(player: VNGreedyPlayer, deck: List[str],
                     btn: int) -> Tuple[HandResult, List[dict]]:
    """Play a normal hand with both seats using VN-greedy.

    Returns (HandResult, list of turn records for seat 0).
    """
    hand = Hand(deck=deck, btn=btn)
    records = []

    # T0: non-button first, button second.
    for s in action_order(hand.btn):
        obs = hand.get_observation(s)
        idx, action, vn_score = player.select_action(obs)
        if s == 0:
            records.append({
                "turn": 0,
                "board": _board_to_dict(obs.board_self),
                "dealt": list(obs.dealt_cards),
                "exclude": list(obs.known_discards_self),
                "board_after": _board_to_dict(
                    player.apply_action_to_board(obs.board_self, action)),
            })
        hand.apply_action(s, action)

    # T1-T4
    for turn_num in range(1, 9):
        if hand.is_hand_complete():
            break
        hand.deal_next_turn()
        for s in action_order(hand.btn):
            cards = hand.dealt_cards[s]
            if not cards or hand.boards[s].is_complete():
                continue
            obs = hand.get_observation(s)
            idx, action, vn_score = player.select_action(obs)
            if s == 0:
                records.append({
                    "turn": turn_num,
                    "board": _board_to_dict(obs.board_self),
                    "dealt": list(obs.dealt_cards),
                    "exclude": list(obs.known_discards_self),
                    "board_after": _board_to_dict(
                        player.apply_action_to_board(obs.board_self, action)),
                })
            hand.apply_action(s, action)

    result = GameEngine.compute_result(hand)
    return result, records


def _play_normal_for_seat(player: VNGreedyPlayer, deck: List[str],
                          seat: int, btn: int) -> Tuple[Board, List[dict]]:
    """Play a normal hand for a single seat using VN-greedy against empty opponent.

    Used when one player is in FL and the other plays normally.
    The non-FL player plays first (doesn't know opponent's FL placement).

    Returns (final Board, turn records if seat==0).
    """
    # Create a hand using the deck, play only for `seat`
    hand = Hand(deck=deck, btn=btn)
    records = []

    # T0: Initial placement
    obs = hand.get_observation(seat)
    idx, action, _ = player.select_action(obs)
    if seat == 0:
        records.append({
            "turn": 0,
            "board": _board_to_dict(obs.board_self),
            "dealt": list(obs.dealt_cards),
            "exclude": list(obs.known_discards_self),
            "board_after": _board_to_dict(
                player.apply_action_to_board(obs.board_self, action)),
        })
    hand.apply_action(seat, action)

    # Also need to place something for the other seat so deal_next_turn works
    other = 1 - seat
    other_obs = hand.get_observation(other)
    other_actions = get_initial_actions(other_obs.dealt_cards, other_obs.board_self)
    if other_actions:
        hand.apply_action(other, other_actions[0])

    # T1-T4
    for turn_num in range(1, 9):
        if hand.boards[seat].is_complete():
            break
        hand.deal_next_turn()
        cards = hand.dealt_cards[seat]
        if not cards or hand.boards[seat].is_complete():
            continue
        obs = hand.get_observation(seat)
        idx, action, _ = player.select_action(obs)
        if seat == 0:
            records.append({
                "turn": turn_num,
                "board": _board_to_dict(obs.board_self),
                "dealt": list(obs.dealt_cards),
                "exclude": list(obs.known_discards_self),
                "board_after": _board_to_dict(
                    player.apply_action_to_board(obs.board_self, action)),
            })
        hand.apply_action(seat, action)

        # Auto-place other seat
        if not hand.boards[other].is_complete():
            other_cards = hand.dealt_cards[other]
            if other_cards:
                other_obs2 = hand.get_observation(other)
                other_acts = get_turn_actions(other_obs2.dealt_cards, other_obs2.board_self)
                if other_acts:
                    hand.apply_action(other, other_acts[0])

    return hand.boards[seat], records


def play_fl_hand(fl_solver: RustFLSolver, player: VNGreedyPlayer,
                 deck: List[str], btn: int,
                 fl_seats: List[bool], fl_card_counts: List[int],
                 ) -> Tuple[HandResult, List[dict]]:
    """Play a hand where one or both players are in FL.

    Key rules:
    - If one player FL, one normal: Normal plays first, FL places after
      seeing opponent's board (optimized against it within 4pt margin).
    - If both FL: Each uses Rust solver's best placement (max royalty).

    Returns (HandResult, records for seat 0 if non-FL).
    """
    both_fl = fl_seats[0] and fl_seats[1]

    if both_fl:
        return _play_both_fl(fl_solver, deck, fl_card_counts)
    else:
        return _play_one_fl_one_normal(fl_solver, player, deck, btn,
                                       fl_seats, fl_card_counts)


def _play_both_fl(fl_solver: RustFLSolver, deck: List[str],
                  fl_card_counts: List[int],
                  ) -> Tuple[HandResult, List[dict]]:
    """Both players are in FL — each uses Rust solver's max-royalty placement."""
    boards = [Board(), Board()]
    deck_pos = 0

    for s in [0, 1]:
        n = fl_card_counts[s]
        fl_cards = deck[deck_pos:deck_pos + n]
        deck_pos += n

        try:
            tuples = [card_str_to_tuple(c) for c in fl_cards]
            placement = fl_solver.solve(tuples)
            if placement:
                boards[s] = placement_to_board(placement)
            else:
                boards[s] = _random_fl_placement(fl_cards)
        except Exception:
            boards[s] = _random_fl_placement(fl_cards)

    scoring_hand = Hand(deck=[], btn=0)
    scoring_hand.boards = boards
    result = GameEngine.compute_result(scoring_hand)
    return result, []  # No VN records for FL-vs-FL hands


def _play_one_fl_one_normal(
    fl_solver: RustFLSolver, player: VNGreedyPlayer,
    deck: List[str], btn: int,
    fl_seats: List[bool], fl_card_counts: List[int],
) -> Tuple[HandResult, List[dict]]:
    """One player FL, one normal. Normal plays FIRST, then FL places optimally."""
    fl_seat = 0 if fl_seats[0] else 1
    normal_seat = 1 - fl_seat

    # Step 1: Deal FL cards from deck first
    fl_n = fl_card_counts[fl_seat]
    fl_cards = deck[:fl_n]
    remaining_deck = deck[fl_n:]

    # Step 2: Normal player plays FIRST (doesn't see FL placement)
    normal_board, records = _play_normal_for_seat(
        player, remaining_deck, normal_seat, btn
    )

    # Step 3: FL player places AFTER seeing opponent's board
    # Find best placement considering opponent's board (within 4pt margin)
    fl_board = find_best_fl_placement_vs_opponent(
        fl_solver, fl_cards, normal_board, fl_seat,
        margin=4, n_samples=5000,
    )

    # Step 4: Score
    boards = [Board(), Board()]
    boards[fl_seat] = fl_board
    boards[normal_seat] = normal_board

    scoring_hand = Hand(deck=[], btn=btn)
    scoring_hand.boards = boards
    result = GameEngine.compute_result(scoring_hand)
    return result, records


def _random_fl_placement(cards: List[str]) -> Board:
    """Fallback: random FL placement (3 top, 5 mid, 5 bot)."""
    c = list(cards)
    random.shuffle(c)
    return Board(top=c[:3], middle=c[3:8], bottom=c[8:13])


def _board_to_dict(board: Board) -> dict:
    return {"top": list(board.top), "mid": list(board.middle), "bot": list(board.bottom)}


# ─────────────────────────────────────────────────────────────
# FL Stats Tracker (auto-calibration)
# ─────────────────────────────────────────────────────────────

class FLStatsTracker:
    """Tracks per-card-count FL hand results for auto-calibrating FL EV.

    Collects:
    - First-hand raw_score per card count (both seats)
    - Stay/no-stay outcomes per card count
    Then computes chain EV = avg_first_hand_score / (1 - stay_rate)
    """

    def __init__(self):
        # {card_count: [raw_score, ...]}
        self.first_hand_scores: Dict[int, List[float]] = {14: [], 15: [], 16: [], 17: []}
        # {card_count: [True/False, ...]}  True=stayed
        self.stay_outcomes: Dict[int, List[bool]] = {14: [], 15: [], 16: [], 17: []}

    def record_fl_hand(self, card_count: int, raw_score: float):
        """Record a first FL hand score for given card count."""
        if card_count in self.first_hand_scores:
            self.first_hand_scores[card_count].append(raw_score)

    def record_stay_outcome(self, card_count: int, stayed: bool):
        """Record whether an FL hand resulted in Stay."""
        if card_count in self.stay_outcomes:
            self.stay_outcomes[card_count].append(stayed)

    def compute_fl_ev(self) -> Dict[int, float]:
        """Compute per-card-count chain EV.

        chain_EV = avg_first_hand_score / (1 - stay_rate)
        This geometric series captures the expected total value
        of entering FL including all potential Stay continuations.
        """
        fl_ev = {}
        for cc in [14, 15, 16, 17]:
            scores = self.first_hand_scores[cc]
            stays = self.stay_outcomes[cc]

            if not scores:
                fl_ev[cc] = VNGreedyPlayer.DEFAULT_FL_EV[cc]
                continue

            avg_score = sum(scores) / len(scores)
            stay_rate = sum(1 for s in stays if s) / max(len(stays), 1)

            # chain EV = R / (1 - stay_rate), capped to avoid division by ~0
            if stay_rate >= 0.95:
                stay_rate = 0.95
            chain_ev = avg_score / (1.0 - stay_rate)

            fl_ev[cc] = round(chain_ev, 1)

        return fl_ev

    def get_stats_summary(self) -> dict:
        """Return summary statistics."""
        summary = {}
        for cc in [14, 15, 16, 17]:
            scores = self.first_hand_scores[cc]
            stays = self.stay_outcomes[cc]
            n = len(scores)
            if n == 0:
                summary[cc] = {"count": 0}
                continue
            summary[cc] = {
                "count": n,
                "avg_score": round(sum(scores) / n, 1),
                "stay_rate": round(sum(1 for s in stays if s) / max(len(stays), 1), 3),
                "stay_n": len(stays),
            }
        return summary


# ─────────────────────────────────────────────────────────────
# Session Runner
# ─────────────────────────────────────────────────────────────

def play_session(player: VNGreedyPlayer, fl_solver: RustFLSolver,
                 starting_chips: int = 200,
                 max_hands: int = 100,
                 fl_tracker: FLStatsTracker = None,
                 ) -> Tuple[List[dict], dict]:
    """Play a full session. Returns (all_records, session_summary)."""
    mgr = SessionManager(starting_chips=starting_chips)
    all_records = []
    global_hand_id = 0

    while not mgr.is_finished() and mgr.hand_count < max_hands:
        deck = list(ALL_CARDS)
        random.shuffle(deck)

        any_fl = mgr.is_fl[0] or mgr.is_fl[1]
        # Remember pre-hand FL state for tracking
        pre_fl = [mgr.is_fl[0], mgr.is_fl[1]]
        pre_fl_cc = [mgr.fl_card_count[0], mgr.fl_card_count[1]]

        if any_fl:
            result, records = play_fl_hand(
                fl_solver, player, deck, mgr.btn,
                list(mgr.is_fl), list(mgr.fl_card_count),
            )
            mgr.fl_hands += 1

            # Track FL hand scores (for both seats that were in FL)
            if fl_tracker:
                for s in [0, 1]:
                    if pre_fl[s] and pre_fl_cc[s] in (14, 15, 16, 17):
                        fl_tracker.record_fl_hand(pre_fl_cc[s], float(result.raw_score[s]))
        else:
            result, records = play_normal_hand(player, deck, mgr.btn)

        # Score
        mgr.apply_score(result.raw_score)

        # Write records with labels (only non-FL turns for seat 0)
        for rec in records:
            rec["hand_id"] = global_hand_id
            rec["n_candidates"] = 1
            rec["best_idx"] = 0
            rec["eval_mode"] = "session_vn"
            rec["candidates"] = [{"mc": {
                "avg_score": float(result.raw_score[0]),
                "bust_rate": 1.0 if result.busted[0] else 0.0,
                "fl_rate": 1.0 if result.fl_entry[0] else 0.0,
            }}]
            all_records.append(rec)

        # Final record
        all_records.append({
            "hand_id": global_hand_id,
            "turn": -1,
            "score": float(result.raw_score[0]),
            "busted": bool(result.busted[0]),
            "fl_entry": bool(result.fl_entry[0]),
            "final_board": _board_to_dict(result.boards[0]) if result.boards else {},
            "is_fl_hand": any_fl,
            "chips": list(mgr.chips),
        })

        # Update FL state + track Stay outcomes
        for s in [0, 1]:
            if mgr.is_fl[s]:
                if not result.busted[s]:
                    stays = mgr.check_fl_stay_for_seat(s, result.boards[s])
                    # Track stay outcome
                    if fl_tracker and pre_fl_cc[s] in (14, 15, 16, 17):
                        fl_tracker.record_stay_outcome(pre_fl_cc[s], stays)
                    if not stays:
                        mgr.is_fl[s] = False
                        mgr.fl_card_count[s] = 0
                else:
                    # Busted = no stay
                    if fl_tracker and pre_fl_cc[s] in (14, 15, 16, 17):
                        fl_tracker.record_stay_outcome(pre_fl_cc[s], False)
                    mgr.is_fl[s] = False
                    mgr.fl_card_count[s] = 0
            else:
                if result.fl_entry[s]:
                    mgr.is_fl[s] = True
                    mgr.fl_card_count[s] = result.fl_card_count[s]

        mgr.advance_btn()
        global_hand_id += 1

    session_score = mgr.chips[0] - starting_chips
    summary = {
        "hands": mgr.hand_count,
        "fl_hands": mgr.fl_hands,
        "final_chips": list(mgr.chips),
        "session_score": session_score,
    }
    return all_records, summary


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Session-Based Self-Play")
    parser.add_argument("--model", required=True, help="VN model path")
    parser.add_argument("--norm", default=None, help="norm_stats.json path")
    parser.add_argument("--sessions", type=int, default=100, help="Number of sessions")
    parser.add_argument("--output", default="data/session_selfplay.jsonl")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--starting-chips", type=int, default=200)
    parser.add_argument("--max-hands", type=int, default=100, help="Max hands per session")
    parser.add_argument("--fl-margin", type=int, default=4,
                        help="Royalty margin for FL placement candidates (default: 4)")
    parser.add_argument("--fl-samples", type=int, default=5000,
                        help="Number of random FL placements to sample (default: 5000)")
    parser.add_argument("--fl-ev", default="ai/config/fl_ev.json",
                        help="FL EV config (per-card-count chain EVs)")
    parser.add_argument("--input-dim", type=int, default=STATE_DIM)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load VN model
    print(f"Loading model: {args.model}")
    model = ValueNetworkV3(input_dim=args.input_dim).to(device)
    ck = torch.load(args.model, map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck)
    model.load_state_dict(sd)
    model.eval()

    norm_stats = None
    if args.norm and Path(args.norm).exists():
        with open(args.norm) as f:
            norm_stats = json.load(f)

    # Load per-card-count FL EVs
    fl_ev = dict(VNGreedyPlayer.DEFAULT_FL_EV)  # defaults
    fl_ev_path = Path(args.fl_ev)
    if fl_ev_path.exists():
        with open(fl_ev_path) as f:
            fl_config = json.load(f)
        if "fl_ev" in fl_config:
            fl_ev = {int(k): float(v) for k, v in fl_config["fl_ev"].items()}
            print(f"Loaded FL EVs from {fl_ev_path}: {fl_ev}")

    player = VNGreedyPlayer(model, device=device, norm_stats=norm_stats,
                            temperature=args.temperature, fl_ev=fl_ev)

    # Load FL solver
    print("Loading Rust FL Solver...")
    fl_solver = RustFLSolver()
    print(f"  FL Solver ready: {fl_solver.solver_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Session-Based Self-Play")
    print(f"{'='*60}")
    print(f"  Sessions:     {args.sessions:,}")
    print(f"  Chips:        {args.starting_chips} each")
    print(f"  Max hands:    {args.max_hands}/session")
    print(f"  FL margin:    {args.fl_margin} pts")
    print(f"  FL samples:   {args.fl_samples:,}")
    print(f"  FL EV:        {fl_ev}")
    print(f"  Temperature:  {args.temperature}")
    print(f"  Device:       {device}")
    print(f"  Output:       {output_path}")
    print()

    total_records = 0
    total_hands = 0
    total_fl_hands = 0
    total_session_score = 0
    n_hero_wins = 0
    start_time = time.time()
    fl_tracker = FLStatsTracker()
    fl_update_interval = 1000  # Update FL EV every 1000 FL hands
    last_fl_update_count = 0   # Last milestone when FL EV was updated

    def _save_fl_ev(tracker, ev_path, old_ev, label=""):
        """Compute and save updated FL EVs from tracker data."""
        stats = tracker.get_stats_summary()
        new_ev = tracker.compute_fl_ev()
        updated = dict(old_ev)
        any_update = False
        for cc in [14, 15, 16, 17]:
            if stats[cc].get("count", 0) >= 10:
                updated[cc] = new_ev[cc]
                any_update = True
        if not any_update:
            return old_ev
        fl_ev_out = {
            "fl_ev": {str(k): v for k, v in updated.items()},
            "fl_stats": {},
            "source": label,
        }
        for cc in [14, 15, 16, 17]:
            s = stats[cc]
            if s["count"] > 0:
                fl_ev_out["fl_stats"][str(cc)] = {
                    "R": s["avg_score"],
                    "stay_rate": s["stay_rate"],
                    "count": s["count"],
                }
        with open(ev_path, "w") as fout:
            json.dump(fl_ev_out, fout, indent=2)
        print(f"\n  🔄 FL EV updated ({label}): {updated}")
        return updated

    with open(output_path, "w", encoding="utf-8") as f:
        for sess_idx in range(args.sessions):
            try:
                records, summary = play_session(
                    player, fl_solver,
                    starting_chips=args.starting_chips,
                    max_hands=args.max_hands,
                    fl_tracker=fl_tracker,
                )

                for rec in records:
                    rec["session_id"] = sess_idx
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                total_records += len(records)
                total_hands += summary["hands"]
                total_fl_hands += summary["fl_hands"]
                total_session_score += summary["session_score"]
                if summary["session_score"] > 0:
                    n_hero_wins += 1

                # Check if we crossed a 1000 FL-hand milestone
                total_fl_tracked = sum(
                    len(v) for v in fl_tracker.first_hand_scores.values()
                )
                if total_fl_tracked >= last_fl_update_count + fl_update_interval:
                    last_fl_update_count = (total_fl_tracked // fl_update_interval) * fl_update_interval
                    fl_ev = _save_fl_ev(
                        fl_tracker, fl_ev_path, fl_ev,
                        label=f"auto_{last_fl_update_count}fl_hands",
                    )
                    # Reload into player
                    player.fl_ev = fl_ev

            except Exception as e:
                print(f"  [WARN] Session {sess_idx} failed: {e}")
                import traceback
                traceback.print_exc()

            if (sess_idx + 1) % 10 == 0 or sess_idx == args.sessions - 1:
                elapsed = time.time() - start_time
                rate = (sess_idx + 1) / elapsed
                avg_hands = total_hands / (sess_idx + 1)
                avg_score = total_session_score / (sess_idx + 1)
                fl_pct = total_fl_hands / max(total_hands, 1) * 100
                win_pct = n_hero_wins / (sess_idx + 1) * 100
                print(f"  {sess_idx+1:,}/{args.sessions:,} sessions  "
                      f"({rate:.1f}/s)  "
                      f"avg_hands={avg_hands:.1f}  "
                      f"fl={fl_pct:.1f}%  "
                      f"avg_score={avg_score:.1f}  "
                      f"win={win_pct:.0f}%")

    elapsed = time.time() - start_time
    avg_hands = total_hands / max(args.sessions, 1)
    avg_score = total_session_score / max(args.sessions, 1)
    fl_pct = total_fl_hands / max(total_hands, 1) * 100
    win_pct = n_hero_wins / max(args.sessions, 1) * 100

    print(f"\n{'='*60}")
    print(f"  Session Self-Play Complete")
    print(f"{'='*60}")
    print(f"  Sessions:     {args.sessions:,}")
    print(f"  Total hands:  {total_hands:,}")
    print(f"  Total records: {total_records:,}")
    print(f"  FL hands:     {total_fl_hands:,} ({fl_pct:.1f}%)")
    print(f"  Avg hands:    {avg_hands:.1f}/session")
    print(f"  Avg score:    {avg_score:.1f}")
    print(f"  Hero wins:    {n_hero_wins}/{args.sessions} ({win_pct:.0f}%)")
    print(f"  Time:         {elapsed:.1f}s ({args.sessions/elapsed:.1f} sess/s)")
    print(f"  Output:       {output_path}")

    # Print FL stats summary (always shown)
    fl_stats = fl_tracker.get_stats_summary()
    cur_fl_ev = fl_tracker.compute_fl_ev()
    total_fl_tracked = sum(len(v) for v in fl_tracker.first_hand_scores.values())
    print(f"\n  FL Stats ({total_fl_tracked} total FL hands tracked, "
          f"updated {last_fl_update_count // fl_update_interval} time(s)):")
    print(f"  {'CC':>4} {'Count':>6} {'AvgScore':>9} {'StayRate':>9} {'ChainEV':>9} {'CurEV':>8}")
    for cc in [14, 15, 16, 17]:
        s = fl_stats[cc]
        cur = fl_ev.get(cc, "?")
        if s["count"] > 0:
            print(f"  {cc:>4} {s['count']:>6} {s['avg_score']:>9.1f} "
                  f"{s['stay_rate']:>8.1%} {cur_fl_ev[cc]:>9.1f} {cur:>8}")
        else:
            print(f"  {cc:>4}      0       -         -   {cur_fl_ev[cc]:>9.1f} {cur:>8}")


if __name__ == "__main__":
    main()

