"""
OFC Pineapple CFR - Depth-Limited Outcome Sampling MCCFR (v3)

Key design decisions to handle OFC's massive game tree:

1. **Outcome Sampling at T0**: Sample ONE action for traverser
   (instead of exploring all) with importance sampling correction.
   This reduces branching from 30x to 1x.

2. **External Sampling at T1+**: Traverse all ~18-26 actions since
   each decision only leads to 1 subsequent decision (the opponent),
   making it manageable.

3. **Depth Limit**: CFR traversal stops after `max_depth` decision
   nodes. Leaf states are evaluated by a heuristic (royalties +
   FL potential + hand strength).

4. **Warm-Start Pruning**: After initial iterations, prune actions
   with deeply negative regret.

Effective tree size per iteration:
  T0: 1 (sampled) × 1 (opponent sampled) = 1 path to T1
  T1: ~26 (traverser explores all) × 1 (opponent sampled) = 26
  T2: ~20 × 1 = 20
  Total ≈ 26 × 20 = 520 leaf evaluations per iteration
  → ~0.05s per iteration → 1000 iterations in ~50s ✓
"""
import random
import time
import pickle
import json
import os
import math
import hashlib
import warnings
import numpy as np
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.cfr.game_state import OFCState, NodeType, create_initial_state, FL_CHAIN_EV
from ai.cfr.abstraction import abstract_info_set
from ai.engine.action_space import Action
from ai.engine.encoding import Board, RANK_VALUES
from ai.engine.turn_order import POSITION_CONTRACT_VERSION
from ai.engine.game_engine import (
    evaluate_hand, hand_category, check_fl_entry,
    get_top_royalty, get_middle_royalty, get_bottom_royalty,
)


CFR_RULES_VERSION = "canonical_joker_bottom_middle_top_20260711"
CFR_ACTION_SCHEMA = "legacy_python_t0_enumeration_t1_t4_compressed_legal_list_index_v1"
CFR_INFO_MODEL = "snapshot_boards_current_hand_own_discards_no_public_history_v1"
CFR_CHECKPOINT_CONTRACT_VERSION = "cfr_checkpoint_contract_v1"
CFR_FL_CONFIG_PATH = Path(__file__).resolve().parents[2] / "ai" / "config" / "fl_ev.json"
CFR_CHECKPOINT_REQUIRED_FIELDS = (
    "position_contract_version",
    "rules_version",
    "action_schema",
    "info_model",
    "fl_config_sha256",
)


def _current_fl_config_sha256() -> str:
    """Hash the exact Fantasyland value configuration used by this checkout."""
    return hashlib.sha256(CFR_FL_CONFIG_PATH.read_bytes()).hexdigest()


def _checkpoint_contract_metadata() -> Dict[str, str]:
    return {
        "checkpoint_contract_version": CFR_CHECKPOINT_CONTRACT_VERSION,
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "rules_version": CFR_RULES_VERSION,
        "action_schema": CFR_ACTION_SCHEMA,
        "info_model": CFR_INFO_MODEL,
        "fl_config_sha256": _current_fl_config_sha256(),
    }


# ─── Info Set Data ────────────────────────────────────────────────

@dataclass
class InfoSetData:
    """Data stored per information set."""
    cumulative_regret: Dict[int, float] = field(default_factory=lambda: defaultdict(float))
    cumulative_strategy: Dict[int, float] = field(default_factory=lambda: defaultdict(float))
    reach_count: int = 0

    def get_strategy(self, n_actions: int) -> np.ndarray:
        """Regret Matching+ strategy."""
        strategy = np.zeros(n_actions, dtype=np.float64)
        for a in range(n_actions):
            strategy[a] = max(0.0, self.cumulative_regret.get(a, 0.0))
        total = strategy.sum()
        if total > 0:
            strategy /= total
        else:
            strategy[:] = 1.0 / n_actions
        return strategy

    def get_average_strategy(self, n_actions: int) -> np.ndarray:
        """Average strategy (converges to Nash equilibrium)."""
        avg = np.zeros(n_actions, dtype=np.float64)
        for a in range(n_actions):
            avg[a] = max(0.0, self.cumulative_strategy.get(a, 0.0))
        total = avg.sum()
        if total > 0:
            avg /= total
        else:
            avg[:] = 1.0 / n_actions
        return avg

    def update_strategy_sum(self, strategy: np.ndarray, weight: float = 1.0):
        for a in range(len(strategy)):
            self.cumulative_strategy[a] = self.cumulative_strategy.get(a, 0.0) + weight * strategy[a]
        self.reach_count += 1


# ─── Info Set Store ──────────────────────────────────────────────

class InfoSetStore:
    """Storage for all information set data."""

    def __init__(self):
        self.data: Dict[str, InfoSetData] = {}
        self._action_counts: Dict[str, int] = {}

    def get(self, key: str, n_actions: int) -> InfoSetData:
        if key not in self.data:
            self.data[key] = InfoSetData()
            self._action_counts[key] = n_actions
        return self.data[key]

    def get_strategy(self, key: str, n_actions: int) -> np.ndarray:
        return self.get(key, n_actions).get_strategy(n_actions)

    def get_average_strategy(self, key: str, n_actions: int) -> np.ndarray:
        return self.get(key, n_actions).get_average_strategy(n_actions)

    @property
    def size(self) -> int:
        return len(self.data)

    def total_regret(self) -> float:
        total = 0.0
        for isd in self.data.values():
            for r in isd.cumulative_regret.values():
                total += abs(r)
        return total

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({"data": dict(self.data), "action_counts": dict(self._action_counts)}, f)

    def load(self, filepath: str):
        with open(filepath, "rb") as f:
            saved = pickle.load(f)
        self.data = saved["data"]
        self._action_counts = saved.get("action_counts", {})

    def export_strategy(self, filepath: str, top_n: int = 100):
        strategies = {}
        sorted_keys = sorted(self.data.keys(), key=lambda k: self.data[k].reach_count, reverse=True)
        for key in sorted_keys[:top_n]:
            n_actions = self._action_counts.get(key, 0)
            if n_actions == 0:
                continue
            avg = self.data[key].get_average_strategy(n_actions)
            strategies[key] = {
                "strategy": avg.tolist(),
                "reach_count": self.data[key].reach_count,
                "n_actions": n_actions,
            }
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(strategies, f, indent=2)


# ─── Heuristic Leaf Evaluation ───────────────────────────────────

def evaluate_board_heuristic(board: Board, opp_board: Board, is_btn: bool) -> float:
    """
    Heuristic evaluation of a (possibly incomplete) board state.

    Returns estimated score from the player's perspective.
    Components:
    1. Current row royalties (if rows are complete)
    2. FL potential bonus
    3. Bust risk penalty
    4. Hand strength differential vs opponent
    """
    score = 0.0

    # ─── Current royalties (partial) ─────────────────────────
    if len(board.top) == 3:
        score += get_top_royalty(board.top) * 1.0
    if len(board.middle) == 5:
        score += get_middle_royalty(board.middle) * 1.0
    if len(board.bottom) == 5:
        score += get_bottom_royalty(board.bottom) * 1.0

    # ─── FL potential ────────────────────────────────────────
    if len(board.top) >= 2:
        fl, fl_cards = check_fl_entry(board.top + ['2s'] * (3 - len(board.top)))
        if fl:
            ev = FL_CHAIN_EV.get(fl_cards, 0)
            score += ev * 0.8  # Discount since board may not be complete

    # Also check partial FL (has A/K pair started)
    top_ranks = [RANK_VALUES.get(c[0], 0) for c in board.top if not c.startswith('X')]
    top_jokers = sum(1 for c in board.top if c.startswith('X'))
    rank_counts = Counter(top_ranks)
    for r, cnt in rank_counts.items():
        if cnt + top_jokers >= 2 and r >= 12:  # QQ+
            if len(board.top) < 3:
                score += FL_CHAIN_EV.get(14, 14.0) * 0.3  # Partial bonus

    # ─── Bust risk ───────────────────────────────────────────
    if len(board.top) >= 2 and len(board.middle) >= 3:
        top_val = evaluate_hand(board.top + ['2s'] * (3 - len(board.top)), 3)
        mid_val = evaluate_hand(board.middle + ['2s'] * (5 - len(board.middle)), 5)
        if top_val > mid_val:
            score -= 8.0  # High bust risk

    if len(board.middle) >= 3 and len(board.bottom) >= 3:
        mid_val = evaluate_hand(board.middle + ['2s'] * (5 - len(board.middle)), 5)
        bot_val = evaluate_hand(board.bottom + ['2s'] * (5 - len(board.bottom)), 5)
        if mid_val > bot_val:
            score -= 6.0

    # ─── Row strength comparison vs opponent ─────────────────
    for row_name, max_size, row_cards, opp_cards in [
        ('top', 3, board.top, opp_board.top),
        ('middle', 5, board.middle, opp_board.middle),
        ('bottom', 5, board.bottom, opp_board.bottom),
    ]:
        if len(row_cards) >= max_size and len(opp_cards) >= max_size:
            my_val = evaluate_hand(row_cards, max_size)
            their_val = evaluate_hand(opp_cards, max_size)
            if my_val > their_val:
                score += 1.5
            elif my_val < their_val:
                score -= 1.5

    # ─── Opponent FL penalty ─────────────────────────────────
    if len(opp_board.top) >= 2:
        opp_fl, opp_fl_cards = check_fl_entry(opp_board.top + ['2s'] * (3 - len(opp_board.top)))
        if opp_fl:
            score -= FL_CHAIN_EV.get(opp_fl_cards, 0) * 0.5

    return score


def _is_valid_t0_action(action: Action) -> bool:
    """Keep every structurally legal T0 action.

    Ace/Joker row preferences are strategy choices, not game rules.  The
    action-space generator has already enforced row capacities, so pruning
    cards by rank here would remove legal actions from CFR and its teachers.
    """
    return True


def _prefilter_actions(actions: List[Action], max_k: int) -> List[int]:
    """Return every legal T0 action for downstream ranking/abstraction."""
    valid = [i for i, a in enumerate(actions) if _is_valid_t0_action(a)]
    return valid if valid else list(range(len(actions)))


# ─── Depth-Limited MCCFR ────────────────────────────────────────

class OFC_CFR:
    """
    Depth-Limited MCCFR for OFC Pineapple.

    Uses:
    - Outcome Sampling at T0 (sample 1 action for traverser)
    - External Sampling at T1+ (explore all actions for traverser)
    - Heuristic leaf evaluation at depth limit
    """

    def __init__(
        self,
        use_abstraction: bool = True,
        t0_top_k: int = 30,
        max_cfr_depth: int = 6,          # Max decision nodes before leaf eval
        prune_threshold: float = -300.0,
        discount_alpha: float = 1.5,
        discount_beta: float = 0.0,
        discount_gamma: float = 2.0,
    ):
        self.store = InfoSetStore()
        self.use_abstraction = use_abstraction
        self.t0_top_k = t0_top_k
        self.max_cfr_depth = max_cfr_depth
        self.prune_threshold = prune_threshold
        self.discount_alpha = discount_alpha
        self.discount_beta = discount_beta
        self.discount_gamma = discount_gamma
        self.iteration = 0

        self.total_nodes_visited = 0
        self.total_terminal_reached = 0
        self.total_leaf_evals = 0

    def _get_info_key(self, state: OFCState, player: int) -> str:
        if self.use_abstraction:
            return abstract_info_set(
                turn=state.turn,
                is_btn=(player == state.btn),
                my_board=state.boards[player],
                opp_board=state.boards[1 - player],
                hand_cards=list(state.hands[player]),
                my_discards=list(state.discards[player]),
            )
        else:
            return state.info_set_key(player)

    def run_iteration(self):
        """Run one CFR iteration (both players as traverser)."""
        for traverser in [0, 1]:
            state = create_initial_state()
            self._cfr_traverse(state, traverser, depth=0)

        self.iteration += 1

        if self.iteration > 0 and self.iteration % 100 == 0:
            self._apply_discount()

    def _cfr_traverse(
        self,
        state: OFCState,
        traverser: int,
        depth: int,
    ) -> float:
        """
        Depth-limited MCCFR traversal.

        At depth limit, use heuristic evaluation.
        """
        self.total_nodes_visited += 1
        node_type = state.node_type

        # ─── Terminal ───
        if node_type == NodeType.TERMINAL:
            self.total_terminal_reached += 1
            return state.terminal_utility(traverser)

        # ─── Depth Limit → Leaf Evaluation ───
        if depth >= self.max_cfr_depth:
            self.total_leaf_evals += 1
            return self._leaf_evaluate(state, traverser)

        # ─── Chance Node ───
        if node_type == NodeType.CHANCE:
            new_state = state.deal_cards()
            return self._cfr_traverse(new_state, traverser, depth)

        # ─── Decision Node ───
        acting_player = int(node_type)
        all_actions = state.get_legal_actions(acting_player)

        if not all_actions:
            new_state = state.copy()
            if acting_player == 0:
                new_state.placed = (True, new_state.placed[1])
            else:
                new_state.placed = (new_state.placed[0], True)
            return self._cfr_traverse(new_state, traverser, depth)

        # Apply T0 constraint filtering
        if state.turn == 0:
            filtered_indices = _prefilter_actions(all_actions, self.t0_top_k)
            actions = [all_actions[i] for i in filtered_indices]
        else:
            actions = all_actions

        n_actions = len(actions)
        info_key = self._get_info_key(state, acting_player)
        info_data = self.store.get(info_key, n_actions)
        strategy = info_data.get_strategy(n_actions)

        if acting_player == traverser:
            # ─── Traverser: External Sampling (explore all) ───
            action_values = np.zeros(n_actions, dtype=np.float64)

            for i, action in enumerate(actions):
                # Pruning for deeply negative regret
                if (self.iteration > 200 and
                    info_data.cumulative_regret.get(i, 0.0) < self.prune_threshold):
                    action_values[i] = 0.0
                    continue

                next_state = state.apply_action(acting_player, action)
                action_values[i] = self._cfr_traverse(next_state, traverser, depth + 1)

            node_value = np.dot(strategy, action_values)

            # Update regrets (RM+)
            for i in range(n_actions):
                regret = action_values[i] - node_value
                new_regret = info_data.cumulative_regret.get(i, 0.0) + regret
                info_data.cumulative_regret[i] = max(new_regret, 0.0)

            info_data.update_strategy_sum(strategy)
            return node_value

        else:
            # ─── Opponent: Sample one action ───
            action_idx = np.random.choice(n_actions, p=strategy)
            action = actions[action_idx]
            next_state = state.apply_action(acting_player, action)
            return self._cfr_traverse(next_state, traverser, depth + 1)

    def _leaf_evaluate(self, state: OFCState, player: int) -> float:
        """Heuristic evaluation at leaf (depth-limited)."""
        # If terminal, use exact scoring
        if state.boards[0].is_complete() and state.boards[1].is_complete():
            return state.terminal_utility(player)

        # Otherwise, heuristic evaluation
        my_board = state.boards[player]
        opp_board = state.boards[1 - player]
        is_btn = (player == state.btn)
        return evaluate_board_heuristic(my_board, opp_board, is_btn)

    def _apply_discount(self):
        """Apply Linear CFR discounting."""
        t = self.iteration
        pos_discount = (t ** self.discount_alpha) / (t ** self.discount_alpha + 1)
        neg_discount = (t ** self.discount_beta) / (t ** self.discount_beta + 1)
        strat_discount = ((t / (t + 1)) ** self.discount_gamma)

        for info_data in self.store.data.values():
            for a in list(info_data.cumulative_regret.keys()):
                r = info_data.cumulative_regret[a]
                if r > 0:
                    info_data.cumulative_regret[a] = r * pos_discount
                else:
                    info_data.cumulative_regret[a] = r * neg_discount
            for a in list(info_data.cumulative_strategy.keys()):
                info_data.cumulative_strategy[a] *= strat_discount

    # ─── Strategy Access ─────────────────────────────────────────

    def get_strategy(self, state: OFCState, player: int) -> Tuple[List[Action], np.ndarray]:
        actions = state.get_legal_actions(player)
        if not actions:
            return [], np.array([])

        if state.turn == 0 and len(actions) > self.t0_top_k:
            filtered_indices = _prefilter_actions(actions, self.t0_top_k)
            actions = [actions[i] for i in filtered_indices]

        info_key = self._get_info_key(state, player)
        avg_strategy = self.store.get_average_strategy(info_key, len(actions))
        return actions, avg_strategy

    def select_action(self, state: OFCState, player: int) -> Tuple[int, Action]:
        actions, strategy = self.get_strategy(state, player)
        if not actions:
            raise ValueError("No legal actions")
        idx = np.random.choice(len(actions), p=strategy)
        return idx, actions[idx]

    def select_action_greedy(self, state: OFCState, player: int) -> Tuple[int, Action]:
        actions, strategy = self.get_strategy(state, player)
        if not actions:
            raise ValueError("No legal actions")
        idx = int(np.argmax(strategy))
        return idx, actions[idx]

    # ─── Persistence ─────────────────────────────────────────────

    def save_checkpoint(self, filepath: str):
        data = {
            "iteration": self.iteration,
            "total_nodes_visited": self.total_nodes_visited,
            "total_terminal_reached": self.total_terminal_reached,
            "total_leaf_evals": self.total_leaf_evals,
            "use_abstraction": self.use_abstraction,
            "t0_top_k": self.t0_top_k,
            "max_cfr_depth": self.max_cfr_depth,
            **_checkpoint_contract_metadata(),
        }
        meta_path = filepath + ".meta.json"
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self.store.save(filepath)

    def load_checkpoint(self, filepath: str, *, allow_legacy: bool = False):
        """Load a checkpoint only after validating its gameplay contract.

        ``allow_legacy`` is intentionally limited to checkpoints whose contract
        metadata is absent or incomplete.  It cannot bypass a field that is
        present but disagrees with the current contract.
        """
        meta_path = filepath + ".meta.json"
        data = None
        if os.path.exists(meta_path):
            try:
                with open(meta_path, encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"CFR checkpoint metadata is unreadable: {meta_path}"
                ) from exc
            if not isinstance(data, dict):
                raise ValueError(
                    f"CFR checkpoint metadata must be a JSON object: {meta_path}"
                )

        expected = _checkpoint_contract_metadata()
        mismatches = {
            field: (expected[field], data[field])
            for field in CFR_CHECKPOINT_REQUIRED_FIELDS
            if data is not None and field in data and data[field] != expected[field]
        }
        if mismatches:
            details = "; ".join(
                f"{field}: expected {wanted!r}, found {actual!r}"
                for field, (wanted, actual) in mismatches.items()
            )
            raise ValueError(
                "CFR checkpoint contract mismatch: "
                f"{details}. allow_legacy cannot override mismatched metadata."
            )

        missing = [
            field
            for field in CFR_CHECKPOINT_REQUIRED_FIELDS
            if data is None or field not in data
        ]
        if missing:
            message = (
                "Legacy CFR checkpoint rejected: missing required contract metadata "
                f"({', '.join(missing)}) in {meta_path}. "
                "Pass allow_legacy=True only for explicit diagnostic access; "
                "the checkpoint is not compatible with the canonical runtime."
            )
            if not allow_legacy:
                raise ValueError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)

        # Contract validation must happen before unpickling the strategy store.
        self.store.load(filepath)
        if data is not None:
            self.iteration = data.get("iteration", 0)
            self.total_nodes_visited = data.get("total_nodes_visited", 0)
            self.total_terminal_reached = data.get("total_terminal_reached", 0)
            self.total_leaf_evals = data.get("total_leaf_evals", 0)

    # ─── Diagnostics ─────────────────────────────────────────────

    def print_stats(self):
        print(f"CFR Statistics (iter={self.iteration}):")
        print(f"  Info Sets:       {self.store.size:,}")
        print(f"  Nodes Visited:   {self.total_nodes_visited:,}")
        print(f"  Terminals:       {self.total_terminal_reached:,}")
        print(f"  Leaf Evals:      {self.total_leaf_evals:,}")
        print(f"  Total Regret:    {self.store.total_regret():,.0f}")
        if self.iteration > 0:
            print(f"  Avg Regret/Iter: {self.store.total_regret() / self.iteration:,.1f}")
