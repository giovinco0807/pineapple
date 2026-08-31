"""
OFC Pineapple - Hybrid MCTS + VN-Greedy Engine

T0: MCTS with PUCT exploration at root, random T1 deals, batch VN evaluation.
    Each simulation: PUCT select → apply T0 action → deal random T1 cards →
    batch-evaluate all T1 candidates with VN → backprop MAX value.
T1+: VN-greedy with FL pair-completion filtering.

Uses BC PolicyNet for priors (T0 pre-filter + PUCT) and VN for evaluation.
"""
import sys
import math
import random
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.encoding import Board, Observation, encode_state, ALL_CARDS
from ai.engine.action_space import (
    get_initial_actions, get_turn_actions, Action, create_action_mask, MAX_ACTIONS
)
from ai.mcts.rollout_evaluator import RolloutEvaluator


# ──────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────

@dataclass
class OFCMCTSConfig:
    num_simulations: int = 1000
    c_puct: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.25
    max_search_depth: int = 3       # search turns beyond current (1=current only)
    fl_ev_scale: float = 0.87       # Optuna-tuned FL EV multiplier
    bust_penalty: float = 16.65     # Optuna-tuned bust penalty
    value_scale: float = 30.0       # normalize raw scores to [-1, 1]
    top_k_filter: int = 20          # T0 policy pre-filter
    vn_top_k: int = 12              # T0 VN pre-filter


# ──────────────────────────────────────────────────────────────────
# Tree Nodes
# ──────────────────────────────────────────────────────────────────

class ActionEdge:
    """Edge from DecisionNode to ChanceNode. Holds Q-value statistics."""
    __slots__ = ['action_idx', 'prior', 'visit_count', 'value_sum', 'child_chance']

    def __init__(self, action_idx: int, prior: float):
        self.action_idx = action_idx
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.child_chance: Optional[ChanceNode] = None

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def puct_score(self, parent_visits: int, c_puct: float) -> float:
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + exploration


class DecisionNode:
    """Node where the player selects an action via PUCT."""
    __slots__ = ['turn', 'edges', 'valid_actions', 'visit_count', 'is_expanded']

    def __init__(self, turn: int):
        self.turn = turn
        self.edges: Dict[int, ActionEdge] = {}
        self.valid_actions: List[Action] = []
        self.visit_count = 0
        self.is_expanded = False


class ChanceNode:
    """Node representing random card deal. Children keyed by sorted card tuple."""
    __slots__ = ['turn', 'children']

    def __init__(self, turn: int):
        self.turn = turn
        self.children: Dict[Tuple[str, ...], DecisionNode] = {}


# ──────────────────────────────────────────────────────────────────
# Main MCTS Engine
# ──────────────────────────────────────────────────────────────────

class OFC_MCTS:
    """Full multi-turn MCTS with chance nodes for OFC Pineapple."""

    def __init__(
        self,
        policy_net: torch.nn.Module,
        value_net: torch.nn.Module,
        config: OFCMCTSConfig = None,
        device: str = "cpu",
        norm_stats: Optional[dict] = None,
        t1_evaluator=None,
    ):
        self.policy_net = policy_net
        self.value_net = value_net
        self.config = config or OFCMCTSConfig()
        self.device = device
        self.t1_evaluator = t1_evaluator  # RolloutEvaluator for T1+ (if None, VN-greedy)
        self.policy_net.eval()
        self.value_net.eval()
        if norm_stats:
            self.score_mean = norm_stats.get('mean', norm_stats.get('score_mean', 0.0))
            self.score_std = norm_stats.get('std', norm_stats.get('score_std', 1.0))
        else:
            self.score_mean = 0.0
            self.score_std = 1.0

    # ──────────────────────────────────────────────────────────────
    # Entry Point
    # ──────────────────────────────────────────────────────────────

    def select_action(self, obs: Observation) -> Tuple[int, Action]:
        """Select best action. T0: MCTS search. T1+: delegate to t1_evaluator or VN-greedy."""
        if obs.turn == 0:
            return self._search_t0(obs)
        elif self.t1_evaluator is not None:
            return self.t1_evaluator.select_action(obs)
        else:
            return self._vn_greedy_t1(obs)

    # ──────────────────────────────────────────────────────────────
    # T1+: VN-Greedy with FL Filter
    # ──────────────────────────────────────────────────────────────

    def _vn_greedy_t1(self, obs: Observation) -> Tuple[int, Action]:
        """VN-greedy selection for T1+ with FL pair completion filtering."""
        valid_actions = get_turn_actions(obs.dealt_cards, obs.board_self)
        if not valid_actions:
            raise ValueError("No valid actions")
        if len(valid_actions) == 1:
            return 0, valid_actions[0]

        # FL filter
        local_actions, eval_indices = self._fl_filter_t1plus(obs, valid_actions)

        # Batch-evaluate candidates with VN
        states = []
        test_boards = []
        for action in local_actions:
            new_board = obs.board_self.copy()
            for card, pos in action.placements:
                getattr(new_board, pos).append(card)
            test_boards.append(new_board)

            discards = list(obs.known_discards_self)
            if action.discard:
                discards.append(action.discard)

            new_obs = Observation(
                board_self=new_board,
                board_opponent=obs.board_opponent,
                dealt_cards=[],
                known_discards_self=discards,
                turn=obs.turn,
                is_btn=obs.is_btn,
                is_fl=obs.is_fl,
                opp_is_fl=obs.opp_is_fl,
                chips_self=obs.chips_self,
                chips_opponent=obs.chips_opponent,
            )
            states.append(encode_state(new_obs))

        with torch.no_grad():
            states_t = torch.FloatTensor(np.array(states)).to(self.device)
            vn_out = self.value_net(states_t)
            values = vn_out['value'].squeeze(-1).cpu().numpy()
            bust_probs = vn_out['bust_prob'].squeeze(-1).cpu().numpy()

        # Denormalize + adjustments
        raw_values = values * self.score_std + self.score_mean
        raw_values -= bust_probs * self.config.bust_penalty

        fl_scale = self.config.fl_ev_scale
        for i, test_board in enumerate(test_boards):
            fl_cards = RolloutEvaluator._check_fl_cards(test_board.top)
            if fl_cards > 0:
                raw_values[i] += RolloutEvaluator.FL_EV.get(fl_cards, 0) * fl_scale
            else:
                raw_values[i] += RolloutEvaluator._fl_partial_bonus(test_board.top) * fl_scale

        best_local = int(np.argmax(raw_values))
        best_orig = eval_indices[best_local]
        return best_orig, valid_actions[best_orig]

    # ──────────────────────────────────────────────────────────────
    # T0: MCTS with Chance Nodes
    # ──────────────────────────────────────────────────────────────

    def _search_t0(self, obs: Observation) -> Tuple[int, Action]:
        """MCTS search for Turn 0 with multi-turn lookahead."""
        valid_actions = get_initial_actions(obs.dealt_cards, obs.board_self)

        if not valid_actions:
            raise ValueError("No valid T0 actions")
        if len(valid_actions) == 1:
            return 0, valid_actions[0]

        # Pre-filter for T0
        candidates = self._prefilter_t0(obs, valid_actions)
        if len(candidates) > self.config.vn_top_k:
            candidates = self._vn_prefilter(obs, candidates)
        local_actions = [valid_actions[i] for i, _ in candidates]
        orig_indices = [i for i, _ in candidates]

        # Get BC priors for candidates
        priors = self._get_priors(obs, valid_actions, orig_indices)

        # Dirichlet noise at root
        if len(local_actions) > 1:
            noise = np.random.dirichlet(
                [self.config.dirichlet_alpha] * len(local_actions)
            )
            frac = self.config.dirichlet_frac
            priors = (1 - frac) * priors + frac * noise

        # Create root DecisionNode
        root = DecisionNode(obs.turn)
        root.valid_actions = local_actions
        for i in range(len(local_actions)):
            root.edges[i] = ActionEdge(action_idx=i, prior=float(priors[i]))
        root.is_expanded = True
        root.visit_count = 0

        # Build unseen cards list (for determinization)
        seen = set(obs.board_self.all_cards())
        seen.update(obs.board_opponent.all_cards())
        seen.update(obs.dealt_cards)
        seen.update(obs.known_discards_self)
        unseen_base = [c for c in ALL_CARDS if c not in seen]

        # Run simulations
        for _ in range(self.config.num_simulations):
            self._simulate(root, obs, unseen_base)

        # Select action by visit count
        best_edge = max(root.edges.values(), key=lambda e: e.visit_count)
        best_local_idx = best_edge.action_idx
        best_orig_idx = orig_indices[best_local_idx]
        return best_orig_idx, valid_actions[best_orig_idx]

    # ──────────────────────────────────────────────────────────────
    # Simulation
    # ──────────────────────────────────────────────────────────────

    def _simulate(
        self, root: DecisionNode, obs: Observation, unseen_base: List[str]
    ):
        """One simulation: PUCT select root action → deal random T1 cards → batch-eval best.

        Since the chance node branching factor is ~C(30,3)=4060, the tree never
        develops depth beyond root. Each simulation creates a unique deal.
        So we simplify to 1-ply: evaluate T1 candidates after a random deal.
        This matches the proven old 2-ply MCTS approach.
        """
        # Determinize: shuffle unseen deck
        unseen = list(unseen_base)
        random.shuffle(unseen)

        # PUCT select at root
        best_edge = max(
            root.edges.values(),
            key=lambda e: e.puct_score(max(root.visit_count, 1), self.config.c_puct)
        )

        # Apply root action to copied board
        board = obs.board_self.copy()
        opp_board = obs.board_opponent.copy()
        discards = list(obs.known_discards_self)

        action = root.valid_actions[best_edge.action_idx]
        for card, pos in action.placements:
            getattr(board, pos).append(card)
        if action.discard:
            discards.append(action.discard)

        # Terminal check (shouldn't happen at T0, but safe)
        if board.is_complete():
            value = self._terminal_evaluate(board, opp_board)
            self._backprop([best_edge], root, value)
            return

        # Deal T1 cards (3 hero + 3 opponent)
        opp_complete = opp_board.is_complete()
        cards_needed = 3 if opp_complete else 6
        if len(unseen) < cards_needed:
            value = self._vn_evaluate(board, opp_board, discards, obs.turn, obs)
            self._backprop([best_edge], root, value)
            return

        hero_cards = unseen[:3]

        # Opponent plays T1 with BC greedy
        if not opp_complete:
            opp_cards = unseen[3:6]
            self._bc_opponent_turn(opp_board, board, opp_cards, obs.turn + 1)

        # Batch-evaluate all T1 candidate placements → return best (MAX)
        value = self._batch_evaluate_best(
            board, opp_board, hero_cards, discards, obs.turn + 1, obs)
        self._backprop([best_edge], root, value)

    def _batch_evaluate_best(
        self, board: Board, opp_board: Board,
        dealt_cards: List[str], discards: List[str],
        turn: int, root_obs: Observation,
    ) -> float:
        """Batch-evaluate all candidate actions for dealt_cards, return best score.

        Like the 2-ply MCTS: evaluates ALL candidates with VN and returns
        the optimistic (max) value, since the hero will choose the best action.
        """
        actions = get_turn_actions(dealt_cards, board)
        if not actions:
            return self._vn_evaluate(board, opp_board, discards, turn, root_obs)

        if len(actions) == 1:
            test_board = board.copy()
            for card, pos in actions[0].placements:
                getattr(test_board, pos).append(card)
            disc = list(discards)
            if actions[0].discard:
                disc.append(actions[0].discard)
            return self._vn_evaluate(test_board, opp_board, disc, turn, root_obs)

        states = []
        test_boards = []
        for action in actions:
            test_board = board.copy()
            for card, pos in action.placements:
                getattr(test_board, pos).append(card)
            test_boards.append(test_board)

            disc = list(discards)
            if action.discard:
                disc.append(action.discard)

            new_obs = Observation(
                board_self=test_board,
                board_opponent=opp_board,
                dealt_cards=[],
                known_discards_self=disc,
                turn=turn,
                is_btn=root_obs.is_btn,
                is_fl=root_obs.is_fl,
                opp_is_fl=root_obs.opp_is_fl,
                chips_self=root_obs.chips_self,
                chips_opponent=root_obs.chips_opponent,
            )
            states.append(encode_state(new_obs))

        with torch.no_grad():
            states_t = torch.FloatTensor(np.array(states)).to(self.device)
            vn_out = self.value_net(states_t)
            values = vn_out['value'].squeeze(-1).cpu().numpy()
            bust_probs = vn_out['bust_prob'].squeeze(-1).cpu().numpy()

        # Denormalize and adjust
        raw_values = values * self.score_std + self.score_mean
        raw_values -= bust_probs * self.config.bust_penalty

        fl_scale = self.config.fl_ev_scale
        for i, test_board in enumerate(test_boards):
            fl_cards = RolloutEvaluator._check_fl_cards(test_board.top)
            if fl_cards > 0:
                raw_values[i] += RolloutEvaluator.FL_EV.get(fl_cards, 0) * fl_scale
            else:
                raw_values[i] += RolloutEvaluator._fl_partial_bonus(test_board.top) * fl_scale

        # Normalize and return max (optimistic evaluation)
        normalized = np.clip(raw_values / self.config.value_scale, -1.0, 1.0)
        return float(normalized.max())

    # ──────────────────────────────────────────────────────────────
    # Opponent Modeling
    # ──────────────────────────────────────────────────────────────

    def _bc_opponent_turn(
        self, opp_board: Board, my_board: Board,
        cards: List[str], turn: int
    ):
        """Play opponent's turn using BC policy (greedy)."""
        if opp_board.is_complete():
            return
        valid_actions = get_turn_actions(cards, opp_board)
        if not valid_actions:
            return
        if len(valid_actions) == 1:
            chosen = valid_actions[0]
        else:
            obs = Observation(
                board_self=opp_board, board_opponent=my_board,
                dealt_cards=cards, known_discards_self=[],
                turn=turn, is_btn=False,
            )
            state_vec = encode_state(obs)
            state_t = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
            mask = create_action_mask(valid_actions)
            mask_t = torch.BoolTensor(mask).unsqueeze(0).to(self.device)
            with torch.no_grad():
                probs = self.policy_net(state_t, mask_t).squeeze(0).cpu().numpy()
            best_idx = int(np.argmax(probs[:len(valid_actions)]))
            chosen = valid_actions[best_idx]
        for card, pos in chosen.placements:
            getattr(opp_board, pos).append(card)

    # ──────────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────────

    def _vn_evaluate(
        self, board: Board, opp_board: Board,
        discards: List[str], turn: int, root_obs: Observation,
    ) -> float:
        """Evaluate board state with VN + FL bonus + bust penalty."""
        obs = Observation(
            board_self=board,
            board_opponent=opp_board,
            dealt_cards=[],
            known_discards_self=discards,
            turn=turn,
            is_btn=root_obs.is_btn,
            is_fl=root_obs.is_fl,
            opp_is_fl=root_obs.opp_is_fl,
            chips_self=root_obs.chips_self,
            chips_opponent=root_obs.chips_opponent,
        )
        state_vec = encode_state(obs)
        state_t = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)

        with torch.no_grad():
            vn_out = self.value_net(state_t)
            raw_value = vn_out['value'].item()
            bust_prob = vn_out['bust_prob'].item()

        # Denormalize
        score = raw_value * self.score_std + self.score_mean

        # Bust penalty
        score -= bust_prob * self.config.bust_penalty

        # FL bonus
        fl_scale = self.config.fl_ev_scale
        fl_cards = RolloutEvaluator._check_fl_cards(board.top)
        if fl_cards > 0:
            score += RolloutEvaluator.FL_EV.get(fl_cards, 0) * fl_scale
        else:
            score += RolloutEvaluator._fl_partial_bonus(board.top) * fl_scale

        # Normalize to [-1, 1]
        return max(-1.0, min(1.0, score / self.config.value_scale))

    def _terminal_evaluate(self, board: Board, opp_board: Board) -> float:
        """Evaluate a complete board with exact scoring."""
        score = RolloutEvaluator._compute_score(board, opp_board)
        return max(-1.0, min(1.0, score / self.config.value_scale))

    # ──────────────────────────────────────────────────────────────
    # Backpropagation
    # ──────────────────────────────────────────────────────────────

    def _backprop(
        self, path: List[ActionEdge], root: DecisionNode, value: float
    ):
        """Propagate value back through path edges and root."""
        for edge in path:
            edge.visit_count += 1
            edge.value_sum += value
        root.visit_count += 1

    # ──────────────────────────────────────────────────────────────
    # T0 Pre-filtering (ported from MultiTurnMCTS)
    # ──────────────────────────────────────────────────────────────

    def _prefilter_t0(
        self, obs: Observation, valid_actions: List[Action]
    ) -> List[Tuple[int, Action]]:
        """FL-aware tiered pre-filter for T0 candidates.

        Tier 1: ONLY A/K/Joker on Top → all kept (ideal FL)
        Tier 2: Nothing on Top → policy top-K
        Tier 3: Mixed cards on Top → policy top-K (fallback)
        """
        state_vec = encode_state(obs)
        state_t = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
        mask = create_action_mask(valid_actions)
        mask_t = torch.BoolTensor(mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = self.policy_net(state_t, mask_t).squeeze(0).cpu().numpy()

        tier1, tier2, tier3 = [], [], []

        for i, action in enumerate(valid_actions):
            p = probs[i] if i < len(probs) else 0.0
            top_cards = [c for c, pos in action.placements if pos == 'top']

            if not top_cards:
                tier2.append((i, action, p))
            else:
                all_fl = all(
                    c.startswith('X') or c[0] in ('A', 'K')
                    for c in top_cards
                )
                if all_fl:
                    tier1.append((i, action, p))
                else:
                    tier3.append((i, action, p))

        candidates = []
        seen = set()
        top_k = self.config.top_k_filter

        for i, action, p in tier1:
            seen.add(i)
            candidates.append((i, action))

        tier2.sort(key=lambda x: x[2], reverse=True)
        for i, action, p in tier2[:top_k]:
            if i not in seen:
                seen.add(i)
                candidates.append((i, action))

        if len(candidates) < top_k:
            tier3.sort(key=lambda x: x[2], reverse=True)
            remaining = top_k - len(candidates)
            for i, action, p in tier3[:remaining]:
                if i not in seen:
                    seen.add(i)
                    candidates.append((i, action))

        return candidates if candidates else [(0, valid_actions[0])]

    def _vn_prefilter(
        self, obs: Observation, candidates: List[Tuple[int, Action]]
    ) -> List[Tuple[int, Action]]:
        """Score candidates with VN, keep top vn_top_k. FL-aware protection."""
        # Identify FL-promising candidates to protect
        fl_protected = []
        fl_protected_indices = set()
        max_protect = 2

        for j, (orig_idx, action) in enumerate(candidates):
            if len(fl_protected) >= max_protect:
                break
            for card, pos in action.placements:
                if pos == 'top':
                    if card.startswith('X') or card[0] in ('A', 'K'):
                        fl_protected.append((orig_idx, action))
                        fl_protected_indices.add(j)
                        break

        # VN-score remaining
        remaining = [
            (j, c) for j, c in enumerate(candidates)
            if j not in fl_protected_indices
        ]
        if not remaining:
            return fl_protected[:self.config.vn_top_k]

        state_batch = []
        remaining_boards = []
        for _, (orig_idx, action) in remaining:
            new_board = obs.board_self.copy()
            for card, pos in action.placements:
                getattr(new_board, pos).append(card)
            remaining_boards.append(new_board)

            discards = list(obs.known_discards_self)
            if action.discard:
                discards.append(action.discard)

            new_obs = Observation(
                board_self=new_board,
                board_opponent=obs.board_opponent,
                dealt_cards=[],
                known_discards_self=discards,
                turn=obs.turn,
                is_btn=obs.is_btn,
                is_fl=obs.is_fl,
                opp_is_fl=obs.opp_is_fl,
                chips_self=obs.chips_self,
                chips_opponent=obs.chips_opponent,
            )
            state_batch.append(encode_state(new_obs))

        with torch.no_grad():
            states_t = torch.FloatTensor(np.array(state_batch)).to(self.device)
            vn_out = self.value_net(states_t)
            vn_scores = vn_out['value'].squeeze(-1).cpu().numpy()
            bust_probs = vn_out['bust_prob'].squeeze(-1).cpu().numpy()

        raw_scores = vn_scores * self.score_std + self.score_mean
        raw_scores -= bust_probs * self.config.bust_penalty
        fl_scale = self.config.fl_ev_scale
        for k_idx in range(len(remaining)):
            board_k = remaining_boards[k_idx]
            fl_cards = RolloutEvaluator._check_fl_cards(board_k.top)
            if fl_cards > 0:
                raw_scores[k_idx] += RolloutEvaluator.FL_EV.get(fl_cards, 0) * fl_scale
            else:
                raw_scores[k_idx] += RolloutEvaluator._fl_partial_bonus(board_k.top) * fl_scale

        scored = [(raw_scores[k], remaining[k][1]) for k in range(len(remaining))]
        scored.sort(key=lambda x: x[0], reverse=True)

        vn_budget = self.config.vn_top_k - len(fl_protected)
        vn_top = [cand for _, cand in scored[:max(vn_budget, 1)]]

        result = fl_protected + vn_top
        return result[:self.config.vn_top_k]

    # ──────────────────────────────────────────────────────────────
    # T1+ FL Filtering
    # ──────────────────────────────────────────────────────────────

    def _fl_filter_t1plus(
        self, obs: Observation, valid_actions: List[Action]
    ) -> Tuple[List[Action], List[int]]:
        """FL pair completion / top protection filter for T1+.

        Returns (local_actions, orig_indices).
        """
        filtered_indices = None

        if obs.turn <= 7 and len(obs.board_self.top) < 3:
            top_ranks = set()
            for c in obs.board_self.top:
                if c.startswith('X'):
                    top_ranks.update(['A', 'K'])
                elif c[0] in ('A', 'K'):
                    top_ranks.add(c[0])

            if top_ranks:
                dealt_ranks = [c[0] for c in obs.dealt_cards if not c.startswith('X')]
                dealt_jokers = any(c.startswith('X') for c in obs.dealt_cards)
                matching = top_ranks & set(dealt_ranks)

                if matching or dealt_jokers:
                    # Force pair completion
                    pair_cands = []
                    for i, action in enumerate(valid_actions):
                        for card, pos in action.placements:
                            if pos == 'top' and (card[0] in top_ranks or card.startswith('X')):
                                pair_cands.append(i)
                                break
                    if pair_cands:
                        filtered_indices = pair_cands
                else:
                    # Protect top
                    protect_cands = []
                    for i, action in enumerate(valid_actions):
                        places_on_top = any(pos == 'top' for _, pos in action.placements)
                        if not places_on_top:
                            protect_cands.append(i)
                    if protect_cands:
                        filtered_indices = protect_cands

        if filtered_indices is None:
            filtered_indices = list(range(len(valid_actions)))

        local_actions = [valid_actions[i] for i in filtered_indices]
        return local_actions, filtered_indices

    # ──────────────────────────────────────────────────────────────
    # BC Prior Computation
    # ──────────────────────────────────────────────────────────────

    def _get_priors(
        self, obs: Observation, valid_actions: List[Action],
        candidate_indices: List[int]
    ) -> np.ndarray:
        """Get BC policy priors for candidate subset, re-normalized."""
        state_vec = encode_state(obs)
        state_t = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
        mask = create_action_mask(valid_actions)
        mask_t = torch.BoolTensor(mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = self.policy_net(state_t, mask_t).squeeze(0).cpu().numpy()

        priors = np.array([
            probs[i] if i < len(probs) else 0.0 for i in candidate_indices
        ])
        total = priors.sum()
        if total > 0:
            priors = priors / total
        else:
            priors = np.ones(len(candidate_indices)) / len(candidate_indices)

        return priors
