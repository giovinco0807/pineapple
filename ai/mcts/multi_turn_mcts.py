"""
OFC Pineapple - Multi-Turn MCTS Engine (Step 1: 2-Turn Lookahead)

IS-MCTS with determinization for T0 decisions.
Each simulation:
  1. UCT-select T0 child node
  2. Determinize: sample unseen cards for T1
  3. Opponent plays T1 with BC policy
  4. Batch-evaluate T1 action candidates with Value Network
  5. Backpropagate best T1 value to T0 node

For T1+ turns, uses VN-greedy evaluation (batch all candidates, pick best).

Performance estimate (T0, 400 simulations):
  - Per simulation: ~1ms (UCT + determinize + 1 BC forward + 1 VN batch)
  - Total T0 decision: ~400ms (vs ~15s for AllRollout r=400)
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
from ai.mcts.mcts import MCTSNode


@dataclass
class MultiTurnConfig:
    """Configuration for multi-turn MCTS."""
    num_simulations: int = 400    # MCTS simulations for T0
    c_puct: float = 1.5           # UCB exploration constant
    dirichlet_alpha: float = 0.3  # Root exploration noise
    dirichlet_frac: float = 0.25  # Fraction of noise to mix
    top_k: int = 20               # Policy-net pre-filter for T0
    vn_top_k: int = 12            # VN pre-filter keeps top-K
    bust_penalty: float = 0.0     # bust_prob * penalty subtracted from VN score
    fl_ev_scale: float = 1.0      # multiplier on FL EV chain values


class MultiTurnMCTS:
    """
    IS-MCTS with 2-turn lookahead for OFC Pineapple.

    T0: MCTS with UCT, each leaf evaluated by 2-ply lookahead:
      Apply T0 action → sample T1 cards → opponent BC T1 → best VN(T1)

    T1+: VN-greedy with FL pair completion filtering.
    """

    def __init__(
        self,
        policy_net: torch.nn.Module,
        value_net: torch.nn.Module,
        config: MultiTurnConfig = None,
        device: str = "cpu",
        norm_stats: Optional[dict] = None,
    ):
        self.policy_net = policy_net
        self.value_net = value_net
        self.config = config or MultiTurnConfig()
        self.device = device
        self.policy_net.eval()
        self.value_net.eval()
        self.score_mean = norm_stats.get('score_mean', 0.0) if norm_stats else 0.0
        self.score_std = norm_stats.get('score_std', 1.0) if norm_stats else 1.0

    def select_action(self, obs: Observation) -> Tuple[int, Action]:
        """Select best action. T0: 2-ply MCTS, T1+: VN greedy."""
        if obs.turn == 0:
            return self._search_t0(obs)
        else:
            return self._vn_greedy(obs)

    # ──────────────────────────────────────────────────────────────────
    # T0: 2-Turn Lookahead MCTS
    # ──────────────────────────────────────────────────────────────────

    def _search_t0(self, obs: Observation) -> Tuple[int, Action]:
        """MCTS with 2-turn lookahead for Turn 0."""
        valid_actions = get_initial_actions(obs.dealt_cards, obs.board_self)
        if not valid_actions:
            raise ValueError("No valid T0 actions")
        if len(valid_actions) == 1:
            return 0, valid_actions[0]

        # Pre-filter: FL rules + policy net top-K
        candidates = self._prefilter_t0(obs, valid_actions)

        # VN pre-filter: further reduce candidates
        if len(candidates) > self.config.vn_top_k:
            candidates = self._vn_prefilter(obs, candidates)

        # Map: local index -> original index in valid_actions
        local_actions = [valid_actions[i] for i, _ in candidates]
        orig_indices = [i for i, _ in candidates]

        # Policy priors for candidates (using full valid_actions for correct indexing)
        priors = self._get_priors(obs, valid_actions, orig_indices)

        # Add Dirichlet noise at root for exploration
        if len(local_actions) > 1:
            noise = np.random.dirichlet(
                [self.config.dirichlet_alpha] * len(local_actions)
            )
            frac = self.config.dirichlet_frac
            priors = (1 - frac) * priors + frac * noise

        # Create and expand root node
        root = MCTSNode(state=obs)
        root.valid_actions = local_actions
        for i in range(len(local_actions)):
            child = MCTSNode(parent=root, action_idx=i, prior=float(priors[i]))
            root.children[i] = child
        root.is_expanded = True

        # Run MCTS simulations
        for _ in range(self.config.num_simulations):
            # UCT select child
            child = max(
                root.children.values(),
                key=lambda c: c.ucb_score(root.visit_count, self.config.c_puct)
            )

            # 2-ply evaluation with fresh determinization
            value = self._evaluate_2ply(obs, local_actions[child.action_idx])

            # Backpropagate
            node = child
            while node is not None:
                node.visit_count += 1
                node.value_sum += value
                node = node.parent

        # Select action by visit count (greedy)
        best_child = max(root.children.values(), key=lambda c: c.visit_count)
        best_local_idx = best_child.action_idx
        best_orig_idx = orig_indices[best_local_idx]

        return best_orig_idx, valid_actions[best_orig_idx]

    def _evaluate_2ply(self, root_obs: Observation, t0_action: Action) -> float:
        """
        Evaluate a T0 action by looking 1 turn ahead.

        1. Apply T0 action to self board
        2. Sample unseen cards → deal 3 to self + 3 to opponent for T1
        3. Opponent plays T1 with BC policy (greedy)
        4. Batch-evaluate all T1 candidates with VN
        5. Return max VN value (best T1 outcome) normalized to [-1, 1]
        """
        # 1. Apply T0 action
        board = root_obs.board_self.copy()
        for card, pos in t0_action.placements:
            getattr(board, pos).append(card)

        opp_board = root_obs.board_opponent.copy()

        # 2. Build unseen cards and sample T1 deal
        seen = set(board.all_cards())
        seen.update(opp_board.all_cards())
        seen.update(root_obs.known_discards_self)
        if t0_action.discard:
            seen.add(t0_action.discard)
        unseen = [c for c in ALL_CARDS if c not in seen]
        random.shuffle(unseen)

        if len(unseen) < 6:
            return self._vn_evaluate_single(board, opp_board, root_obs, turn=1)

        my_t1_cards = unseen[:3]
        opp_t1_cards = unseen[3:6]

        # 3. Opponent plays T1 with BC policy
        self._bc_opponent_turn(opp_board, board, opp_t1_cards, turn=1)

        # 4. Generate my T1 action candidates
        t1_actions = get_turn_actions(my_t1_cards, board)
        if not t1_actions:
            return self._vn_evaluate_single(board, opp_board, root_obs, turn=1)

        # 5. Batch-evaluate T1 candidates with VN + FL bonus
        from ai.mcts.rollout_evaluator import RolloutEvaluator
        discards_base = list(root_obs.known_discards_self)

        states = []
        t1_boards = []  # Track boards for FL bonus calculation
        for action in t1_actions:
            test_board = board.copy()
            for card, pos in action.placements:
                getattr(test_board, pos).append(card)
            t1_boards.append(test_board)

            discards = list(discards_base)
            if action.discard:
                discards.append(action.discard)

            new_obs = Observation(
                board_self=test_board,
                board_opponent=opp_board,
                dealt_cards=[],
                known_discards_self=discards,
                turn=1,
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

        # Denormalize to raw score
        raw_values = values * self.score_std + self.score_mean

        # Bust penalty: high bust_prob → penalize score
        # bust_cost ≈ 6 (scoop) + opp_avg_royalty
        raw_values -= bust_probs * self.config.bust_penalty

        # Add FL bonus: VN undervalues FL routes (84% non-FL training data),
        # so we add explicit FL partial bonus from rollout evaluator's chain EV
        fl_scale = self.config.fl_ev_scale
        for i, test_board in enumerate(t1_boards):
            fl_cards = RolloutEvaluator._check_fl_cards(test_board.top)
            if fl_cards > 0:
                raw_values[i] += RolloutEvaluator.FL_EV.get(fl_cards, 0) * fl_scale
            else:
                raw_values[i] += RolloutEvaluator._fl_partial_bonus(test_board.top) * fl_scale

        # Normalize to [-1, 1]
        normalized = np.clip(raw_values / 30.0, -1.0, 1.0)

        return float(normalized.max())

    # ──────────────────────────────────────────────────────────────────
    # Opponent Modeling
    # ──────────────────────────────────────────────────────────────────

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
                board_self=opp_board,
                board_opponent=my_board,
                dealt_cards=cards,
                known_discards_self=[],
                turn=turn,
                is_btn=False,
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

    # ──────────────────────────────────────────────────────────────────
    # VN Evaluation
    # ──────────────────────────────────────────────────────────────────

    def _vn_evaluate_single(
        self, board: Board, opp_board: Board,
        root_obs: Observation, turn: int,
        discard: Optional[str] = None,
    ) -> float:
        """Single-state VN evaluation, normalized to [-1, 1]."""
        discards = list(root_obs.known_discards_self)
        if discard:
            discards.append(discard)

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
            raw = self.value_net(state_t)['value'].item()

        raw_score = raw * self.score_std + self.score_mean
        return max(-1.0, min(1.0, raw_score / 30.0))

    # ──────────────────────────────────────────────────────────────────
    # Pre-filtering
    # ──────────────────────────────────────────────────────────────────

    def _prefilter_t0(
        self, obs: Observation, valid_actions: List[Action]
    ) -> List[Tuple[int, Action]]:
        """FL-aware tiered pre-filter for T0 candidates.

        Three tiers (same logic as RolloutEvaluator._filter_top_k):
          Tier 1: ONLY A/K/Joker on Top → all kept (ideal FL, preserves slots)
          Tier 2: Nothing on Top → policy top-K (safe non-FL)
          Tier 3: Mixed cards on Top → policy top-K (fallback)
        """
        state_vec = encode_state(obs)
        state_t = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
        mask = create_action_mask(valid_actions)
        mask_t = torch.BoolTensor(mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = self.policy_net(state_t, mask_t).squeeze(0).cpu().numpy()

        tier1 = []  # Only A/K/Joker on Top (FL-ideal)
        tier2 = []  # Nothing on Top (safe)
        tier3 = []  # Others

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

        # Build candidate list: all tier1, top-K from tier2, fill with tier3
        candidates = []
        seen = set()

        for i, action, p in tier1:
            seen.add(i)
            candidates.append((i, action))

        tier2.sort(key=lambda x: x[2], reverse=True)
        for i, action, p in tier2[:self.config.top_k]:
            if i not in seen:
                seen.add(i)
                candidates.append((i, action))

        if len(candidates) < self.config.top_k:
            tier3.sort(key=lambda x: x[2], reverse=True)
            remaining = self.config.top_k - len(candidates)
            for i, action, p in tier3[:remaining]:
                if i not in seen:
                    seen.add(i)
                    candidates.append((i, action))

        return candidates if candidates else [(0, valid_actions[0])]

    def _policy_topk(
        self, obs: Observation, valid_actions: List[Action],
        candidates: List[Tuple[int, Action]], k: int
    ) -> List[Tuple[int, Action]]:
        """Filter candidates by policy-net probability, keep top-K."""
        state_vec = encode_state(obs)
        state_t = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
        mask = create_action_mask(valid_actions)
        mask_t = torch.BoolTensor(mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = self.policy_net(state_t, mask_t).squeeze(0).cpu().numpy()

        scored = [
            (probs[i] if i < len(probs) else 0.0, (i, a))
            for i, a in candidates
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [cand for _, cand in scored[:k]]

    def _vn_prefilter(
        self, obs: Observation, candidates: List[Tuple[int, Action]]
    ) -> List[Tuple[int, Action]]:
        """Score candidates with VN and keep top vn_top_k.

        FL-aware: protects up to 2 FL-promising candidates (A/K on top)
        from being dropped by VN scoring.
        """
        # Identify FL-promising candidates to protect
        fl_protected = []
        fl_protected_indices = set()
        max_protect = 2

        for j, (orig_idx, action) in enumerate(candidates):
            if len(fl_protected) >= max_protect:
                break
            for card, pos in action.placements:
                if pos == 'top':
                    if (not card.startswith('X') and card[0] in ('A', 'K')) or \
                       card.startswith('X'):
                        fl_protected.append((orig_idx, action))
                        fl_protected_indices.add(j)
                        break

        # VN-score remaining candidates
        remaining = [
            (j, c) for j, c in enumerate(candidates)
            if j not in fl_protected_indices
        ]
        if not remaining:
            return fl_protected[:self.config.vn_top_k]

        from ai.mcts.rollout_evaluator import RolloutEvaluator

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

        # Denormalize, apply bust penalty, and add FL bonus
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

        # Sort remaining by FL-augmented VN score
        scored = [(raw_scores[k], remaining[k][1]) for k in range(len(remaining))]
        scored.sort(key=lambda x: x[0], reverse=True)

        # Combine: FL protected + VN top picks
        vn_budget = self.config.vn_top_k - len(fl_protected)
        vn_top = [cand for _, cand in scored[:max(vn_budget, 1)]]

        result = fl_protected + vn_top
        return result[:self.config.vn_top_k]

    def _get_priors(
        self, obs: Observation, valid_actions: List[Action],
        candidate_indices: List[int]
    ) -> np.ndarray:
        """Get policy-net priors for candidate subset of valid_actions.

        Runs PolicyNet on full valid_actions (for correct positional indexing),
        then extracts and re-normalizes priors for the candidate subset.
        """
        state_vec = encode_state(obs)
        state_t = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
        mask = create_action_mask(valid_actions)
        mask_t = torch.BoolTensor(mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = self.policy_net(state_t, mask_t).squeeze(0).cpu().numpy()

        # Extract priors for candidate indices
        priors = np.array([
            probs[i] if i < len(probs) else 0.0 for i in candidate_indices
        ])
        total = priors.sum()
        if total > 0:
            priors = priors / total
        else:
            priors = np.ones(len(candidate_indices)) / len(candidate_indices)

        return priors

    # ──────────────────────────────────────────────────────────────────
    # T1+: VN Greedy with FL Pair Completion
    # ──────────────────────────────────────────────────────────────────

    def _vn_greedy(self, obs: Observation) -> Tuple[int, Action]:
        """VN-only greedy for T1+ with FL pair completion filtering.

        If top has A/K and dealt cards match → force pair completion.
        If top has A/K but no match → protect top (don't place there).
        Otherwise → VN picks best among all candidates.
        """
        valid_actions = get_turn_actions(obs.dealt_cards, obs.board_self)
        if not valid_actions:
            raise ValueError("No valid actions")
        if len(valid_actions) == 1:
            return 0, valid_actions[0]

        # FL pair completion filter for T1+
        eval_indices = list(range(len(valid_actions)))
        if obs.turn <= 7 and len(obs.board_self.top) < 3:
            fl_indices = self._fl_filter_t1(obs, valid_actions)
            if fl_indices:
                eval_indices = fl_indices

        # Batch-evaluate candidates with VN
        states = []
        for i in eval_indices:
            action = valid_actions[i]
            new_board = obs.board_self.copy()
            for card, pos in action.placements:
                getattr(new_board, pos).append(card)

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
            values = self.value_net(states_t)['value'].squeeze(-1).cpu().numpy()

        best_local = int(np.argmax(values))
        best_orig = eval_indices[best_local]
        return best_orig, valid_actions[best_orig]

    @staticmethod
    def _fl_filter_t1(
        obs: Observation, valid_actions: List[Action]
    ) -> Optional[List[int]]:
        """FL pair completion filter for T1+.

        Returns filtered action indices, or None if no FL filtering needed.
        """
        top_ranks = set()
        for c in obs.board_self.top:
            if c.startswith('X'):
                top_ranks.update(['A', 'K'])
            elif c[0] in ('A', 'K'):
                top_ranks.add(c[0])

        if not top_ranks:
            return None

        # Check if dealt cards can complete the pair
        dealt_ranks = [c[0] for c in obs.dealt_cards if not c.startswith('X')]
        dealt_jokers = any(c.startswith('X') for c in obs.dealt_cards)
        matching = top_ranks & set(dealt_ranks)

        if matching or dealt_jokers:
            # Force pair completion: only actions placing matching rank on top
            pair_cands = []
            for i, action in enumerate(valid_actions):
                for card, pos in action.placements:
                    if pos == 'top' and (card[0] in top_ranks or card.startswith('X')):
                        pair_cands.append(i)
                        break
            if pair_cands:
                return pair_cands
        else:
            # Protect top: avoid placing non-matching cards there
            protect_cands = []
            for i, action in enumerate(valid_actions):
                places_on_top = any(pos == 'top' for _, pos in action.placements)
                if not places_on_top:
                    protect_cands.append(i)
            if protect_cands:
                return protect_cands

        return None
