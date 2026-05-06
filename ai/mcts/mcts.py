"""
OFC Pineapple - Monte Carlo Tree Search Engine

IS-MCTS with Progressive Widening for imperfect information OFC.
Uses BC-trained PolicyNet for priors and ValueNet for leaf evaluation.
"""
import sys
import math
import random
import copy
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
from ai.engine.game_engine import GameEngine, Hand


# ---------------------------------------------------------------------------
# MCTSNode
# ---------------------------------------------------------------------------
class MCTSNode:
    """A single node in the MCTS tree."""

    __slots__ = [
        "node_type", "player", "visits", "value_sum",
        "children", "valid_actions", "priors", "is_expanded"
    ]

    def __init__(self, node_type: str, player: int = -1):
        self.node_type = node_type  # "DECISION" or "CHANCE"
        self.player = player        # 0 or 1 (only for DECISION)
        self.visits = 0
        self.value_sum = 0.0        # Always from perspective of Player 0
        self.children = {}          # Action -> MCTSNode (for DECISION) or tuple(cards) -> MCTSNode (for CHANCE)
        self.valid_actions = []     # List[Action]
        self.priors = {}            # Action -> float
        self.is_expanded = False

    @property
    def q_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def ucb_score(self, parent_visits: int, prior: float, c_puct: float = 1.5, invert: bool = False) -> float:
        exploration = c_puct * prior * math.sqrt(parent_visits) / (1 + self.visits)
        q = self.q_value
        if invert:
            q = -q
        return q + exploration


# ---------------------------------------------------------------------------
# MCTS Config
# ---------------------------------------------------------------------------
@dataclass
class MCTSConfig:
    num_simulations: int = 800
    c_puct: float = 1.5
    temperature: float = 1.0       # Action selection temperature
    progressive_widening_c: float = 2.5
    progressive_widening_alpha: float = 0.5
    max_children: int = 100        # Cap for progressive widening
    dirichlet_alpha: float = 0.3   # Root exploration noise
    dirichlet_frac: float = 0.25   # Fraction of noise to mix in


# ---------------------------------------------------------------------------
# MCTS Search
# ---------------------------------------------------------------------------
class MCTS:
    """
    IS-MCTS with Progressive Widening for OFC Pineapple.
    """

    def __init__(
        self,
        policy_net: torch.nn.Module,
        value_net: torch.nn.Module,
        config: MCTSConfig = MCTSConfig(),
        device: str = "cpu",
    ):
        self.policy_net = policy_net
        self.value_net = value_net
        self.config = config
        self.device = device
        self.policy_net.eval()
        self.value_net.eval()

    def search(
        self,
        obs: Observation,
        board_state: dict,
    ) -> Tuple[int, Dict[int, float], List[Action]]:
        """
        Run MCTS from the given observation.
        """
        if obs.turn == 0:
            valid_actions = get_initial_actions(obs.dealt_cards, obs.board_self)
        else:
            valid_actions = get_turn_actions(obs.dealt_cards, obs.board_self)

        if not valid_actions:
            return 0, {0: 1.0}, []

        if len(valid_actions) == 1:
            return 0, {0: 1.0}, valid_actions

        # Root is a DecisionNode for Player 0 (Hero)
        root = MCTSNode(node_type="DECISION", player=0)
        
        # Expand root immediately
        self._expand_node_from_obs(root, obs, valid_actions)
        
        # Add Dirichlet noise at root
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(valid_actions))
        frac = self.config.dirichlet_frac
        for i, a in enumerate(valid_actions):
            root.priors[a] = (1 - frac) * root.priors[a] + frac * noise[i]

        for _ in range(self.config.num_simulations):
            sim_state = self._build_determinized_state(obs)
            self._traverse(root, sim_state)

        # Extract action probabilities from visit counts
        action_probs = self._get_action_probs(root)
        best_action_idx = self._select_action(action_probs)

        return best_action_idx, action_probs, root.valid_actions

    def _build_determinized_state(self, obs: Observation) -> Hand:
        """Create a consistent full game state (Hand) from the Observation."""
        sim_state = Hand.__new__(Hand)
        sim_state.btn = 0 if obs.is_btn else 1
        sim_state.boards = [obs.board_self.copy(), obs.board_opponent.copy()]
        sim_state.dealt_cards = [list(obs.dealt_cards), []]
        sim_state.discards = [list(obs.known_discards_self), []]
        sim_state.turn = obs.turn
        sim_state.placed = [False, False]

        # If Hero (0) is not btn, Villain (1) acted first in the current turn
        if sim_state.btn == 1:
            sim_state.placed[1] = True

        hero_cards = set(obs.board_self.top + obs.board_self.middle + obs.board_self.bottom + obs.dealt_cards + obs.known_discards_self)
        villain_cards = set(obs.board_opponent.top + obs.board_opponent.middle + obs.board_opponent.bottom)
        avail = list(set(ALL_CARDS) - hero_cards - villain_cards)
        random.shuffle(avail)

        num_villain_discards = obs.turn if sim_state.btn == 1 else max(0, obs.turn - 1)
        sim_state.discards[1] = avail[:num_villain_discards]
        avail = avail[num_villain_discards:]

        # If Villain hasn't placed this turn, they need dealt cards
        if not sim_state.placed[1]:
            num_deal = 5 if obs.turn == 0 else 3
            sim_state.dealt_cards[1] = avail[:num_deal]
            avail = avail[num_deal:]

        sim_state.deck = avail
        return sim_state

    def _traverse(self, node: MCTSNode, sim_state: Hand) -> float:
        """Traverse the tree, expanding and evaluating as needed."""
        if sim_state.is_hand_complete():
            result = GameEngine.compute_result(sim_state)
            # Normalised score
            val = result.raw_score[0] / 20.0
            return max(-1.0, min(1.0, val))

        if sim_state.is_turn_complete():
            sim_state.deal_next_turn()
            if node.node_type != "CHANCE":
                raise ValueError("Expected CHANCE node")
            
            c0 = tuple(sorted(sim_state.dealt_cards[0]))
            if c0 not in node.children:
                node.children[c0] = MCTSNode(node_type="DECISION", player=sim_state.btn)
            
            next_node = node.children[c0]
            v = self._traverse(next_node, sim_state)
            node.visits += 1
            node.value_sum += v
            return v

        player = sim_state.btn if not sim_state.placed[sim_state.btn] else 1 - sim_state.btn
        assert node.node_type == "DECISION" and node.player == player

        if not node.is_expanded:
            valid_actions = get_turn_actions(sim_state.dealt_cards[player], sim_state.boards[player]) if sim_state.turn > 0 else get_initial_actions(sim_state.dealt_cards[player], sim_state.boards[player])
            obs = sim_state.get_observation(player)
            self._expand_node_from_obs(node, obs, valid_actions)
            v = self._evaluate_network(sim_state)
            node.visits += 1
            node.value_sum += v
            return v

        best_action = self._select_action_ucb(node)
        sim_state.apply_action(player, best_action)

        if best_action not in node.children:
            if sim_state.is_turn_complete():
                node.children[best_action] = MCTSNode(node_type="CHANCE")
            else:
                next_p = sim_state.btn if not sim_state.placed[sim_state.btn] else 1 - sim_state.btn
                node.children[best_action] = MCTSNode(node_type="DECISION", player=next_p)

        next_node = node.children[best_action]
        v = self._traverse(next_node, sim_state)

        node.visits += 1
        node.value_sum += v
        return v

    def _expand_node_from_obs(self, node: MCTSNode, obs: Observation, valid_actions: List[Action]):
        state_vec = encode_state(obs)
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
        mask = create_action_mask(valid_actions)
        mask_tensor = torch.BoolTensor(mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = self.policy_net(state_tensor, mask_tensor).squeeze(0).cpu().numpy()

        priors = probs[:len(valid_actions)]
        total = priors.sum()
        if total > 0:
            priors = priors / total
        else:
            priors = np.ones(len(valid_actions)) / len(valid_actions)

        node.valid_actions = valid_actions
        node.priors = {a: p for a, p in zip(valid_actions, priors)}
        node.is_expanded = True

    def _evaluate_network(self, sim_state: Hand) -> float:
        obs0 = sim_state.get_observation(0)
        vec0 = encode_state(obs0)
        obs1 = sim_state.get_observation(1)
        vec1 = encode_state(obs1)

        tensor = torch.FloatTensor(np.array([vec0, vec1])).to(self.device)
        with torch.no_grad():
            pred = self.value_net(tensor)

        royalty = pred["royalty_ev"].cpu().numpy()
        bust = pred["bust_prob"].cpu().numpy()
        fl = pred["fl_prob"].cpu().numpy()

        val0 = royalty[0] - bust[0] * 11.0 + fl[0] * 8.0
        val1 = royalty[1] - bust[1] * 11.0 + fl[1] * 8.0
        
        value = val0 - val1
        return max(-1.0, min(1.0, value / 20.0))

    def _select_action_ucb(self, node: MCTSNode) -> Action:
        k = min(
            len(node.valid_actions),
            max(1, int(math.ceil(
                self.config.progressive_widening_c *
                (node.visits ** self.config.progressive_widening_alpha)
            )))
        )
        k = min(k, self.config.max_children)

        # Sort valid actions by prior to select top-k for progressive widening
        candidates = sorted(node.valid_actions, key=lambda a: node.priors[a], reverse=True)[:k]
        
        invert = (node.player == 1) # P1 minimizes P0's value
        
        best_score = -float('inf')
        best_action = candidates[0]
        for a in candidates:
            child = node.children.get(a)
            if child is None:
                # Unexplored action has infinite UCB
                return a
            
            score = child.ucb_score(node.visits, node.priors[a], self.config.c_puct, invert)
            if score > best_score:
                best_score = score
                best_action = a
                
        return best_action

    def _get_action_probs(self, root: MCTSNode) -> Dict[int, float]:
        visits = {i: 0 for i in range(len(root.valid_actions))}
        for i, a in enumerate(root.valid_actions):
            if a in root.children:
                visits[i] = root.children[a].visits

        total = sum(visits.values())
        if total == 0:
            n = len(visits)
            return {idx: 1.0 / n for idx in visits}

        if self.config.temperature == 0:
            best = max(visits, key=visits.get)
            return {idx: (1.0 if idx == best else 0.0) for idx in visits}

        temp = self.config.temperature
        scaled = {idx: (count ** (1.0 / temp)) for idx, count in visits.items()}
        total_scaled = sum(scaled.values())
        return {idx: v / total_scaled for idx, v in scaled.items()}

    def _select_action(self, action_probs: Dict[int, float]) -> int:
        indices = list(action_probs.keys())
        probs = [action_probs[i] for i in indices]
        return random.choices(indices, weights=probs, k=1)[0]
