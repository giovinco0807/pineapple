"""
OFC Pineapple - Rollout-based Action Evaluator

Replaces MCTS. For each candidate action, simulates N complete games
using PolicyNet for both players, and picks the action with the highest
average final score.

Usage:
    evaluator = RolloutEvaluator(policy_net, device="cuda")
    best_idx, best_action = evaluator.select_action(obs)
"""
import sys
import random
import copy
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.encoding import (
    Board, Observation, encode_state, ALL_CARDS
)
from ai.engine.action_space import (
    get_initial_actions, get_turn_actions, create_action_mask, Action
)
from ai.engine.game_engine import (
    GameEngine, evaluate_hand, get_top_royalty,
    get_middle_royalty, get_bottom_royalty
)


class RolloutEvaluator:
    """Evaluate actions by Monte Carlo rollouts with PolicyNet guidance."""

    def __init__(
        self,
        policy_net: torch.nn.Module,
        n_rollouts: int = 200,
        top_k: int = 20,
        device: str = "cpu",
    ):
        self.policy_net = policy_net
        self.n_rollouts = n_rollouts
        self.top_k = top_k
        self.device = device
        self.policy_net.eval()

    def select_action(
        self, obs: Observation
    ) -> Tuple[int, Action]:
        """Select the best action by rollout evaluation.

        Returns:
            (best_index, best_action) among valid_actions
        """
        if obs.turn == 0:
            valid_actions = get_initial_actions(obs.dealt_cards, obs.board_self)
        else:
            valid_actions = get_turn_actions(obs.dealt_cards, obs.board_self)

        if not valid_actions:
            raise ValueError("No valid actions")
        if len(valid_actions) == 1:
            return 0, valid_actions[0]

        # For Turn 0: filter to top-K by policy prior
        if obs.turn == 0 and len(valid_actions) > self.top_k:
            candidates = self._filter_top_k(obs, valid_actions)
        else:
            candidates = list(enumerate(valid_actions))

        # Evaluate each candidate by N rollouts
        best_idx = -1
        best_score = float("-inf")

        for orig_idx, action in candidates:
            avg = self._evaluate_action(obs, action)
            if avg > best_score:
                best_score = avg
                best_idx = orig_idx

        return best_idx, valid_actions[best_idx]

    def select_action_with_scores(
        self, obs: Observation
    ) -> Tuple[int, Action, List[float]]:
        """Like select_action but also returns per-action average scores.

        Used by self-play training to build soft targets.
        """
        if obs.turn == 0:
            valid_actions = get_initial_actions(obs.dealt_cards, obs.board_self)
        else:
            valid_actions = get_turn_actions(obs.dealt_cards, obs.board_self)

        if not valid_actions:
            raise ValueError("No valid actions")
        if len(valid_actions) == 1:
            return 0, valid_actions[0], [0.0]

        if obs.turn == 0 and len(valid_actions) > self.top_k:
            candidates = self._filter_top_k(obs, valid_actions)
        else:
            candidates = list(enumerate(valid_actions))

        scores = [float("-inf")] * len(valid_actions)
        best_idx = -1
        best_score = float("-inf")

        for orig_idx, action in candidates:
            avg = self._evaluate_action(obs, action)
            scores[orig_idx] = avg
            if avg > best_score:
                best_score = avg
                best_idx = orig_idx

        return best_idx, valid_actions[best_idx], scores

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _filter_top_k(
        self, obs: Observation, valid_actions: List[Action]
    ) -> List[Tuple[int, Action]]:
        """Use PolicyNet to keep only top-K most promising actions."""
        state_vec = encode_state(obs)
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
        mask = create_action_mask(valid_actions)
        mask_tensor = torch.BoolTensor(mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = self.policy_net(state_tensor, mask_tensor).squeeze(0).cpu().numpy()

        # Get indices of top-K actions by policy probability
        n = min(len(valid_actions), len(probs))
        top_indices = np.argsort(probs[:n])[-self.top_k :][::-1]

        return [(int(i), valid_actions[i]) for i in top_indices]

    def _evaluate_action(self, obs: Observation, action: Action) -> float:
        """Run N rollouts for one action and return average score."""
        total = 0.0

        for _ in range(self.n_rollouts):
            score = self._single_rollout(obs, action)
            total += score

        return total / self.n_rollouts

    def _single_rollout(self, obs: Observation, action: Action) -> float:
        """One complete rollout: apply action, then play out the rest."""
        # Apply the candidate action to my board
        my_board = obs.board_self.copy()
        for card, pos in action.placements:
            getattr(my_board, pos).append(card)

        # Opponent board (as seen by this player)
        opp_board = obs.board_opponent.copy()

        # Build unseen card pool
        seen = set()
        seen.update(my_board.all_cards())
        seen.update(opp_board.all_cards())
        seen.update(obs.known_discards_self)
        if action.discard:
            seen.add(action.discard)
        unseen = [c for c in ALL_CARDS if c not in seen]
        random.shuffle(unseen)

        card_idx = 0
        current_turn = obs.turn

        # Play out remaining turns
        for turn in range(current_turn + 1, 9):
            # My cards (3 for Pineapple)
            if not my_board.is_complete():
                if card_idx + 3 > len(unseen):
                    break
                my_cards = unseen[card_idx : card_idx + 3]
                card_idx += 3
                self._policy_playout_turn(my_board, opp_board, my_cards, turn)

            # Opponent cards
            if not opp_board.is_complete():
                if card_idx + 3 > len(unseen):
                    break
                opp_cards = unseen[card_idx : card_idx + 3]
                card_idx += 3
                self._policy_playout_turn(opp_board, my_board, opp_cards, turn)

        # Score the final boards
        return self._compute_score(my_board, opp_board)

    def _policy_playout_turn(
        self,
        board: Board,
        opp_board: Board,
        cards: List[str],
        turn: int,
    ):
        """Use PolicyNet to pick an action for one turn during playout."""
        if board.is_complete():
            return

        valid_actions = get_turn_actions(cards, board)
        if not valid_actions:
            return

        if len(valid_actions) == 1:
            chosen = valid_actions[0]
        else:
            obs = Observation(
                board_self=board,
                board_opponent=opp_board,
                dealt_cards=cards,
                known_discards_self=[],
                turn=turn,
                is_btn=True,
            )
            state_vec = encode_state(obs)
            state_tensor = (
                torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
            )
            mask = create_action_mask(valid_actions)
            mask_tensor = (
                torch.BoolTensor(mask).unsqueeze(0).to(self.device)
            )

            with torch.no_grad():
                probs = (
                    self.policy_net(state_tensor, mask_tensor)
                    .squeeze(0)
                    .cpu()
                    .numpy()
                )

            best = int(np.argmax(probs[: len(valid_actions)]))
            chosen = valid_actions[best]

        # Apply action
        for card, pos in chosen.placements:
            getattr(board, pos).append(card)

    @staticmethod
    def _compute_score(my_board: Board, opp_board: Board) -> float:
        """Compute final score from my perspective (line comparison + royalties)."""
        # Evaluate hands
        my_vals = {
            "top": evaluate_hand(my_board.top, 3),
            "middle": evaluate_hand(my_board.middle, 5),
            "bottom": evaluate_hand(my_board.bottom, 5),
        }
        opp_vals = {
            "top": evaluate_hand(opp_board.top, 3),
            "middle": evaluate_hand(opp_board.middle, 5),
            "bottom": evaluate_hand(opp_board.bottom, 5),
        }

        # Bust check
        my_busted = my_vals["top"] > my_vals["middle"] or my_vals["middle"] > my_vals["bottom"]
        opp_busted = opp_vals["top"] > opp_vals["middle"] or opp_vals["middle"] > opp_vals["bottom"]

        # Royalties
        my_royalty = 0
        opp_royalty = 0
        if not my_busted:
            my_royalty = (
                get_top_royalty(my_board.top)
                + get_middle_royalty(my_board.middle)
                + get_bottom_royalty(my_board.bottom)
            )
        if not opp_busted:
            opp_royalty = (
                get_top_royalty(opp_board.top)
                + get_middle_royalty(opp_board.middle)
                + get_bottom_royalty(opp_board.bottom)
            )

        # Score
        if my_busted and opp_busted:
            return 0.0
        if my_busted:
            return float(-6 - opp_royalty)
        if opp_busted:
            return float(6 + my_royalty)

        # Line comparison
        line_total = 0
        for line in ["top", "middle", "bottom"]:
            if my_vals[line] > opp_vals[line]:
                line_total += 1
            elif my_vals[line] < opp_vals[line]:
                line_total -= 1

        scoop_bonus = 3 if abs(line_total) == 3 else 0
        score = line_total
        score += scoop_bonus if line_total > 0 else (-scoop_bonus if line_total < 0 else 0)
        score += my_royalty - opp_royalty

        return float(score)
