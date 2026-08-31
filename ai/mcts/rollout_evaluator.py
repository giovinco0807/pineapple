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
import itertools
from dataclasses import dataclass
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.encoding import (
    Board, Observation, encode_state, ALL_CARDS, _row_rank_numeric
)
from ai.engine.action_space import (
    get_initial_actions, get_turn_actions, create_action_mask, Action, MAX_ACTIONS
)
from ai.engine.game_engine import (
    GameEngine, evaluate_hand, get_top_royalty,
    get_middle_royalty, get_bottom_royalty,
    evaluate_board_with_joker_constraint,
    evaluate_row_with_joker_constraint,
)

import json

ACTION_VALUE_SUITS = "hdcs"
ACTION_VALUE_SUIT_MAPPINGS = [
    dict(zip(ACTION_VALUE_SUITS, perm))
    for perm in itertools.permutations(ACTION_VALUE_SUITS)
]


def _load_fl_ev():
    """Load FL EV from config. Supports direct values or chain formula."""
    config_path = Path(__file__).parent.parent / "config" / "fl_ev.json"
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        # Direct mode: use explicit FL_EV values
        if cfg.get("reward_mode") == "direct" and "fl_ev_direct" in cfg:
            fl_ev = {int(k): v for k, v in cfg["fl_ev_direct"].items()}
            return fl_ev
        # Chain formula: Net EV = (R - opp_avg) / (1 - stay_rate)
        opp = cfg["opponent_avg_royalty"]
        fl_ev = {}
        for cards, s in cfg["fl_stats"].items():
            net_r = s["R"] - opp
            fl_ev[int(cards)] = net_r / (1 - s["stay_rate"])
        return fl_ev
    except Exception:
        return {14: 3, 15: 10, 16: 15, 17: 20}


@dataclass
class RolloutResult:
    """Per-rollout detailed result for deep evaluation."""
    score: float
    busted: bool
    fl_qualified: bool
    fl_card_count: int   # 0, 14, 15, 16, 17
    my_royalty: int
    opp_busted: bool


@dataclass(frozen=True)
class _PlayoutStep:
    """One remaining public action in canonical BB-then-BTN order."""

    turn: int
    hero: bool
    is_btn: bool


class RolloutEvaluator:
    """Evaluate actions by Monte Carlo rollouts with PolicyNet guidance."""

    def __init__(
        self,
        policy_net: torch.nn.Module,
        n_rollouts: int = 200,
        top_k: int = 20,
        device: str = "cpu",
        ppo_model=None,
        ppo_temperature: float = 0.7,
        value_net: torch.nn.Module = None,
        vn_top_k: int = 5,
        norm_stats: dict = None,
        use_policy_playout: bool = False,
        full_width: bool = False,
        action_value_net: torch.nn.Module = None,
        action_value_nets_by_turn: Optional[Dict[int, torch.nn.Module]] = None,
        action_value_top_k: int = 0,
        action_value_top_k_by_turn: Optional[Dict[int, int]] = None,
        action_value_bust_weight: float = 0.0,
        action_value_fl_weight: float = 0.0,
        action_value_fl_any_weight: float = 0.0,
        action_value_fl_qq_weight: float = 0.0,
        action_value_fl_kk_weight: float = 0.0,
        action_value_fl_aa_weight: float = 0.0,
        action_value_fl_trips_weight: float = 0.0,
        action_value_suit_ensemble_turns: Optional[Iterable[int]] = None,
        action_value_suit_ensemble_size: int = 8,
    ):
        self.policy_net = policy_net
        self.n_rollouts = n_rollouts
        self.top_k = top_k
        self.device = device
        self.policy_net.eval()
        self.ppo_model = ppo_model  # Optional: use PPO for playout policy
        self.ppo_temperature = ppo_temperature
        self.use_policy_playout = use_policy_playout
        self.full_width = full_width
        self.action_value_net = action_value_net
        self.action_value_nets_by_turn = {
            int(turn): net
            for turn, net in (action_value_nets_by_turn or {}).items()
            if net is not None
        }
        self.action_value_top_k = action_value_top_k
        self.action_value_top_k_by_turn = {
            int(turn): int(top_k)
            for turn, top_k in (action_value_top_k_by_turn or {}).items()
            if int(top_k) > 0
        }
        self.action_value_bust_weight = action_value_bust_weight
        # action_value_fl_weight is kept as a backward-compatible alias for
        # FL-any.  New runs should use the explicit per-type weights.
        self.action_value_fl_any_weight = action_value_fl_weight + action_value_fl_any_weight
        self.action_value_fl_qq_weight = action_value_fl_qq_weight
        self.action_value_fl_kk_weight = action_value_fl_kk_weight
        self.action_value_fl_aa_weight = action_value_fl_aa_weight
        self.action_value_fl_trips_weight = action_value_fl_trips_weight
        self.action_value_suit_ensemble_turns = {
            int(turn) for turn in (action_value_suit_ensemble_turns or [])
        }
        self.action_value_suit_ensemble_size = max(
            1,
            min(len(ACTION_VALUE_SUIT_MAPPINGS), int(action_value_suit_ensemble_size or 8)),
        )
        if self.action_value_net is not None:
            self.action_value_net.eval()
        for net in self.action_value_nets_by_turn.values():
            net.eval()
        self.per_turn_bc = {}  # {turn_num: PolicyNetwork} for per-turn BC
        # VN prefilter: score candidates and keep top vn_top_k before rollout
        self.value_net = value_net
        self.vn_top_k = vn_top_k
        if value_net is not None:
            self.value_net.eval()
        if norm_stats:
            self.score_mean = norm_stats.get('mean', norm_stats.get('score_mean', 0.0))
            self.score_std = norm_stats.get('std', norm_stats.get('score_std', 1.0))
        else:
            self.score_mean = 0.0
            self.score_std = 1.0
        # bust_prob penalty: adjusted_score = rollout_avg - bust_penalty * bust_prob
        self.bust_penalty = 0.0
        # VN-truncated rollout: play `depth` turns, then batch VN eval
        self.vn_truncate_depth = 0   # 0 = disabled (full rollout)
        self.vn_truncate_n = 500     # rollouts per candidate in truncated mode
        self.vn_hybrid = True        # True = full playout + VN blend, False = truncate only

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

        action_value_net = self._action_value_net_for_turn(obs.turn)
        if self.full_width or action_value_net is not None:
            candidates = list(enumerate(valid_actions))
        else:
            # FL intervention: filter candidates by rule-based FL decision
            candidates = self._get_fl_filtered_candidates(obs, valid_actions)

        if action_value_net is not None and (not self.full_width or self.n_rollouts <= 0):
            if self.n_rollouts <= 0:
                best_idx = self._select_by_action_value(obs, candidates)
                return best_idx, valid_actions[best_idx]
            candidates = self._action_value_prefilter(obs, candidates)

        # VN prefilter: score candidates with VN, keep top vn_top_k
        if self.value_net is not None and len(candidates) > self.vn_top_k:
            candidates = self._vn_prefilter(obs, candidates)

        # VN-truncated rollout mode: play depth turns, batch VN eval
        if self.vn_truncate_depth > 0 and self.value_net is not None:
            best_idx = self._evaluate_actions_vn_truncated(
                obs, candidates, self.vn_truncate_n, self.vn_truncate_depth)
            return best_idx, valid_actions[best_idx]

        # Get VN bust_prob for each candidate (if bust_penalty enabled)
        bust_probs = {}
        if self.bust_penalty > 0 and self.value_net is not None:
            bust_probs = self._get_candidate_bust_probs(obs, candidates)

        # Evaluate each candidate by N rollouts
        best_idx = -1
        best_score = float("-inf")

        for orig_idx, action in candidates:
            avg = self._evaluate_action(obs, action)
            # Adjust by bust_prob penalty
            if orig_idx in bust_probs:
                avg -= self.bust_penalty * bust_probs[orig_idx]
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

        action_value_net = self._action_value_net_for_turn(obs.turn)
        if self.full_width or action_value_net is not None:
            candidates = list(enumerate(valid_actions))
        else:
            # FL intervention: filter candidates by rule-based FL decision
            candidates = self._get_fl_filtered_candidates(obs, valid_actions)

        if action_value_net is not None and (not self.full_width or self.n_rollouts <= 0):
            if self.n_rollouts <= 0:
                scored = self._score_candidates_action_value(obs, candidates)
                scores = [float("-inf")] * len(valid_actions)
                best_pos = 0
                best_score = float("-inf")
                for j, (orig_idx, _action) in enumerate(candidates):
                    score = float(scored[j])
                    scores[orig_idx] = score
                    if score > best_score:
                        best_score = score
                        best_pos = j
                best_idx = candidates[best_pos][0]
                return best_idx, valid_actions[best_idx], scores
            candidates = self._action_value_prefilter(obs, candidates)

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
    # FL Intervention Logic
    # ------------------------------------------------------------------

    @staticmethod
    def _should_go_fl(dealt_cards: List[str]) -> Optional[str]:
        """Rule-based FL decision from Turn 0's 5 cards.

        Returns 'AA', 'KK', or None.
        Triggers on single A/K to support aggressive FL pursuit (target 28%+).
        """
        ranks = [c[0] for c in dealt_cards if not c.startswith('X')]
        jokers = sum(1 for c in dealt_cards if c.startswith('X'))
        a_count = ranks.count('A')
        k_count = ranks.count('K')

        # Trigger on any A (pair, single, or joker-assisted)
        if a_count >= 1 or jokers >= 1:
            return 'AA'
        if k_count >= 1:
            return 'KK'
        return None

    @staticmethod
    def _filter_fl_candidates(
        valid_actions: List[Action], target_rank: str
    ) -> List[Tuple[int, Action]]:
        """Filter actions to only those placing target rank (A/K) on top."""
        fl_candidates = []
        for i, action in enumerate(valid_actions):
            for card, pos in action.placements:
                if pos == 'top':
                    if card[0] == target_rank or card.startswith('X'):
                        fl_candidates.append((i, action))
                        break
        return fl_candidates

    def _get_fl_filtered_candidates(
        self, obs: Observation, valid_actions: List[Action]
    ) -> List[Tuple[int, Action]]:
        """Get candidates with FL intervention applied.

        T0: if A/K in hand, only consider actions placing A/K on top
        T1+: if top has A/K and hand has matching rank, filter to pair-completing actions
        """
        if obs.turn == 0:
            fl_decision = self._should_go_fl(obs.dealt_cards)
            if fl_decision:
                target = fl_decision[0]  # 'A' or 'K'
                fl_cands = self._filter_fl_candidates(valid_actions, target)
                if fl_cands:
                    # Apply top-K filter within FL candidates
                    if len(fl_cands) > self.top_k:
                        all_candidates = self._filter_top_k(obs, valid_actions)
                        fl_set = set(i for i, _ in fl_cands)
                        filtered = [(i, a) for i, a in all_candidates if i in fl_set]
                        # Ensure at least some FL candidates survive
                        if len(filtered) < 5:
                            filtered = fl_cands[:self.top_k]
                        return filtered
                    return fl_cands
            # No FL decision or no FL candidates: normal flow
            if len(valid_actions) > self.top_k:
                return self._filter_top_k(obs, valid_actions)
            return list(enumerate(valid_actions))

        # T1+: FL pair completion and top protection
        if obs.turn <= 7 and len(obs.board_self.top) < 3:
            top_ranks = set()
            for c in obs.board_self.top:
                if not c.startswith('X') and c[0] in ('K', 'A'):
                    top_ranks.add(c[0])
                elif c.startswith('X'):
                    top_ranks.update(['A', 'K'])  # Joker pairs with anything

            if top_ranks:
                # Check if dealt cards have a matching rank
                dealt_ranks = [c[0] for c in obs.dealt_cards if not c.startswith('X')]
                dealt_jokers = any(c.startswith('X') for c in obs.dealt_cards)
                matching = top_ranks & set(dealt_ranks)

                if matching or dealt_jokers:
                    # Mode 1: MATCH FOUND - force place matching rank on top
                    pair_cands = []
                    for i, action in enumerate(valid_actions):
                        for card, pos in action.placements:
                            if pos == 'top' and (card[0] in top_ranks or card.startswith('X')):
                                pair_cands.append((i, action))
                                break
                    if pair_cands:
                        return pair_cands
                else:
                    # Mode 2: NO MATCH - protect top slots
                    # Filter to actions that don't place anything on top
                    protect_cands = []
                    for i, action in enumerate(valid_actions):
                        places_on_top = any(pos == 'top' for _, pos in action.placements)
                        if not places_on_top:
                            protect_cands.append((i, action))
                    if protect_cands:
                        return protect_cands

        return list(enumerate(valid_actions))

    # ------------------------------------------------------------------
    # Action-value reranker

    def _action_value_net_for_turn(self, turn: int) -> Optional[torch.nn.Module]:
        return self.action_value_nets_by_turn.get(int(turn), self.action_value_net)
    # ------------------------------------------------------------------

    _SUIT_MAPPINGS = ACTION_VALUE_SUIT_MAPPINGS

    @staticmethod
    def _spread_indices(n_items: int, n_selected: int) -> List[int]:
        n_items = max(0, int(n_items))
        n_selected = max(0, min(int(n_selected), n_items))
        if n_selected <= 0:
            return []
        if n_selected == 1:
            return [0]
        step = (n_items - 1) / float(n_selected - 1)
        indices = [int(round(i * step)) for i in range(n_selected)]
        # Rounding can collide for unusual sizes; fill deterministically.
        seen = []
        for idx in indices:
            if idx not in seen:
                seen.append(idx)
        for idx in range(n_items):
            if len(seen) >= n_selected:
                break
            if idx not in seen:
                seen.append(idx)
        return seen[:n_selected]

    def _selected_suit_mappings(self) -> List[Dict[str, str]]:
        indices = self._spread_indices(len(self._SUIT_MAPPINGS), self.action_value_suit_ensemble_size)
        return [self._SUIT_MAPPINGS[idx] for idx in indices]

    @staticmethod
    def _permute_card_suit(card: str, mapping: Dict[str, str]) -> str:
        card = str(card)
        if len(card) < 2 or card.startswith("X") or card[-1] not in mapping:
            return card
        return card[:-1] + mapping[card[-1]]

    @classmethod
    def _permute_cards_suit(cls, cards: Iterable[str], mapping: Dict[str, str]) -> List[str]:
        return [cls._permute_card_suit(card, mapping) for card in cards]

    @classmethod
    def _permute_board_suit(cls, board: Board, mapping: Dict[str, str]) -> Board:
        return Board(
            top=cls._permute_cards_suit(board.top, mapping),
            middle=cls._permute_cards_suit(board.middle, mapping),
            bottom=cls._permute_cards_suit(board.bottom, mapping),
        )

    @classmethod
    def _permute_observation_suit(cls, obs: Observation, mapping: Dict[str, str]) -> Observation:
        return Observation(
            board_self=cls._permute_board_suit(obs.board_self, mapping),
            board_opponent=cls._permute_board_suit(obs.board_opponent, mapping),
            dealt_cards=cls._permute_cards_suit(obs.dealt_cards, mapping),
            known_discards_self=cls._permute_cards_suit(obs.known_discards_self, mapping),
            turn=obs.turn,
            is_btn=obs.is_btn,
            is_fl=obs.is_fl,
            opp_is_fl=obs.opp_is_fl,
            chips_self=obs.chips_self,
            chips_opponent=obs.chips_opponent,
        )

    def _post_action_obs_for_reranker(self, obs: Observation, action: Action) -> Observation:
        """Build a post-action observation for reranking."""
        new_board = obs.board_self.copy()
        for card, pos in action.placements:
            getattr(new_board, pos).append(card)

        # Opponent row shape matters once it can influence route pressure and
        # line comparison.  For BB T0 we keep opponent cards as unavailable but
        # do not encode their row placement, matching the teacher converter.
        encode_opponent_shape = not (obs.turn == 0 and not obs.is_btn)
        opponent_board = obs.board_opponent if encode_opponent_shape else Board()

        unavailable = []
        seen_self = set(new_board.all_cards())
        seen_self.update(opponent_board.all_cards())
        for card in list(obs.known_discards_self) + obs.board_opponent.all_cards():
            if card not in seen_self and card not in unavailable:
                unavailable.append(card)
        if action.discard and action.discard not in unavailable:
            unavailable.append(action.discard)

        return Observation(
            board_self=new_board,
            board_opponent=opponent_board,
            dealt_cards=[],
            known_discards_self=unavailable,
            turn=obs.turn,
            is_btn=obs.is_btn,
            is_fl=False,
            opp_is_fl=False,
            chips_self=obs.chips_self,
            chips_opponent=obs.chips_opponent,
        )

    def _score_encoded_states_action_value(
        self,
        state_batch: List[np.ndarray],
        *,
        turn: Optional[int] = None,
        action_value_net: Optional[torch.nn.Module] = None,
        conditional_gate_mask: Optional[np.ndarray] = None,
        conditional_gate_masks: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        details = self._score_encoded_states_action_value_details(
            state_batch,
            turn=turn,
            action_value_net=action_value_net,
            conditional_gate_mask=conditional_gate_mask,
            conditional_gate_masks=conditional_gate_masks,
        )
        return details["model_score"]

    def _score_encoded_states_action_value_details(
        self,
        state_batch: List[np.ndarray],
        *,
        turn: Optional[int] = None,
        action_value_net: Optional[torch.nn.Module] = None,
        conditional_gate_mask: Optional[np.ndarray] = None,
        conditional_gate_masks: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        if not state_batch:
            return {
                "model_score": np.array([], dtype=np.float32),
                "raw_score": np.array([], dtype=np.float32),
                "bust_prob": np.array([], dtype=np.float32),
                "fl_prob": np.array([], dtype=np.float32),
                "fl_type_probs": np.zeros((0, 4), dtype=np.float32),
            }
        net = action_value_net or self.action_value_net
        if net is None:
            return {
                "model_score": np.array([], dtype=np.float32),
                "raw_score": np.array([], dtype=np.float32),
                "bust_prob": np.array([], dtype=np.float32),
                "fl_prob": np.array([], dtype=np.float32),
                "fl_type_probs": np.zeros((0, 4), dtype=np.float32),
            }

        model_scores = []
        raw_scores = []
        bust_probs = []
        fl_probs = []
        fl_type_probs = []
        with torch.no_grad():
            for start in range(0, len(state_batch), 4096):
                end = min(len(state_batch), start + 4096)
                batch = torch.FloatTensor(np.array(state_batch[start:end])).to(self.device)
                gate_tensor = None
                if conditional_gate_mask is not None:
                    gate_tensor = torch.as_tensor(
                        np.asarray(conditional_gate_mask[start:end], dtype=bool),
                        dtype=torch.bool,
                        device=batch.device,
                    )
                gate_tensors = None
                if conditional_gate_masks is not None:
                    gate_tensors = {
                        str(gate): torch.as_tensor(
                            np.asarray(mask[start:end], dtype=bool),
                            dtype=torch.bool,
                            device=batch.device,
                        )
                        for gate, mask in conditional_gate_masks.items()
                    }
                if hasattr(net, "predict_components"):
                    if turn is not None:
                        turn_tensor = torch.full(
                            (batch.shape[0],),
                            int(turn),
                            dtype=torch.long,
                            device=batch.device,
                        )
                        try:
                            if gate_tensors is not None and hasattr(net, "predict_components_with_gate_masks"):
                                out = net.predict_components_with_gate_masks(
                                    batch,
                                    gate_masks=gate_tensors,
                                    turn=turn_tensor,
                                )
                            elif gate_tensor is not None and hasattr(net, "predict_components_with_gate_mask"):
                                out = net.predict_components_with_gate_mask(
                                    batch,
                                    gate_mask=gate_tensor,
                                    turn=turn_tensor,
                                )
                            else:
                                out = net.predict_components(batch, turn=turn_tensor)
                        except TypeError:
                            out = net.predict_components(batch)
                    else:
                        if gate_tensors is not None and hasattr(net, "predict_components_with_gate_masks"):
                            out = net.predict_components_with_gate_masks(batch, gate_masks=gate_tensors)
                        elif gate_tensor is not None and hasattr(net, "predict_components_with_gate_mask"):
                            out = net.predict_components_with_gate_mask(batch, gate_mask=gate_tensor)
                        else:
                            out = net.predict_components(batch)
                    score_t = out["score"]
                    bust_t = out.get("bust_prob", torch.zeros_like(score_t))
                    fl_t = out.get("fl_prob", torch.zeros_like(score_t))
                    fl_type_t = out.get(
                        "fl_type_probs",
                        torch.zeros((batch.shape[0], 4), dtype=score_t.dtype, device=score_t.device),
                    )
                    adjusted_t = score_t
                    if self.action_value_bust_weight:
                        adjusted_t = adjusted_t - self.action_value_bust_weight * bust_t
                    if self.action_value_fl_any_weight:
                        adjusted_t = adjusted_t + self.action_value_fl_any_weight * fl_t
                    if self.action_value_fl_qq_weight and "fl_qq" in out:
                        adjusted_t = adjusted_t + self.action_value_fl_qq_weight * out["fl_qq"]
                    if self.action_value_fl_kk_weight and "fl_kk" in out:
                        adjusted_t = adjusted_t + self.action_value_fl_kk_weight * out["fl_kk"]
                    if self.action_value_fl_aa_weight and "fl_aa" in out:
                        adjusted_t = adjusted_t + self.action_value_fl_aa_weight * out["fl_aa"]
                    if self.action_value_fl_trips_weight and "fl_trips" in out:
                        adjusted_t = adjusted_t + self.action_value_fl_trips_weight * out["fl_trips"]
                else:
                    raw = net(batch)
                    score_t = raw["score"] if isinstance(raw, dict) else raw.squeeze(-1)
                    bust_t = torch.zeros_like(score_t)
                    fl_t = torch.zeros_like(score_t)
                    fl_type_t = torch.zeros((batch.shape[0], 4), dtype=score_t.dtype, device=score_t.device)
                    adjusted_t = score_t
                model_scores.append(adjusted_t.detach().cpu().numpy())
                raw_scores.append(score_t.detach().cpu().numpy())
                bust_probs.append(bust_t.detach().cpu().numpy())
                fl_probs.append(fl_t.detach().cpu().numpy())
                fl_type_probs.append(fl_type_t.detach().cpu().numpy())

        return {
            "model_score": np.concatenate(model_scores).astype(np.float32, copy=False),
            "raw_score": np.concatenate(raw_scores).astype(np.float32, copy=False),
            "bust_prob": np.concatenate(bust_probs).astype(np.float32, copy=False),
            "fl_prob": np.concatenate(fl_probs).astype(np.float32, copy=False),
            "fl_type_probs": np.concatenate(fl_type_probs).astype(np.float32, copy=False),
        }

    def _score_candidates_action_value(
        self, obs: Observation, candidates: List[Tuple[int, Action]]
    ) -> np.ndarray:
        """Return reranker scores aligned to ``candidates``."""
        return self.score_candidates_action_value_details(obs, candidates)["model_score"]

    def score_candidates_action_value_details(
        self, obs: Observation, candidates: List[Tuple[int, Action]]
    ) -> Dict[str, np.ndarray]:
        """Return reranker score components aligned to ``candidates``."""
        if not candidates:
            return {
                "model_score": np.array([], dtype=np.float32),
                "raw_score": np.array([], dtype=np.float32),
                "bust_prob": np.array([], dtype=np.float32),
                "fl_prob": np.array([], dtype=np.float32),
                "fl_type_probs": np.zeros((0, 4), dtype=np.float32),
            }
        action_value_net = self._action_value_net_for_turn(obs.turn)
        if action_value_net is None:
            return {
                "model_score": np.array([], dtype=np.float32),
                "raw_score": np.array([], dtype=np.float32),
                "bust_prob": np.array([], dtype=np.float32),
                "fl_prob": np.array([], dtype=np.float32),
                "fl_type_probs": np.zeros((0, 4), dtype=np.float32),
            }

        post_action_obs = [
            self._post_action_obs_for_reranker(obs, action)
            for _orig_idx, action in candidates
        ]
        conditional_gate_mask = None
        conditional_gate_masks = None
        if hasattr(action_value_net, "candidate_gate_masks"):
            gate_map = action_value_net.candidate_gate_masks(obs, post_action_obs)
            if gate_map is not None:
                conditional_gate_masks = {
                    str(gate): np.asarray(values, dtype=bool)
                    for gate, values in gate_map.items()
                }
        elif hasattr(action_value_net, "candidate_gate_mask"):
            gate_values = action_value_net.candidate_gate_mask(obs, post_action_obs)
            if gate_values is not None:
                conditional_gate_mask = np.asarray(gate_values, dtype=bool)

        if int(obs.turn) in self.action_value_suit_ensemble_turns:
            mappings = self._selected_suit_mappings()
            states = []
            expanded_gate_mask = None
            expanded_gate_masks = None
            if conditional_gate_mask is not None:
                expanded_gate_mask = np.repeat(conditional_gate_mask, len(mappings))
            if conditional_gate_masks is not None:
                expanded_gate_masks = {
                    gate: np.repeat(mask, len(mappings))
                    for gate, mask in conditional_gate_masks.items()
                }
            for candidate_obs in post_action_obs:
                for mapping in mappings:
                    states.append(encode_state(self._permute_observation_suit(candidate_obs, mapping)))
            flat = self._score_encoded_states_action_value_details(
                states,
                turn=obs.turn,
                action_value_net=action_value_net,
                conditional_gate_mask=expanded_gate_mask,
                conditional_gate_masks=expanded_gate_masks,
            )
            if len(flat["model_score"]) != len(candidates) * len(mappings):
                return {
                    "model_score": flat["model_score"][: len(candidates)],
                    "raw_score": flat["raw_score"][: len(candidates)],
                    "bust_prob": flat["bust_prob"][: len(candidates)],
                    "fl_prob": flat["fl_prob"][: len(candidates)],
                    "fl_type_probs": flat["fl_type_probs"][: len(candidates)],
                }
            out = {}
            for key in ("model_score", "raw_score", "bust_prob", "fl_prob"):
                out[key] = flat[key].reshape(len(candidates), len(mappings)).mean(axis=1).astype(
                    np.float32,
                    copy=False,
                )
            out["fl_type_probs"] = flat["fl_type_probs"].reshape(
                len(candidates),
                len(mappings),
                4,
            ).mean(axis=1).astype(np.float32, copy=False)
            return out

        state_batch = [encode_state(candidate_obs) for candidate_obs in post_action_obs]
        return self._score_encoded_states_action_value_details(
            state_batch,
            turn=obs.turn,
            action_value_net=action_value_net,
            conditional_gate_mask=conditional_gate_mask,
            conditional_gate_masks=conditional_gate_masks,
        )

    def _select_by_action_value(
        self, obs: Observation, candidates: List[Tuple[int, Action]]
    ) -> int:
        scores = self._score_candidates_action_value(obs, candidates)
        if len(scores) == 0:
            return candidates[0][0]
        return candidates[int(np.argmax(scores))][0]

    def _action_value_prefilter_budget(self, obs: Observation) -> int:
        budget = int(self.action_value_top_k or 0)
        turn_budget = int(self.action_value_top_k_by_turn.get(int(obs.turn), 0))
        if turn_budget > 0:
            budget = max(budget, turn_budget)
        return budget

    def _action_value_prefilter(
        self, obs: Observation, candidates: List[Tuple[int, Action]]
    ) -> List[Tuple[int, Action]]:
        budget = self._action_value_prefilter_budget(obs)
        if budget <= 0 or len(candidates) <= budget:
            return candidates
        scores = self._score_candidates_action_value(obs, candidates)
        order = np.argsort(-scores)[:budget]
        return [candidates[int(i)] for i in order]

    def _vn_prefilter(
        self, obs: Observation, candidates: List[Tuple[int, Action]]
    ) -> List[Tuple[int, Action]]:
        """Score candidates with VN and keep top vn_top_k.

        FL-aware: protects up to 2 FL-promising candidates from being
        dropped by VN scoring. VN undervalues FL routes because training
        data is 84% non-FL. Protected FL candidates get rollout evaluation
        where multi-turn playout correctly values FL potential.
        """
        # --- Step 1: Identify FL-promising candidates to protect ---
        fl_protected = []
        fl_protected_set = set()
        max_fl_protect = 2

        top_cards = obs.board_self.top
        top_ranks = set()
        for c in top_cards:
            if c.startswith('X'):
                top_ranks.update(['A', 'K'])
            elif len(c) >= 2 and c[:-1] in ('A', 'K', 'Q'):
                top_ranks.add(c[:-1])

        for i, (orig_idx, action) in enumerate(candidates):
            if len(fl_protected) >= max_fl_protect:
                break
            is_fl_candidate = False
            for card, pos in action.placements:
                if pos == 'top':
                    # T0: placing A/K on top = FL intent
                    if obs.turn == 0 and not card.startswith('X'):
                        rank = card[:-1]
                        if rank in ('A', 'K'):
                            is_fl_candidate = True
                    # T1+: completing pair on top
                    elif obs.turn > 0 and top_ranks:
                        rank = card[:-1] if not card.startswith('X') else 'X'
                        if rank in top_ranks or card.startswith('X'):
                            is_fl_candidate = True
            if is_fl_candidate:
                fl_protected.append((orig_idx, action))
                fl_protected_set.add(i)

        # --- Step 2: VN score remaining candidates ---
        remaining = [(i, c) for i, c in enumerate(candidates) if i not in fl_protected_set]

        if not remaining:
            return fl_protected[:self.vn_top_k]

        state_batch = []
        for _, (orig_idx, action) in remaining:
            new_board = obs.board_self.copy()
            for card, pos in action.placements:
                getattr(new_board, pos).append(card)

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
                is_fl=obs.is_fl,
                opp_is_fl=obs.opp_is_fl,
                chips_self=obs.chips_self,
                chips_opponent=obs.chips_opponent,
            )
            state_batch.append(encode_state(new_obs))

        with torch.no_grad():
            states_t = torch.FloatTensor(np.array(state_batch)).to(self.device)
            output = self.value_net(states_t)
            vn_scores = output['value'].squeeze(-1).cpu().tolist()
            # Use bust_prob to adjust VN prefilter scores (zero-cost, already computed)
            if self.bust_penalty > 0:
                bust_probs = output['bust_prob'].squeeze(-1).cpu().tolist()
                vn_scores = [v - self.bust_penalty * b for v, b in zip(vn_scores, bust_probs)]

        # Sort remaining by VN score
        scored = [(vn_scores[j], remaining[j][1]) for j in range(len(remaining))]
        scored.sort(key=lambda x: x[0], reverse=True)

        # --- Step 3: Combine protected FL + VN top picks ---
        vn_budget = self.vn_top_k - len(fl_protected)
        vn_top = [cand for _, cand in scored[:max(vn_budget, 1)]]

        result = fl_protected + vn_top
        return result[:self.vn_top_k]


    def _get_candidate_bust_probs(
        self, obs: Observation, candidates: List[Tuple[int, Action]]
    ) -> dict:
        """Get VN bust_prob for each candidate action."""
        state_batch = []
        idx_map = []
        for orig_idx, action in candidates:
            new_board = obs.board_self.copy()
            for card, pos in action.placements:
                getattr(new_board, pos).append(card)
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
                is_fl=obs.is_fl,
                opp_is_fl=obs.opp_is_fl,
                chips_self=obs.chips_self,
                chips_opponent=obs.chips_opponent,
            )
            state_batch.append(encode_state(new_obs))
            idx_map.append(orig_idx)

        with torch.no_grad():
            states_t = torch.FloatTensor(np.array(state_batch)).to(self.device)
            output = self.value_net(states_t)
            bust_probs = output['bust_prob'].squeeze(-1).cpu().tolist()

        return {idx_map[j]: bust_probs[j] for j in range(len(idx_map))}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _filter_top_k(
        self, obs: Observation, valid_actions: List[Action]
    ) -> List[Tuple[int, Action]]:
        """Filter Turn 0 candidates: only A/K/Joker on Top.

        Three tiers:
          Tier 1: Actions that place ONLY A/K/Joker on Top (ideal FL)
          Tier 2: Actions that place nothing on Top (no FL, safe)
          Tier 3: All others (fallback)
        """
        state_vec = encode_state(obs)
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
        mask = create_action_mask(valid_actions)
        mask_tensor = torch.BoolTensor(mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = self.policy_net(state_tensor, mask_tensor).squeeze(0).cpu().numpy()

        tier1 = []  # Only A/K/Joker on Top
        tier2 = []  # Nothing on Top
        tier3 = []  # Others

        for i, action in enumerate(valid_actions):
            p = probs[i] if i < len(probs) else 0
            top_cards = [c for c, pos in action.placements if pos == 'top']

            if not top_cards:
                tier2.append((i, action, p))
            else:
                all_valid = all(
                    c.startswith('X') or c[0] in ('A', 'K')
                    for c in top_cards
                )
                if all_valid:
                    tier1.append((i, action, p))
                else:
                    tier3.append((i, action, p))

        # Build candidate list: all tier1, top-K from tier2, fallback tier3
        candidates = []
        seen = set()

        for i, action, p in tier1:
            if i not in seen:
                seen.add(i)
                candidates.append((i, action))

        tier2.sort(key=lambda x: x[2], reverse=True)
        for i, action, p in tier2[:self.top_k]:
            if i not in seen:
                seen.add(i)
                candidates.append((i, action))

        if len(candidates) < self.top_k:
            tier3.sort(key=lambda x: x[2], reverse=True)
            remaining = self.top_k - len(candidates)
            for i, action, p in tier3[:remaining]:
                if i not in seen:
                    seen.add(i)
                    candidates.append((i, action))

        return candidates if candidates else [(0, valid_actions[0])]

    @staticmethod
    def _remaining_playout_steps(
        current_turn: int,
        root_is_btn: bool,
        end_turn: int = 8,
    ) -> List[_PlayoutStep]:
        """Return remaining actions in canonical BB-first street order.

        The root action has already been applied.  A BB root therefore still
        has the BTN response pending on the current street.  A BTN root has
        completed the current street, so the next action is BB on the next
        street.  ``hero`` identifies the root observation's player, while
        ``is_btn`` is the acting player's positional flag.
        """
        current_turn = int(current_turn)
        end_turn = int(end_turn)
        if current_turn < 0 or end_turn < current_turn:
            return []

        steps: List[_PlayoutStep] = []
        if not root_is_btn:
            steps.append(_PlayoutStep(turn=current_turn, hero=False, is_btn=True))

        for turn in range(current_turn + 1, end_turn + 1):
            if root_is_btn:
                # Opponent is BB, root hero is BTN.
                steps.append(_PlayoutStep(turn=turn, hero=False, is_btn=False))
                steps.append(_PlayoutStep(turn=turn, hero=True, is_btn=True))
            else:
                # Root hero is BB, opponent is BTN.
                steps.append(_PlayoutStep(turn=turn, hero=True, is_btn=False))
                steps.append(_PlayoutStep(turn=turn, hero=False, is_btn=True))
        return steps

    def _evaluate_actions_vn_truncated(
        self, obs: Observation, candidates: List[Tuple[int, Action]],
        n_rollouts: int, depth: int = 1,
    ) -> int:
        """Hybrid or truncated evaluation with VN.

        Hybrid (vn_hybrid=True):
          1. Apply action → play `depth` turns → VN snapshot
          2. Continue to completion → exact final score
          3. Score = 0.5 * exact + 0.5 * VN prediction

        Truncated (vn_hybrid=False):
          1. Apply action → play `depth` turns
          2. If complete → exact score. Otherwise → VN score only.

        All VN evaluations batched in one forward pass.
        Returns: orig_idx of best candidate.
        """
        hybrid = self.vn_hybrid
        vn_weight = 0.5  # Blend weight for VN vs rollout (hybrid mode)

        # Per-rollout data: (c_idx, exact_score, vn_batch_idx_or_None)
        rollout_data = []
        vn_batch = []  # encoded states for batch VN eval

        for c_idx, (orig_idx, action) in enumerate(candidates):
            for _ in range(n_rollouts):
                # Apply candidate action
                my_board = obs.board_self.copy()
                for card, pos in action.placements:
                    getattr(my_board, pos).append(card)

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
                vn_idx = None  # Will be set if we capture VN snapshot
                my_discards = list(obs.known_discards_self)
                if action.discard:
                    my_discards.append(action.discard)
                opp_discards = []

                # Phase 1: Play `depth` turns → VN snapshot point
                snapshot_turn = min(current_turn + depth, 8)
                remaining_steps = self._remaining_playout_steps(
                    current_turn, obs.is_btn, end_turn=8
                )
                for step in (s for s in remaining_steps if s.turn <= snapshot_turn):
                    if my_board.is_complete() and opp_board.is_complete():
                        break
                    board = my_board if step.hero else opp_board
                    other_board = opp_board if step.hero else my_board
                    discards = my_discards if step.hero else opp_discards
                    if board.is_complete():
                        continue
                    draw_count = 5 if board.card_count() == 0 else 3
                    if card_idx + draw_count > len(unseen):
                        break
                    cards = unseen[card_idx:card_idx + draw_count]
                    card_idx += draw_count
                    self._do_playout_turn(
                        board, other_board, cards, step.turn, discards, step.is_btn
                    )

                # Capture VN snapshot (if board not yet complete)
                board_complete = my_board.is_complete() and opp_board.is_complete()
                if not board_complete:
                    my_discards = list(obs.known_discards_self)
                    if action.discard:
                        my_discards.append(action.discard)
                    snap_obs = Observation(
                        board_self=my_board.copy() if hybrid else my_board,
                        board_opponent=opp_board.copy() if hybrid else opp_board,
                        dealt_cards=[],
                        known_discards_self=my_discards,
                        turn=snapshot_turn,
                        is_btn=obs.is_btn,
                        is_fl=obs.is_fl,
                        opp_is_fl=obs.opp_is_fl,
                        chips_self=obs.chips_self,
                        chips_opponent=obs.chips_opponent,
                    )
                    vn_idx = len(vn_batch)
                    vn_batch.append(encode_state(snap_obs))

                # Phase 2: Continue playing to completion (hybrid only)
                exact_score = None
                if hybrid or board_complete:
                    if not board_complete:
                        for step in (s for s in remaining_steps if s.turn > snapshot_turn):
                            if my_board.is_complete() and opp_board.is_complete():
                                break
                            board = my_board if step.hero else opp_board
                            other_board = opp_board if step.hero else my_board
                            discards = my_discards if step.hero else opp_discards
                            if board.is_complete():
                                continue
                            draw_count = 5 if board.card_count() == 0 else 3
                            if card_idx + draw_count > len(unseen):
                                break
                            cards = unseen[card_idx:card_idx + draw_count]
                            card_idx += draw_count
                            self._do_playout_turn(
                                board, other_board, cards, step.turn, discards, step.is_btn
                            )
                    exact_score = self._compute_score(my_board, opp_board)

                rollout_data.append((c_idx, exact_score, vn_idx))

        # Batch VN evaluation
        vn_scores = None
        if vn_batch:
            with torch.no_grad():
                all_vn = []
                for i in range(0, len(vn_batch), 4096):
                    batch_slice = torch.FloatTensor(
                        np.array(vn_batch[i:i + 4096])).to(self.device)
                    output = self.value_net(batch_slice)
                    vals = output['value'].squeeze(-1).cpu().numpy()
                    vals = vals * self.score_std + self.score_mean
                    all_vn.append(vals)
                vn_scores = np.concatenate(all_vn)

        # Combine: blended score per rollout
        n_cands = len(candidates)
        totals = [0.0] * n_cands
        counts = [0] * n_cands

        for c_idx, exact, vn_idx in rollout_data:
            if vn_idx is not None and vn_scores is not None:
                vn_val = float(vn_scores[vn_idx])
                if exact is not None:
                    # Hybrid: blend rollout + VN
                    score = (1.0 - vn_weight) * exact + vn_weight * vn_val
                else:
                    # Truncated: VN only
                    score = vn_val
            else:
                # Board completed before VN snapshot depth
                score = exact if exact is not None else 0.0
            totals[c_idx] += score
            counts[c_idx] += 1

        best_c_idx = 0
        best_avg = float('-inf')
        for c_idx in range(n_cands):
            if counts[c_idx] > 0:
                avg = totals[c_idx] / counts[c_idx]
                if avg > best_avg:
                    best_avg = avg
                    best_c_idx = c_idx

        return candidates[best_c_idx][0]  # Return orig_idx

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

        # Track discards for Observation construction
        my_discards = list(obs.known_discards_self)
        if action.discard:
            my_discards.append(action.discard)
        opp_discards = []

        # Choose playout method
        use_ppo = self.ppo_model is not None

        # Complete the current street when Hero is BB, then play every future
        # street in canonical BB-then-BTN order.
        for step in self._remaining_playout_steps(current_turn, obs.is_btn):
            if my_board.is_complete() and opp_board.is_complete():
                break
            board = my_board if step.hero else opp_board
            other_board = opp_board if step.hero else my_board
            discards = my_discards if step.hero else opp_discards
            if board.is_complete():
                continue
            draw_count = 5 if board.card_count() == 0 else 3
            if card_idx + draw_count > len(unseen):
                break
            cards = unseen[card_idx : card_idx + draw_count]
            card_idx += draw_count
            if use_ppo:
                self._ppo_playout_turn(
                    board, other_board, cards, step.turn, discards, step.is_btn
                )
            elif self.use_policy_playout or board.card_count() == 0:
                self._bc_playout_turn(
                    board, other_board, cards, step.turn, discards, step.is_btn
                )
            else:
                self._rule_playout_turn(board, other_board, cards, step.turn)

        # Score the final boards
        return self._compute_score(my_board, opp_board)

    def _single_rollout_detailed(self, obs: Observation, action: Action) -> RolloutResult:
        """One complete rollout returning detailed statistics (bust/FL/royalty)."""
        # Apply the candidate action to my board
        my_board = obs.board_self.copy()
        for card, pos in action.placements:
            getattr(my_board, pos).append(card)

        # Opponent board
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

        my_discards = list(obs.known_discards_self)
        if action.discard:
            my_discards.append(action.discard)
        opp_discards = []

        use_ppo = self.ppo_model is not None

        for step in self._remaining_playout_steps(current_turn, obs.is_btn):
            if my_board.is_complete() and opp_board.is_complete():
                break
            board = my_board if step.hero else opp_board
            other_board = opp_board if step.hero else my_board
            discards = my_discards if step.hero else opp_discards
            if board.is_complete():
                continue
            draw_count = 5 if board.card_count() == 0 else 3
            if card_idx + draw_count > len(unseen):
                break
            cards = unseen[card_idx : card_idx + draw_count]
            card_idx += draw_count
            if use_ppo:
                self._ppo_playout_turn(
                    board, other_board, cards, step.turn, discards, step.is_btn
                )
            elif self.use_policy_playout or board.card_count() == 0:
                self._bc_playout_turn(
                    board, other_board, cards, step.turn, discards, step.is_btn
                )
            else:
                self._rule_playout_turn(board, other_board, cards, step.turn)

        # Detailed evaluation of final boards.  Do not independently maximize
        # each raw row: that disagrees with Joker bust-prevention.
        my_eval = evaluate_board_with_joker_constraint(
            my_board.top, my_board.middle, my_board.bottom
        )
        opp_eval = evaluate_board_with_joker_constraint(
            opp_board.top, opp_board.middle, opp_board.bottom
        )
        my_busted = bool(my_eval["busted"])
        opp_busted = bool(opp_eval["busted"])
        fl_card_count = int(my_eval["fl_card_count"])
        fl_qualified = bool(my_eval["fl_entry"])
        my_royalty = int(my_eval["royalties"]["total"])

        score = self._compute_score(my_board, opp_board)

        return RolloutResult(
            score=score,
            busted=my_busted,
            fl_qualified=fl_qualified,
            fl_card_count=fl_card_count,
            my_royalty=my_royalty,
            opp_busted=opp_busted,
        )

    def _evaluate_action_detailed(self, obs: Observation, action: Action,
                                   n_rollouts: int = 2000) -> dict:
        """Run N rollouts for one action and return detailed statistics."""
        results = [self._single_rollout_detailed(obs, action) for _ in range(n_rollouts)]

        scores = [r.score for r in results]
        non_bust_royalties = [r.my_royalty for r in results if not r.busted]

        fl_type_counts = Counter()
        for r in results:
            if r.fl_qualified:
                fl_type_counts[r.fl_card_count] += 1

        n = len(results)
        return {
            'avg_score': sum(scores) / n,
            'std_score': (sum((s - sum(scores)/n)**2 for s in scores) / n) ** 0.5,
            'bust_prob': sum(1 for r in results if r.busted) / n,
            'fl_prob': sum(1 for r in results if r.fl_qualified) / n,
            'fl_type_dist': dict(fl_type_counts),
            'avg_royalty': sum(non_bust_royalties) / max(len(non_bust_royalties), 1),
            'opp_bust_prob': sum(1 for r in results if r.opp_busted) / n,
            'n_rollouts': n,
        }

    def _bc_playout_turn(
        self,
        board: Board,
        opp_board: Board,
        cards: List[str],
        turn: int,
        discards: List[str],
        is_btn: bool,
    ):
        """BC policy-guided playout with temperature sampling.

        Higher fidelity than rule-based playout for FL pursuit modeling.
        """
        if board.is_complete():
            return

        # T4 exhaustive: last placement turn, enumerate all
        if board.card_count() == 11:
            self._exhaustive_t4_turn(board, opp_board, cards, discards)
            return

        valid_actions = (
            get_initial_actions(cards, board)
            if board.card_count() == 0
            else get_turn_actions(cards, board)
        )
        if not valid_actions:
            return

        if len(valid_actions) == 1:
            chosen = valid_actions[0]
        else:
            obs = Observation(
                board_self=board,
                board_opponent=opp_board,
                dealt_cards=cards,
                known_discards_self=discards,
                turn=turn,
                is_btn=is_btn,
            )
            state_vec = encode_state(obs)
            state_t = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
            mask = create_action_mask(valid_actions)
            mask_t = torch.BoolTensor(mask).unsqueeze(0).to(self.device)

            # Use per-turn BC model if available, else default policy
            model = self.per_turn_bc.get(turn, self.policy_net)

            with torch.no_grad():
                probs = model(state_t, mask_t).squeeze(0).cpu()

            # Sort actions by BC probability (descending) and pick first safe one
            n_valid = len(valid_actions)
            logits = torch.log(probs[:n_valid] + 1e-8)
            scaled = logits / 0.8  # temperature
            sampled_probs = torch.softmax(scaled, dim=0)

            # Rank actions by probability
            sorted_indices = torch.argsort(sampled_probs, descending=True).tolist()

            chosen = None
            n_filtered = 0
            for idx in sorted_indices:
                if idx >= n_valid:
                    continue
                action = valid_actions[idx]
                if self._action_is_safe(board, action):
                    chosen = action
                    break
                n_filtered += 1

            # Fallback: if all bust, use BC top-1
            if chosen is None:
                chosen = valid_actions[sorted_indices[0]]
                # Debug: uncomment to see filter stats
                # import sys; print(f"  [BUST_FILTER] T{turn} ALL_UNSAFE n={n_valid} cc={board.card_count()}", file=sys.stderr)

        for card, pos in chosen.placements:
            getattr(board, pos).append(card)
        if chosen.discard:
            discards.append(chosen.discard)

    def _ppo_playout_turn(
        self,
        board: Board,
        opp_board: Board,
        cards: List[str],
        turn: int,
        discards: List[str],
        is_btn: bool,
    ):
        """PPO-based stochastic playout turn with temperature control.

        Uses PPO's policy logits with temperature scaling for diverse sampling.
        Temperature < 1.0 = more deterministic, > 1.0 = more random.
        """
        if board.is_complete():
            return

        valid_actions = (
            get_initial_actions(cards, board)
            if board.card_count() == 0
            else get_turn_actions(cards, board)
        )
        if not valid_actions:
            return

        if len(valid_actions) == 1:
            chosen = valid_actions[0]
        else:
            # Construct Observation for PPO
            obs = Observation(
                board_self=board,
                board_opponent=opp_board,
                dealt_cards=cards,
                known_discards_self=discards,
                turn=turn,
                is_btn=is_btn,
            )
            state_vec = encode_state(obs)

            # Get logits from PPO policy
            ppo_device = next(self.ppo_model.policy.parameters()).device
            state_t = torch.FloatTensor(state_vec).unsqueeze(0).to(ppo_device)
            with torch.no_grad():
                features = self.ppo_model.policy.extract_features(
                    state_t, self.ppo_model.policy.features_extractor
                )
                latent_pi, _ = self.ppo_model.policy.mlp_extractor(features)
                logits = self.ppo_model.policy.action_net(latent_pi).squeeze(0).cpu()

            # Apply action mask (set invalid actions to -inf)
            n_valid = len(valid_actions)
            masked_logits = logits[:n_valid].clone()

            # Temperature scaling
            temp = self.ppo_temperature
            scaled_logits = masked_logits / temp

            # Sample from distribution
            probs = torch.softmax(scaled_logits, dim=0)
            action_idx = torch.multinomial(probs, 1).item()

            if action_idx < n_valid:
                chosen = valid_actions[action_idx]
            else:
                chosen = valid_actions[0]

        # Apply action and track discard
        for card, pos in chosen.placements:
            getattr(board, pos).append(card)
        if chosen.discard:
            discards.append(chosen.discard)

    def _exhaustive_t4_turn(
        self,
        board: Board,
        opp_board: Board,
        cards: List[str],
        discards: List[str] = None,
    ):
        """T4 exhaustive: enumerate all placements, pick best by exact scoring."""
        if board.is_complete():
            return
        valid_actions = get_turn_actions(cards, board)
        if not valid_actions:
            return
        if len(valid_actions) == 1:
            chosen = valid_actions[0]
        else:
            best_score = float('-inf')
            chosen = valid_actions[0]
            for action in valid_actions:
                test = board.copy()
                for card, pos in action.placements:
                    getattr(test, pos).append(card)
                score = self._compute_score(test, opp_board)
                if score > best_score:
                    best_score = score
                    chosen = action
        for card, pos in chosen.placements:
            getattr(board, pos).append(card)
        if discards is not None and chosen.discard:
            discards.append(chosen.discard)

    def _do_playout_turn(
        self,
        board: Board,
        opp_board: Board,
        cards: List[str],
        turn: int,
        discards: List[str] = None,
        is_btn: bool = True,
    ):
        """Unified playout dispatcher: T4 exhaustive, per-turn BC, or rule-based."""
        if board.card_count() == 11:
            self._exhaustive_t4_turn(board, opp_board, cards, discards)
        elif board.card_count() == 0:
            # A BB root can leave the BTN T0 response pending.  Initial
            # placement needs five cards and the initial action space.
            self._bc_playout_turn(
                board,
                opp_board,
                cards,
                turn,
                discards if discards is not None else [],
                is_btn,
            )
        elif self.use_policy_playout or self.per_turn_bc:
            self._bc_playout_turn(board, opp_board, cards, turn,
                                  discards if discards is not None else [], is_btn)
        else:
            self._rule_playout_turn(board, opp_board, cards, turn)

    @staticmethod
    def _check_fl_priority(board: Board, valid_actions: List[Action]) -> Optional[Action]:
        """If top has A/K/Q with open slots and hand has matching rank, prefer that action.

        Only fires when top has exactly 1 card that is A/K/Q (or joker),
        and an action places the matching rank into top.
        """
        top = board.top
        if len(top) != 1:  # Only when 1 card in top (2 slots left)
            return None

        # Get rank of existing top card
        top_card = top[0]
        if top_card.startswith('X'):
            target_rank = None  # Joker: accept any A/K/Q to pair
        elif len(top_card) >= 2:
            target_rank = top_card[:-1]
            if target_rank not in ('A', 'K', 'Q'):
                return None  # Not a FL-capable rank
        else:
            return None

        # Find action that places matching rank to top
        for action in valid_actions:
            for card, pos in action.placements:
                if pos == 'top':
                    if card.startswith('X'):
                        # Joker to top pairs with anything
                        return action
                    card_rank = card[:-1] if len(card) >= 2 else None
                    if target_rank is None:
                        # Existing is joker: any A/K/Q to top forms pair
                        if card_rank in ('A', 'K', 'Q'):
                            return action
                    elif card_rank == target_rank:
                        return action
        return None

    # Card strength for rule-based playout (higher = stronger)
    _RANK_STRENGTH = {
        '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
        '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14,
    }

    # ------------------------------------------------------------------
    #  Helper methods for RYO's rule-based playout
    # ------------------------------------------------------------------

    def _card_rank_value(self, card: str) -> int:
        """Card rank value: 2..A=2..14, Joker=-1."""
        if card.startswith('X'):
            return -1
        return self._RANK_STRENGTH.get(card[0], 0)

    def _is_fl_mode(self, board: Board) -> bool:
        """True if Top has 1-2 cards and at least one is A/K/Joker."""
        if len(board.top) == 0 or len(board.top) >= 3:
            return False
        for c in board.top:
            if c.startswith('X') or c[0] in ('A', 'K'):
                return True
        return False

    def _top_needs_pair(self, board: Board) -> Optional[str]:
        """Return which rank Top needs to complete FL pair.

        Returns 'A', 'K', 'X' (joker on top), or None (pair done).
        """
        if len(board.top) >= 3:
            return None

        top_ranks = []
        has_joker = False
        for c in board.top:
            if c.startswith('X'):
                has_joker = True
            else:
                top_ranks.append(c[0])

        # Already have pair?
        from collections import Counter
        rank_counts = Counter(top_ranks)
        for r, cnt in rank_counts.items():
            if cnt >= 2:
                return None
            if cnt >= 1 and has_joker:
                return None

        # Need another card to pair
        if 'A' in top_ranks:
            return 'A'
        if 'K' in top_ranks:
            return 'K'
        if has_joker:
            return 'X'  # Joker on top, need A or K
        return None

    def _makes_pair_in_row(self, card: str, row_cards: list) -> bool:
        """Check if placing card would make a pair in the row."""
        if card.startswith('X'):
            return len(row_cards) > 0  # Joker pairs with anything
        card_rank = card[0]
        for c in row_cards:
            if c.startswith('X'):
                return True
            if c[0] == card_rank:
                return True
        return False

    # ------------------------------------------------------------------
    #  Rule-based playout turn (replaces BC-based _policy_playout_turn)
    # ------------------------------------------------------------------

    def _rule_playout_turn(
        self,
        board: Board,
        opp_board: Board,
        cards: List[str],
        turn: int,
    ):
        """Rule-based playout using RYO's strategy.

        Turn 1-8 rules:
          0. HARD FILTER: remove bust-creating actions
          1. FL mode: A/K/Joker → Top (give up at Turn 3+ if pair incomplete)
          2. Make pairs: Bot first, then Mid
          3. Strong cards → Bot, weak → Mid
          4. Never bust (Bot >= Mid >= Top)
        """
        if board.is_complete():
            return

        valid_actions = get_turn_actions(cards, board)
        if not valid_actions:
            return

        if len(valid_actions) == 1:
            chosen = valid_actions[0]
        else:
            # ── HARD BUST FILTER: remove actions that create bust ──
            safe_actions = []
            for action in valid_actions:
                test = board.copy()
                for card, pos in action.placements:
                    getattr(test, pos).append(card)
                is_safe = True
                # Complete-row checks use the same Joker constraints as final
                # scoring.  Independent row maxima can incorrectly reject a
                # legal QQX/KK/... placement.
                if not self._completed_row_order_is_safe(test):
                    is_safe = False
                # Partial row bust check
                if (is_safe
                        and len(test.middle) >= 2 and len(test.bottom) >= 2
                        and not (len(test.middle) == 5 and len(test.bottom) == 5)):
                    if self._hand_strength(test.middle) > self._hand_strength(test.bottom):
                        is_safe = False
                if (is_safe
                        and len(test.top) >= 2 and len(test.middle) >= 2
                        and not (len(test.top) == 3 and len(test.middle) == 5)):
                    if self._hand_strength(test.top) > self._hand_strength(test.middle):
                        is_safe = False
                if is_safe:
                    safe_actions.append(action)

            # Use safe actions if available, else fall back to all
            candidates = safe_actions if safe_actions else valid_actions

            best_score = -9999
            chosen = candidates[0]

            fl_mode = self._is_fl_mode(board)
            needed = self._top_needs_pair(board) if fl_mode else None

            # Give up FL at turn 7+ if pair not complete (only 1 turn left)
            if fl_mode and turn >= 7 and needed is not None:
                fl_mode = False
                needed = None

            for action in candidates:
                score = self._score_ryo_action(
                    board, action, turn, fl_mode, needed
                )
                if score > best_score:
                    best_score = score
                    chosen = action

        # Apply action
        for card, pos in chosen.placements:
            getattr(board, pos).append(card)

    @staticmethod
    def _hand_strength(cards: List[str]) -> tuple:
        """Return (pair_count, max_rank) for bust comparison.

        For bust ordering: Bot >= Mid >= Top.
        A higher tuple = stronger hand.
        pair_count is PRIMARY (pair always > high card).
        max_rank is SECONDARY (among same pair_count).
        """
        if not cards:
            return (0, 0)
        ranks = []
        joker_count = 0
        for c in cards:
            if c.startswith('X'):
                joker_count += 1
            else:
                r = c[0]
                rv = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,
                      'T':10,'J':11,'Q':12,'K':13,'A':14}.get(r, 0)
                ranks.append(rv)

        if not ranks:
            return (0, 0)

        from collections import Counter
        rank_counts = Counter(ranks)
        pair_count = sum(1 for c in rank_counts.values() if c >= 2)

        # Joker can form a pair with any existing card
        if joker_count > 0 and pair_count == 0 and ranks:
            pair_count = 1

        return (pair_count, max(ranks))

    def _action_is_safe(self, board: Board, action: Action) -> bool:
        """Check if an action would create a definite or likely bust.

        Same logic as the HARD BUST FILTER in _rule_playout_turn.
        Returns True if the action is safe (no bust detected).
        """
        test = board.copy()
        for card, pos in action.placements:
            getattr(test, pos).append(card)

        # Complete row bust check (definitive under canonical Joker rules).
        if not self._completed_row_order_is_safe(test):
            return False

        # Partial row bust check (heuristic)
        if (len(test.middle) >= 2 and len(test.bottom) >= 2
                and not (len(test.middle) == 5 and len(test.bottom) == 5)):
            if self._hand_strength(test.middle) > self._hand_strength(test.bottom):
                return False
        if (len(test.top) >= 2 and len(test.middle) >= 2
                and not (len(test.top) == 3 and len(test.middle) == 5)):
            if self._hand_strength(test.top) > self._hand_strength(test.middle):
                return False

        return True

    @staticmethod
    def _completed_row_order_is_safe(board: Board) -> bool:
        """Check only row-order violations that are already definitive."""
        middle_complete = len(board.middle) == 5
        bottom_complete = len(board.bottom) == 5
        top_complete = len(board.top) == 3

        if middle_complete and bottom_complete:
            bottom_value = evaluate_hand(board.bottom, 5)
            _middle, middle_value = evaluate_row_with_joker_constraint(
                board.middle, 5, bottom_value
            )
            if middle_value > bottom_value:
                return False

        if top_complete and middle_complete:
            if bottom_complete:
                return not bool(evaluate_board_with_joker_constraint(
                    board.top, board.middle, board.bottom
                )["busted"])

            # With an unfinished Bottom, a Joker in Top/Middle may still be
            # downgraded later.  Only a natural ordering violation is final.
            has_joker = any(
                card in ("X1", "X2", "JK")
                for card in (*board.top, *board.middle)
            )
            if not has_joker and evaluate_hand(board.top, 3) > evaluate_hand(board.middle, 5):
                return False

        return True

    def _score_ryo_action(
        self, board: Board, action: Action, turn: int,
        fl_mode: bool, needed: Optional[str]
    ) -> float:
        """Score an action based on RYO's rules. Higher = better.

        Priority order:
          ① BUST PREVENTION — absolute veto via hand strength comparison
          ② Row-fill balance — ensure Mid/Bot develop evenly
          ③ FL completion (+50) — pursue Fantasyland
          ④ Pair-making (+60 ~ +80) — build hand strength
          ⑤ Strength routing — strong→Bot, weak→Mid
        """
        score = 0.0

        # ──────────────────────────────────────────────
        # ① BUST CHECK: simulate action, compare hand strengths
        # ──────────────────────────────────────────────
        test_board = board.copy()
        for card, pos in action.placements:
            getattr(test_board, pos).append(card)

        t_len = len(test_board.top)
        m_len = len(test_board.middle)
        b_len = len(test_board.bottom)

        # Definitive bust check for complete rows using canonical Joker rules.
        if not self._completed_row_order_is_safe(test_board):
            score -= 1000

        # Partial-row bust check using _hand_strength (pair_count, max_rank)
        if m_len >= 2 and b_len >= 2 and not (m_len == 5 and b_len == 5):
            mid_str = self._hand_strength(test_board.middle)
            bot_str = self._hand_strength(test_board.bottom)
            if mid_str > bot_str:
                penalty = -100 - 100 * (min(m_len, b_len) - 2)  # -100 to -300
                score += penalty

        if t_len >= 2 and m_len >= 2 and not (t_len == 3 and m_len == 5):
            top_str = self._hand_strength(test_board.top)
            mid_str2 = self._hand_strength(test_board.middle)
            if top_str > mid_str2:
                penalty = -100 - 100 * (min(t_len, m_len) - 2)
                score += penalty

        # ──────────────────────────────────────────────
        # ② ROW-FILL BALANCE: prevent empty Mid
        # ──────────────────────────────────────────────
        bot_mid_diff = b_len - m_len
        if bot_mid_diff >= 3:
            score -= 50
        elif bot_mid_diff >= 2:
            score -= 25

        for card, pos in action.placements:
            if pos == 'middle' and m_len < b_len:
                score += 15
            elif pos == 'bottom' and b_len < m_len:
                score += 15

        # ──────────────────────────────────────────────
        # ③④⑤ FL / Pair / Strength scoring
        # ──────────────────────────────────────────────
        for card, pos in action.placements:
            is_joker = card.startswith('X')
            rank = card[0] if not is_joker else 'X'
            rank_val = self._card_rank_value(card)

            # --- FL rules ---
            if fl_mode and pos == 'top':
                if needed == 'A' and (rank == 'A' or is_joker):
                    score += 50
                elif needed == 'K' and (rank == 'K' or is_joker):
                    score += 50
                elif needed == 'X' and rank in ('A', 'K'):
                    score += 50
                elif is_joker:
                    score += 40
                elif rank in ('A', 'K'):
                    score += 30
                else:
                    score -= 30

            elif not fl_mode and pos == 'top':
                if len(board.top) < 3:
                    if is_joker or rank in ('A', 'K'):
                        score += 5
                    else:
                        score -= 5

            # --- Pair-making ---
            if pos == 'bottom':
                if self._makes_pair_in_row(card, board.bottom):
                    score += 80
                score += rank_val * 1.5

            if pos == 'middle':
                if self._makes_pair_in_row(card, board.middle):
                    score += 60
                score += rank_val * 1

            # --- Early turn joker placement ---
            if is_joker and turn <= 2:
                if pos == 'middle':
                    score -= 10
                elif pos == 'top' and fl_mode:
                    score += 10
                elif pos == 'bottom':
                    score += 5

        return score

    # FL Expected Value from config
    FL_EV = _load_fl_ev()

    @staticmethod
    def _check_fl_cards(top_cards) -> int:
        """Return FL card count (14-17) or 0 if not FL-qualifying."""
        if len(top_cards) < 3:
            return 0
        ranks, jokers = [], 0
        for c in top_cards:
            if c in ("X1", "X2"):
                jokers += 1
            elif len(c) >= 2:
                ranks.append(c[:-1])
        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1
        for r, count in rank_counts.items():
            if count + jokers >= 3:
                return 17
        rank_vals = {'Q': 14, 'K': 15, 'A': 16}
        best = 0
        for r, count in rank_counts.items():
            if r in rank_vals and count + jokers >= 2:
                best = max(best, rank_vals[r])
        return best

    @staticmethod
    def _fl_partial_bonus(top_cards) -> float:
        """Partial FL bonus for incomplete top row (< 3 cards).

        When top has 1-2 cards including A/K/Q/Joker, add a fraction of FL_EV
        based on estimated pair completion probability.

        Estimates:
          1 card in top, 2 slots remaining:
            - A alone: ~30% chance to pair (3 remaining A's in ~30 unseen cards)
            - K alone: ~30%
            - Q alone: ~30%
            - Joker: treat as A potential (~30%)
          2 cards in top, 1 slot remaining:
            - A+x (no pair): ~10% chance (1 slot, need specific rank)
            - Already have pair but < QQ: no bonus
        """
        if len(top_cards) >= 3 or not top_cards:
            return 0.0

        RANK_MAP = {'A': 16, 'K': 15, 'Q': 14}
        fl_ev = RolloutEvaluator.FL_EV

        ranks = []
        jokers = 0
        for c in top_cards:
            if c in ('X1', 'X2'):
                jokers += 1
            elif len(c) >= 2:
                ranks.append(c[:-1])

        slots = 3 - len(top_cards)

        if len(top_cards) == 1:
            # 1 card placed, 2 slots remain
            # High card or joker: decent chance to pair over 2 more draws
            if jokers == 1:
                # Joker can pair with anything → treat as AA potential
                return fl_ev.get(16, 66.3) * 0.25
            if ranks and ranks[0] in RANK_MAP:
                ev_key = RANK_MAP[ranks[0]]
                return fl_ev.get(ev_key, 0) * 0.20
            return 0.0

        if len(top_cards) == 2:
            # 2 cards placed, 1 slot remains
            from collections import Counter
            rank_counts = Counter(ranks)

            # Already have a pair?
            pairs = [(r, c) for r, c in rank_counts.items() if c >= 2]
            if pairs:
                best_pair = max((RANK_MAP.get(r, 0), r) for r, _ in pairs)
                if best_pair[0] >= 14:  # QQ+
                    # Already FL-ready with 1 slot left, very likely to complete
                    return fl_ev.get(best_pair[0], 0) * 0.85
                return 0.0  # Pair but below QQ

            # Joker + high card?
            if jokers >= 1 and ranks:
                r = ranks[0]
                if r in RANK_MAP:
                    # Joker + A/K/Q → already a pair, just need any 3rd card
                    return fl_ev.get(RANK_MAP[r], 0) * 0.85
                return 0.0

            # Two jokers?
            if jokers >= 2:
                return fl_ev.get(16, 66.3) * 0.85  # AA equivalent

            # Two different cards, 1 slot: need to pair one of them
            # ~6% chance per high card (1 slot, ~3 outs in ~20 unseen)
            bonus = 0.0
            for r in ranks:
                if r in RANK_MAP:
                    bonus = max(bonus, fl_ev.get(RANK_MAP[r], 0) * 0.08)
            return bonus

        return 0.0

    @staticmethod
    def _compute_score(my_board: Board, opp_board: Board) -> float:
        """Compute final score from my perspective (line comparison + royalties + FL EV)."""
        my_eval = evaluate_board_with_joker_constraint(
            my_board.top, my_board.middle, my_board.bottom
        )
        opp_eval = evaluate_board_with_joker_constraint(
            opp_board.top, opp_board.middle, opp_board.bottom
        )
        my_vals = my_eval["values"]
        opp_vals = opp_eval["values"]
        my_busted = bool(my_eval["busted"])
        opp_busted = bool(opp_eval["busted"])
        my_royalty = int(my_eval["royalties"]["total"])
        opp_royalty = int(opp_eval["royalties"]["total"])

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

        # FL Expected Value bonus (completed top row)
        my_fl = int(my_eval["fl_card_count"])
        opp_fl = int(opp_eval["fl_card_count"])
        if not my_busted and my_fl > 0:
            score += RolloutEvaluator.FL_EV.get(my_fl, 0)
        if not opp_busted and opp_fl > 0:
            score -= RolloutEvaluator.FL_EV.get(opp_fl, 0)

        # FL partial bonus (incomplete top row with FL potential)
        if not my_busted and my_fl == 0:
            score += RolloutEvaluator._fl_partial_bonus(my_board.top)
        if not opp_busted and opp_fl == 0:
            score -= RolloutEvaluator._fl_partial_bonus(opp_board.top)

        return float(score)

    @staticmethod
    def compute_score_raw(my_board: Board, opp_board: Board) -> float:
        """Score without FL EV bonus. For self-play where FL is actually played."""
        my_eval = evaluate_board_with_joker_constraint(
            my_board.top, my_board.middle, my_board.bottom
        )
        opp_eval = evaluate_board_with_joker_constraint(
            opp_board.top, opp_board.middle, opp_board.bottom
        )
        my_vals = my_eval["values"]
        opp_vals = opp_eval["values"]
        my_busted = bool(my_eval["busted"])
        opp_busted = bool(opp_eval["busted"])
        my_royalty = int(my_eval["royalties"]["total"])
        opp_royalty = int(opp_eval["royalties"]["total"])
        if my_busted and opp_busted:
            return 0.0
        if my_busted:
            return float(-6 - opp_royalty)
        if opp_busted:
            return float(6 + my_royalty)
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
