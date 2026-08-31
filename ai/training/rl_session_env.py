"""Real-game session environment for RL data collection.

This module is intentionally policy-agnostic.  It owns the session rules,
score capping, Fantasyland transitions, and decision logging; agents only
choose actions from legal actions.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ai.engine.action_space import (
    Action,
    create_regular_turn_mask,
    get_initial_actions,
    get_semantic_action_index,
    get_turn_actions,
)
from ai.engine.encoding import ALL_CARDS, Board, Observation, encode_state
from ai.engine.game_engine import (
    GameEngine,
    Hand,
    HandResult,
    check_fl_entry,
    evaluate_hand,
    evaluate_board_with_joker_constraint,
    get_bottom_royalty,
    get_middle_royalty,
    get_top_royalty,
)
from ai.engine.scoring import check_fl_stay_from_cards
from ai.engine.turn_order import action_order
from ai.training.generate_data import heuristic_score


def board_to_dict(board: Board) -> Dict[str, List[str]]:
    return {
        "top": list(board.top),
        "middle": list(board.middle),
        "bottom": list(board.bottom),
    }


def action_to_dict(action: Action) -> Dict[str, Any]:
    return {
        "placements": [{"card": card, "pos": pos} for card, pos in action.placements],
        "discard": action.discard,
    }


def action_index(action: Action, actions: Sequence[Action], obs: Observation) -> int:
    if obs.turn == 0:
        return int(actions.index(action))
    return int(get_semantic_action_index(action, obs.dealt_cards))


def legal_mask_for(obs: Observation, actions: Sequence[Action]) -> List[bool]:
    if obs.turn == 0:
        return [True] * len(actions)
    return create_regular_turn_mask(obs.dealt_cards, obs.board_self).astype(bool).tolist()


@dataclass
class AgentDecision:
    action: Action
    action_index: int
    candidate_actions: List[int] = field(default_factory=list)
    policy_logits: Optional[List[float]] = None
    policy_probs: Optional[List[float]] = None
    value_estimates: Optional[List[float]] = None
    info: Dict[str, Any] = field(default_factory=dict)


class Agent:
    """Policy interface used by the RL session environment."""

    def choose_action(
        self,
        obs: Observation,
        actions: Sequence[Action],
        legal_mask: Sequence[bool],
    ) -> AgentDecision:
        raise NotImplementedError


class HeuristicAgent(Agent):
    """Fallback agent that evaluates multiple legal actions with a heuristic."""

    def __init__(self, top_k: int = 5, temperature: float = 0.2):
        self.top_k = top_k
        self.temperature = temperature

    def choose_action(
        self,
        obs: Observation,
        actions: Sequence[Action],
        legal_mask: Sequence[bool],
    ) -> AgentDecision:
        if not actions:
            raise ValueError("No legal actions")

        scored: List[Tuple[float, int, Action]] = []
        for action in actions:
            idx = action_index(action, actions, obs)
            scored.append((heuristic_score(action, obs.board_self), idx, action))
        scored.sort(key=lambda x: x[0], reverse=True)
        candidates = scored[: max(1, min(self.top_k, len(scored)))]

        if self.temperature <= 0 or len(candidates) == 1:
            chosen_score, chosen_idx, chosen_action = candidates[0]
            probs = [1.0 if i == 0 else 0.0 for i in range(len(candidates))]
        else:
            logits = np.array([s for s, _, _ in candidates], dtype=np.float64)
            logits = logits / max(self.temperature, 1e-6)
            logits -= logits.max()
            probs_np = np.exp(logits)
            probs_np /= probs_np.sum()
            pick = int(np.random.choice(len(candidates), p=probs_np))
            chosen_score, chosen_idx, chosen_action = candidates[pick]
            probs = probs_np.tolist()

        return AgentDecision(
            action=chosen_action,
            action_index=int(chosen_idx),
            candidate_actions=[int(idx) for _, idx, _ in candidates],
            policy_logits=[float(s) for s, _, _ in candidates],
            policy_probs=[float(p) for p in probs],
            value_estimates=[float(s) for s, _, _ in candidates],
            info={"agent": "heuristic", "chosen_score": float(chosen_score)},
        )


@dataclass
class SessionState:
    starting_chips: int = 200
    finish_gap: int = 40
    chips: List[int] = field(default_factory=lambda: [200, 200])
    btn: int = 0
    is_fl: List[bool] = field(default_factory=lambda: [False, False])
    fl_card_count: List[int] = field(default_factory=lambda: [0, 0])
    hand_count: int = 0
    fl_hands: int = 0
    next_fl_chain_id: int = 1
    fl_chain_id: List[Optional[int]] = field(default_factory=lambda: [None, None])
    fl_chain_scores: Dict[int, List[int]] = field(default_factory=dict)

    @classmethod
    def new(cls, starting_chips: int = 200, finish_gap: int = 40) -> "SessionState":
        return cls(
            starting_chips=starting_chips,
            finish_gap=finish_gap,
            chips=[starting_chips, starting_chips],
            btn=random.randint(0, 1),
        )

    def score_gap(self, seat: int = 0) -> int:
        return int(self.chips[seat] - self.chips[1 - seat])

    def should_end_session(self) -> bool:
        return (not self.is_fl[0] and not self.is_fl[1]
                and abs(self.chips[0] - self.chips[1]) >= self.finish_gap)

    def apply_capped_score(self, raw_score: Sequence[int]) -> List[int]:
        """Apply zero-floor, conserved-chip scoring and return actual deltas."""
        if raw_score[0] > raw_score[1] and raw_score[0] > 0:
            transfer = min(int(raw_score[0]), int(self.chips[1]))
            delta = [transfer, -transfer]
        elif raw_score[1] > raw_score[0] and raw_score[1] > 0:
            transfer = min(int(raw_score[1]), int(self.chips[0]))
            delta = [-transfer, transfer]
        else:
            delta = [0, 0]

        self.chips[0] += delta[0]
        self.chips[1] += delta[1]
        self.chips[0] = max(0, self.chips[0])
        self.chips[1] = max(0, self.chips[1])
        return delta

    def advance_button(self) -> None:
        self.btn = 1 - self.btn
        self.hand_count += 1

    def start_fl_chain(self, seat: int, card_count: int) -> None:
        chain_id = self.next_fl_chain_id
        self.next_fl_chain_id += 1
        self.is_fl[seat] = True
        self.fl_card_count[seat] = int(card_count)
        self.fl_chain_id[seat] = chain_id
        self.fl_chain_scores[chain_id] = []

    def close_fl_chain(self, seat: int) -> Optional[int]:
        chain_id = self.fl_chain_id[seat]
        self.is_fl[seat] = False
        self.fl_card_count[seat] = 0
        self.fl_chain_id[seat] = None
        return chain_id


def _make_observation(hand: Hand, state: SessionState, seat: int) -> Observation:
    obs = hand.get_observation(seat)
    obs.is_fl = state.is_fl[seat]
    obs.opp_is_fl = state.is_fl[1 - seat]
    obs.chips_self = state.chips[seat]
    obs.chips_opponent = state.chips[1 - seat]
    return obs


def _actions_for(obs: Observation) -> List[Action]:
    if obs.turn == 0:
        return get_initial_actions(obs.dealt_cards, obs.board_self)
    return get_turn_actions(obs.dealt_cards, obs.board_self)


def _decision_record(
    session_id: int,
    hand_id: int,
    seat: int,
    obs: Observation,
    actions: Sequence[Action],
    decision: AgentDecision,
    capped_chips: Sequence[int],
) -> Dict[str, Any]:
    return {
        "session_id": int(session_id),
        "hand_id": int(hand_id),
        "seat": int(seat),
        "turn": int(obs.turn),
        "is_btn": bool(obs.is_btn),
        "is_fl": bool(obs.is_fl),
        "opp_is_fl": bool(obs.opp_is_fl),
        "chips_self": int(obs.chips_self),
        "chips_opp": int(obs.chips_opponent),
        "score_gap": int(obs.chips_self - obs.chips_opponent),
        "board_self": board_to_dict(obs.board_self),
        "board_opponent": board_to_dict(obs.board_opponent),
        "dealt_cards": list(obs.dealt_cards),
        "known_discards": list(obs.known_discards_self),
        "state": encode_state(obs).astype(float).tolist(),
        "legal_mask": legal_mask_for(obs, actions),
        "chosen_action": action_to_dict(decision.action),
        "chosen_action_index": int(decision.action_index),
        "candidate_actions": list(decision.candidate_actions),
        "policy_logits": decision.policy_logits,
        "policy_probs": decision.policy_probs,
        "value_estimates": decision.value_estimates,
        "agent_info": dict(decision.info),
        "chips": list(capped_chips),
        "hand_raw_score": None,
        "capped_score_delta": None,
        "session_return": None,
        "fl_entry": None,
        "fl_card_count": None,
        "fl_hand_score": None,
        "fl_stayed": None,
        "fl_chain_id": None,
        "fl_chain_total_score": None,
    }


def _apply_agent_action(
    hand: Hand,
    state: SessionState,
    agents: Sequence[Agent],
    seat: int,
    session_id: int,
    hand_id: int,
) -> Dict[str, Any]:
    obs = _make_observation(hand, state, seat)
    actions = _actions_for(obs)
    mask = legal_mask_for(obs, actions)
    decision = agents[seat].choose_action(obs, actions, mask)
    if decision.action not in actions:
        raise ValueError("Agent selected an action that is not legal")
    hand.apply_action(seat, decision.action)
    return _decision_record(session_id, hand_id, seat, obs, actions, decision, state.chips)


def play_normal_hand(
    state: SessionState,
    agents: Sequence[Agent],
    deck: Sequence[str],
    session_id: int,
    hand_id: int,
) -> Tuple[HandResult, List[Dict[str, Any]]]:
    hand = Hand(deck=list(deck), btn=state.btn, is_fl=[False, False])
    records: List[Dict[str, Any]] = []

    for seat in action_order(hand.btn):
        records.append(_apply_agent_action(hand, state, agents, seat, session_id, hand_id))

    while not hand.is_hand_complete():
        hand.deal_next_turn()
        for seat in action_order(hand.btn):
            if hand.boards[seat].is_complete():
                hand.placed[seat] = True
                continue
            records.append(_apply_agent_action(hand, state, agents, seat, session_id, hand_id))

    return GameEngine.compute_result(hand), records


def _row_royalty(board: Board) -> int:
    evaluated = evaluate_board_with_joker_constraint(
        board.top, board.middle, board.bottom
    )
    if evaluated["busted"]:
        return -10_000
    return int(evaluated["royalties"]["total"])


def _sample_fl_board(cards: Sequence[str]) -> Board:
    shuffled = list(cards)
    random.shuffle(shuffled)
    return Board(top=shuffled[:3], middle=shuffled[3:8], bottom=shuffled[8:13])


def _score_fl_board(board: Board, opp_board: Optional[Board], fl_seat: int) -> int:
    if opp_board is None:
        return _row_royalty(board)
    scoring = Hand(deck=[], btn=0)
    scoring.boards = [Board(), Board()]
    scoring.boards[fl_seat] = board
    scoring.boards[1 - fl_seat] = opp_board
    return int(GameEngine.compute_result(scoring).raw_score[fl_seat])


def choose_fl_board(
    cards: Sequence[str],
    opp_board: Optional[Board] = None,
    fl_seat: int = 0,
    fl_solver: Any = None,
    n_samples: int = 5000,
) -> Board:
    """Choose a valid FL placement, using a solver when available and sampling as fallback."""
    if fl_solver is not None:
        try:
            from ai.training.selfplay_session import card_str_to_tuple, placement_to_board

            placement = fl_solver.solve([card_str_to_tuple(c) for c in cards])
            if placement:
                return placement_to_board(placement)
        except Exception:
            pass

    best_board: Optional[Board] = None
    best_score = -math.inf
    for _ in range(max(1, n_samples)):
        board = _sample_fl_board(cards)
        if not board.is_complete():
            continue
        score = _score_fl_board(board, opp_board, fl_seat)
        if score > best_score:
            best_score = score
            best_board = board
    if best_board is None:
        best_board = _sample_fl_board(cards)
    return best_board


def _play_normal_for_one_seat(
    state: SessionState,
    agents: Sequence[Agent],
    deck: Sequence[str],
    normal_seat: int,
    session_id: int,
    hand_id: int,
) -> Tuple[Board, List[Dict[str, Any]]]:
    hand = Hand(deck=list(deck), btn=state.btn, is_fl=[False, False])
    records: List[Dict[str, Any]] = []
    dummy = HeuristicAgent(top_k=1, temperature=0.0)

    for seat in action_order(hand.btn):
        if seat == normal_seat:
            records.append(_apply_agent_action(hand, state, agents, seat, session_id, hand_id))
        else:
            obs = _make_observation(hand, state, seat)
            actions = _actions_for(obs)
            hand.apply_action(seat, dummy.choose_action(obs, actions, legal_mask_for(obs, actions)).action)

    while not hand.boards[normal_seat].is_complete():
        hand.deal_next_turn()
        for seat in action_order(hand.btn):
            if hand.boards[seat].is_complete():
                hand.placed[seat] = True
                continue
            if seat == normal_seat:
                records.append(_apply_agent_action(hand, state, agents, seat, session_id, hand_id))
            else:
                obs = _make_observation(hand, state, seat)
                actions = _actions_for(obs)
                if actions:
                    hand.apply_action(seat, dummy.choose_action(obs, actions, legal_mask_for(obs, actions)).action)

    return hand.boards[normal_seat], records


def play_fl_hand(
    state: SessionState,
    agents: Sequence[Agent],
    deck: Sequence[str],
    session_id: int,
    hand_id: int,
    fl_solver: Any = None,
    fl_samples: int = 5000,
) -> Tuple[HandResult, List[Dict[str, Any]], List[Dict[str, Any]]]:
    fl_events: List[Dict[str, Any]] = []
    boards = [Board(), Board()]
    deck = list(deck)

    if state.is_fl[0] and state.is_fl[1]:
        pos = 0
        for seat in [0, 1]:
            n = state.fl_card_count[seat]
            cards = deck[pos:pos + n]
            pos += n
            boards[seat] = choose_fl_board(cards, fl_solver=fl_solver, n_samples=fl_samples)
    else:
        fl_seat = 0 if state.is_fl[0] else 1
        normal_seat = 1 - fl_seat
        n = state.fl_card_count[fl_seat]
        fl_cards = deck[:n]
        normal_board, records = _play_normal_for_one_seat(
            state, agents, deck[n:], normal_seat, session_id, hand_id
        )
        boards[normal_seat] = normal_board
        boards[fl_seat] = choose_fl_board(
            fl_cards,
            opp_board=normal_board,
            fl_seat=fl_seat,
            fl_solver=fl_solver,
            n_samples=fl_samples,
        )
        scoring = Hand(deck=[], btn=state.btn)
        scoring.boards = boards
        return GameEngine.compute_result(scoring), records, fl_events

    scoring = Hand(deck=[], btn=state.btn)
    scoring.boards = boards
    return GameEngine.compute_result(scoring), [], fl_events


def _annotate_hand_records(
    records: List[Dict[str, Any]],
    result: HandResult,
    capped_delta: Sequence[int],
    pre_fl: Sequence[bool],
    pre_chain_ids: Sequence[Optional[int]],
    fl_stayed: Sequence[Optional[bool]],
) -> None:
    for rec in records:
        seat = rec["seat"]
        rec["hand_raw_score"] = int(result.raw_score[seat])
        rec["capped_score_delta"] = int(capped_delta[seat])
        rec["fl_entry"] = bool(result.fl_entry[seat])
        rec["fl_card_count"] = int(result.fl_card_count[seat])
        rec["fl_hand_score"] = int(result.raw_score[seat]) if pre_fl[seat] else None
        rec["fl_stayed"] = fl_stayed[seat]
        rec["fl_chain_id"] = pre_chain_ids[seat]


def _update_fl_state(
    state: SessionState,
    result: HandResult,
    pre_fl: Sequence[bool],
    pre_cc: Sequence[int],
    pre_chain_ids: Sequence[Optional[int]],
) -> Tuple[List[Optional[bool]], List[Dict[str, Any]]]:
    stayed: List[Optional[bool]] = [None, None]
    events: List[Dict[str, Any]] = []

    for seat in [0, 1]:
        if pre_fl[seat]:
            chain_id = pre_chain_ids[seat]
            if chain_id is not None:
                state.fl_chain_scores.setdefault(chain_id, []).append(int(result.raw_score[seat]))
            if result.busted[seat]:
                did_stay = False
            else:
                did_stay, _ = check_fl_stay_from_cards(
                    result.boards[seat].top,
                    result.boards[seat].bottom,
                    pre_cc[seat],
                    middle_cards=result.boards[seat].middle,
                )
            stayed[seat] = bool(did_stay)
            if not did_stay:
                total = sum(state.fl_chain_scores.get(chain_id, [])) if chain_id is not None else 0
                events.append({
                    "seat": seat,
                    "fl_chain_id": chain_id,
                    "fl_card_count": int(pre_cc[seat]),
                    "fl_stayed": False,
                    "fl_chain_total_score": int(total),
                })
                state.close_fl_chain(seat)
        elif result.fl_entry[seat]:
            state.start_fl_chain(seat, result.fl_card_count[seat])
            events.append({
                "seat": seat,
                "fl_chain_id": state.fl_chain_id[seat],
                "fl_card_count": int(result.fl_card_count[seat]),
                "fl_entry": True,
            })

    return stayed, events


def play_session(
    agents: Sequence[Agent],
    session_id: int = 0,
    starting_chips: int = 200,
    finish_gap: int = 40,
    max_hands: int = 200,
    fl_solver: Any = None,
    fl_samples: int = 5000,
    rng_seed: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if rng_seed is not None:
        random.seed(rng_seed)
        np.random.seed(rng_seed)
    if len(agents) != 2:
        raise ValueError("play_session expects exactly two agents")

    state = SessionState.new(starting_chips=starting_chips, finish_gap=finish_gap)
    records: List[Dict[str, Any]] = []
    fl_events: List[Dict[str, Any]] = []

    while not state.should_end_session() and state.hand_count < max_hands:
        deck = list(ALL_CARDS)
        random.shuffle(deck)
        pre_fl = list(state.is_fl)
        pre_cc = list(state.fl_card_count)
        pre_chain_ids = list(state.fl_chain_id)

        if state.is_fl[0] or state.is_fl[1]:
            state.fl_hands += 1
            result, hand_records, hand_fl_events = play_fl_hand(
                state, agents, deck, session_id, state.hand_count, fl_solver, fl_samples
            )
            fl_events.extend(hand_fl_events)
        else:
            result, hand_records = play_normal_hand(
                state, agents, deck, session_id, state.hand_count
            )

        capped_delta = state.apply_capped_score(result.raw_score)
        stayed, update_events = _update_fl_state(state, result, pre_fl, pre_cc, pre_chain_ids)
        fl_events.extend(update_events)
        _annotate_hand_records(hand_records, result, capped_delta, pre_fl, pre_chain_ids, stayed)
        records.extend(hand_records)
        state.advance_button()

    session_returns = [state.chips[0] - starting_chips, state.chips[1] - starting_chips]
    chain_totals = {cid: sum(scores) for cid, scores in state.fl_chain_scores.items()}
    for rec in records:
        rec["session_return"] = int(session_returns[rec["seat"]])
        cid = rec.get("fl_chain_id")
        if cid is not None:
            rec["fl_chain_total_score"] = int(chain_totals.get(cid, 0))

    summary = {
        "session_id": int(session_id),
        "hands": int(state.hand_count),
        "fl_hands": int(state.fl_hands),
        "final_chips": list(state.chips),
        "session_return": list(session_returns),
        "btn_next": int(state.btn),
        "ended_by_gap": bool(state.should_end_session()),
        "max_hands_reached": bool(state.hand_count >= max_hands),
        "fl_events": fl_events,
        "fl_chain_totals": {int(k): int(v) for k, v in chain_totals.items()},
    }
    return records, summary


__all__ = [
    "Agent",
    "AgentDecision",
    "HeuristicAgent",
    "SessionState",
    "play_session",
    "play_normal_hand",
    "play_fl_hand",
    "choose_fl_board",
]
