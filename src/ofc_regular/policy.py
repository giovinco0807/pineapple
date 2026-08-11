"""Playable policies for regular OFC Pineapple."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from .action_key import (
    ACTION_KEY_SCHEMA,
    action_key,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from .action_space import Action, generate_actions, generate_turn_actions
from .cards import ALL_CARDS, validate_cards
from .hu_turn3_gate_model import HuTurn3GateModel
from .hu_infoset import ActorObservation, card_free_metadata
from .hu_turn3_model import hu_policy_sample
from .state import Board
from .teacher import DEFAULT_FL_EV, evaluate_turn_actions
from .turn3_model import Turn3RidgeModel


@dataclass
class RegularAiPolicy:
    """Baseline playable AI.

    Current behavior:
    - opening: exact legal-placement enumeration. If an opening action-value
      model is provided, it is used directly. Otherwise, if Turn1 model is
      provided, each opening placement is evaluated by one-ply future sampling
      into Turn1.
    - 5-card board: Turn1 action-value model if provided, otherwise random
    - 7-card board: Turn2 action-value model if provided, otherwise random
    - 9-card board: Turn3 action-value model if provided, otherwise random
    - 11-card board: exact final-turn search
    """

    opening_model: Turn3RidgeModel | None = None
    hu_turn0_model: object | None = None
    hu_turn0_candidate_topk: int = 60
    hu_turn0_min_margin: float = 0.0
    hu_turn0_min_margin_by_seat: dict[str, float] | None = None
    hu_turn0_allowed_seats: tuple[str, ...] | None = None
    hu_turn0_safe_selector_model: object | None = None
    hu_turn0_safe_selector_enabled: bool = False
    hu_turn0_safe_selector_threshold: float = 0.0
    hu_turn0_safe_selector_threshold_by_seat: dict[str, float] | None = None
    hu_turn0_decision_log_path: str | Path | None = None
    hu_turn0_decision_log: list[dict[str, Any]] | None = None
    turn1_model: Turn3RidgeModel | None = None
    turn2_model: Turn3RidgeModel | None = None
    turn3_model: Turn3RidgeModel | None = None
    hu_turn1_model: object | None = None
    hu_turn1_min_margin: float | None = None
    hu_turn1_decision_log_path: str | Path | None = None
    hu_turn1_decision_log: list[dict[str, Any]] | None = None
    decision_context: dict[str, Any] = field(default_factory=dict)
    hu_turn3_model: object | None = None
    hu_turn3_reference_model: object | None = None
    hu_turn3_support_model: object | None = None
    hu_turn3_gate_model: HuTurn3GateModel | None = None
    hu_turn3_min_margin: float = 0.0
    hu_turn3_reference_min_margin: float = 0.0
    hu_turn3_min_support_margin: float = 0.0
    hu_turn3_min_model_score: float | None = None
    hu_turn3_allowed_seats: tuple[str, ...] | None = None
    hu_turn3_min_gate_probability: float = 0.0
    hu_turn3_max_self_regret: float | None = None
    hu_turn3_stage7_enabled: bool = True
    hu_turn3_decision_log_path: str | Path | None = None
    hu_turn3_decision_log: list[dict[str, Any]] | None = None
    seat: str = "first"
    seed: int = 42
    fl_ev: dict[int, float] | None = None
    opening_lookahead_samples: int = 64

    _CARD_FREE_CONTEXT_ATTRIBUTES = frozenset(
        {"decision_context", "hu_turn2_context", "topk_context"}
    )

    def __setattr__(self, name: str, value: Any) -> None:
        # Subclasses add T1/T2 context attributes after the dataclass
        # constructor.  Guard assignment itself so a later plain-dict
        # replacement cannot bypass the information-set boundary.
        if name in self._CARD_FREE_CONTEXT_ATTRIBUTES:
            value = card_free_metadata(value)
        super().__setattr__(name, value)

    def __post_init__(self) -> None:
        self.decision_context = card_free_metadata(self.decision_context)
        self.rng = random.Random(self.seed)
        if self.fl_ev is None:
            self.fl_ev = dict(DEFAULT_FL_EV)
        if self.opening_lookahead_samples < 0:
            raise ValueError("opening_lookahead_samples must be non-negative")
        if self.hu_turn0_candidate_topk <= 0:
            raise ValueError("hu_turn0_candidate_topk must be positive")
        selector_thresholds = [self.hu_turn0_safe_selector_threshold]
        if self.hu_turn0_safe_selector_threshold_by_seat is not None:
            selector_thresholds.extend(self.hu_turn0_safe_selector_threshold_by_seat.values())
        if any(not 0.0 <= float(value) <= 1.0 for value in selector_thresholds):
            raise ValueError("hu_turn0_safe_selector_threshold must be in [0, 1]")

    def choose_action_observation(
        self,
        observation: ActorObservation,
        *,
        hand_id: str | int | None = None,
        game_id: str | int | None = None,
        decision_seed: int | None = None,
    ) -> Action:
        """Choose through the policy-safe observation boundary.

        The legacy ``dead_cards`` adapter contains only the opponent's public
        board and this actor's own private discards.
        """
        if observation.seat != self.seat:
            raise ValueError(
                f"observation seat {observation.seat!r} does not match policy seat {self.seat!r}"
            )
        return self.choose_action(
            observation.hero_board,
            observation.dealt_cards,
            dead_cards=observation.legacy_dead_cards(),
            opponent_board=observation.opponent_public_board,
            hand_id=hand_id,
            game_id=game_id,
            decision_seed=decision_seed,
            street=observation.street,
        )

    def choose_action(
        self,
        board: Board,
        dealt_cards: Iterable[str],
        *,
        dead_cards: Iterable[str] = (),
        opponent_board: Board | None = None,
        hand_id: str | int | None = None,
        game_id: str | int | None = None,
        decision_seed: int | None = None,
        street: str | None = None,
    ) -> Action:
        dealt = tuple(dealt_cards)
        dead = tuple(dead_cards)
        decision_rng = (
            random.Random(decision_seed) if decision_seed is not None else self.rng
        )
        if board.card_count() == 0:
            actions = generate_actions(board, dealt)
            if (
                actions
                and self.hu_turn0_model is not None
                and self.opening_model is not None
                and opponent_board is not None
            ):
                action = self._choose_hu_turn0_action(
                    board,
                    dealt,
                    actions=actions,
                    dead_cards=dead,
                    opponent_board=opponent_board,
                    hand_id=hand_id,
                    game_id=game_id,
                    decision_seed=decision_seed,
                    street=street,
                )
                if action is not None:
                    return action
            if actions and self.opening_model is not None:
                sample = policy_sample(board, dealt, actions)
                action_index = self._safe_choose_index(
                    self.opening_model, sample, actions
                )
                if action_index is not None:
                    return actions[action_index]
            if actions and self.turn1_model is not None:
                return choose_opening_action_by_turn1_lookahead(
                    board,
                    dealt,
                    actions,
                    downstream_model=self.turn1_model,
                    dead_cards=dead,
                    future_samples=self.opening_lookahead_samples,
                    rng=decision_rng,
                )
            if actions:
                return decision_rng.choice(actions)

        if board.card_count() == 11:
            terminal_opponent = (
                opponent_board
                if opponent_board is not None and opponent_board.is_complete()
                else None
            )
            ranked = evaluate_turn_actions(
                board,
                dealt,
                opponent_board=terminal_opponent,
                fl_ev=self.fl_ev,
            )
            if ranked:
                return ranked[0].action

        if board.card_count() == 9 and self.hu_turn3_model is not None and opponent_board is not None:
            action = self._choose_hu_turn3_action(
                board,
                dealt,
                dead_cards=dead,
                opponent_board=opponent_board,
                hand_id=hand_id,
                game_id=game_id,
                decision_seed=decision_seed,
                street=street,
            )
            if action is not None:
                return action

        if board.card_count() == 9 and self.turn3_model is not None:
            actions = generate_turn_actions(board, dealt)
            if actions:
                sample = policy_sample(board, dealt, actions)
                action_index = self._safe_choose_index(
                    self.turn3_model, sample, actions
                )
                if action_index is not None:
                    return actions[action_index]

        if board.card_count() == 7 and self.turn2_model is not None:
            actions = generate_turn_actions(board, dealt)
            if actions:
                sample = policy_sample(board, dealt, actions)
                action_index = self._safe_choose_index(
                    self.turn2_model, sample, actions
                )
                if action_index is not None:
                    return actions[action_index]

        if board.card_count() == 5 and self.hu_turn1_model is not None and opponent_board is not None:
            action = self._choose_hu_turn1_action(
                board,
                dealt,
                dead_cards=dead,
                opponent_board=opponent_board,
                hand_id=hand_id,
                game_id=game_id,
                decision_seed=decision_seed,
                street=street,
            )
            if action is not None:
                return action

        if board.card_count() == 5 and self.turn1_model is not None:
            actions = generate_turn_actions(board, dealt)
            if actions:
                sample = policy_sample(board, dealt, actions)
                action_index = self._safe_choose_index(
                    self.turn1_model, sample, actions
                )
                if action_index is not None:
                    return actions[action_index]

        actions = generate_actions(board, dealt)
        if not actions:
            raise ValueError("no legal actions")
        return decision_rng.choice(actions)

    def _choose_hu_turn0_action(
        self,
        board: Board,
        dealt: tuple[str, ...],
        *,
        actions: list[Action],
        dead_cards: tuple[str, ...],
        opponent_board: Board,
        hand_id: str | int | None,
        game_id: str | int | None,
        decision_seed: int | None,
        street: str | None,
    ) -> Action | None:
        started_at = time.perf_counter()
        fallback_sample = policy_sample(board, dealt, actions)
        fallback_predictions, fallback_reason = self._safe_predict(
            self.opening_model,
            fallback_sample,
            len(actions),
        )
        if fallback_predictions is None:
            return None
        fallback_index = self._argmax_index(fallback_predictions, actions)
        candidate_indices = sorted(
            range(len(actions)),
            key=lambda index: (
                -fallback_predictions[index],
                action_key(actions[index]).sort_key(),
            ),
        )[: min(self.hu_turn0_candidate_topk, len(actions))]
        if fallback_index not in candidate_indices:
            candidate_indices.append(fallback_index)
        candidate_actions = [actions[index] for index in candidate_indices]
        fallback_candidate_index = candidate_indices.index(fallback_index)

        candidate_index: int | None = None
        final_index = fallback_index
        predicted_margin: float | None = None
        candidate_score: float | None = None
        fallback_score: float | None = None
        safe_selector_score: float | None = None
        legality_check_result = "not_evaluated"
        no_override_reason = ""
        threshold = (
            float(self.hu_turn0_min_margin_by_seat.get(self.seat, self.hu_turn0_min_margin))
            if self.hu_turn0_min_margin_by_seat is not None
            else float(self.hu_turn0_min_margin)
        )
        safe_selector_threshold = (
            float(
                self.hu_turn0_safe_selector_threshold_by_seat.get(
                    self.seat,
                    self.hu_turn0_safe_selector_threshold,
                )
            )
            if self.hu_turn0_safe_selector_threshold_by_seat is not None
            else float(self.hu_turn0_safe_selector_threshold)
        )

        if self.hu_turn0_allowed_seats is not None and self.seat not in self.hu_turn0_allowed_seats:
            no_override_reason = "seat_disabled"
        else:
            try:
                sample = hu_policy_sample(
                    board,
                    dealt,
                    candidate_actions,
                    opponent_board=opponent_board,
                    dead_cards=dead_cards,
                    seat=self.seat,
                    to_act_order=self.seat,
                )
            except Exception:
                sample = None
                no_override_reason = "feature_failed"
            if sample is not None:
                predictions, predict_reason = self._safe_predict(
                    self.hu_turn0_model,
                    sample,
                    len(candidate_actions),
                )
                if predictions is None:
                    no_override_reason = (
                        "prediction_failed"
                        if predict_reason == "fallback_to_stage3"
                        else predict_reason
                    )
                else:
                    candidate_local_index = self._argmax_index(predictions, candidate_actions)
                    candidate_index = candidate_indices[candidate_local_index]
                    candidate_score = predictions[candidate_local_index]
                    fallback_score = predictions[fallback_candidate_index]
                    predicted_margin = candidate_score - fallback_score
                    if candidate_index == fallback_index:
                        no_override_reason = "same_as_baseline"
                    elif predicted_margin < threshold:
                        no_override_reason = "below_hu_turn0_margin"
                    else:
                        if self.hu_turn0_safe_selector_enabled:
                            if self.hu_turn0_safe_selector_model is None:
                                no_override_reason = "safe_selector_unavailable"
                            else:
                                try:
                                    from .hu_turn0_safe_selector import (
                                        score_hu_turn0_safe_selector,
                                    )

                                    safe_selector_score = score_hu_turn0_safe_selector(
                                        self.hu_turn0_safe_selector_model,
                                        {
                                            "seat": self.seat,
                                            "hero_board": board_to_json(board),
                                            "opponent_board": board_to_json(opponent_board),
                                            "cards_to_place": list(dealt),
                                            "dead_cards": list(dead_cards),
                                            "candidate_action": action_to_json(
                                                board,
                                                actions[candidate_index],
                                            ),
                                            "baseline_action": action_to_json(
                                                board,
                                                actions[fallback_index],
                                            ),
                                            "candidate_action_index": candidate_index,
                                            "baseline_action_index": fallback_index,
                                            "candidate_topk": self.hu_turn0_candidate_topk,
                                            "candidate_pool_count": len(candidate_indices),
                                            "action_count": len(actions),
                                            "predicted_margin": predicted_margin,
                                            "candidate_score": candidate_score,
                                            "baseline_score": fallback_score,
                                        },
                                    )
                                except Exception:
                                    no_override_reason = "safe_selector_failed"
                                else:
                                    if safe_selector_score < safe_selector_threshold:
                                        no_override_reason = "below_hu_turn0_safe_selector"
                        if not no_override_reason:
                            legality_check_result = self._candidate_legality_result(
                                board,
                                actions,
                                candidate_index,
                            )
                            if legality_check_result == "legal":
                                final_index = candidate_index
                            else:
                                no_override_reason = "illegal_candidate"

        override_fired = final_index != fallback_index
        self._log_hu_turn0_decision(
            self._hu_turn0_decision_record(
                board=board,
                opponent_board=opponent_board,
                dealt=dealt,
                dead_cards=dead_cards,
                actions=actions,
                candidate_indices=candidate_indices,
                hand_id=hand_id,
                game_id=game_id,
                decision_seed=decision_seed,
                street=street,
                fallback_index=fallback_index,
                candidate_index=candidate_index,
                final_index=final_index,
                override_fired=override_fired,
                no_override_reason=no_override_reason or fallback_reason,
                threshold=threshold,
                predicted_margin=predicted_margin,
                candidate_score=candidate_score,
                fallback_score=fallback_score,
                safe_selector_score=safe_selector_score,
                safe_selector_threshold=safe_selector_threshold,
                legality_check_result=legality_check_result,
                latency_ms=self._elapsed_ms(started_at),
            )
        )
        return actions[final_index]

    def _choose_hu_turn1_action(
        self,
        board: Board,
        dealt: tuple[str, ...],
        *,
        dead_cards: tuple[str, ...],
        opponent_board: Board,
        hand_id: str | int | None,
        game_id: str | int | None,
        decision_seed: int | None,
        street: str | None,
    ) -> Action | None:
        actions = generate_turn_actions(board, dealt)
        if not actions:
            return None

        started_at = time.perf_counter()
        to_act_order = "second" if opponent_board.card_count() > board.card_count() else "first"
        fallback_sample = policy_sample(board, dealt, actions)
        fallback_index = self._safe_choose_index(
            self.turn1_model, fallback_sample, actions
        )
        sample = hu_policy_sample(
            board,
            dealt,
            actions,
            opponent_board=opponent_board,
            dead_cards=dead_cards,
            seat=self.seat,
            to_act_order=to_act_order,
        )

        candidate_index: int | None = None
        final_index: int | None = None
        no_override_reason = ""
        predicted_margin: float | None = None
        candidate_score: float | None = None
        fallback_score: float | None = None
        legality_check_result = "legal"

        if self.hu_turn1_min_margin is None:
            candidate_index = self._safe_choose_index(
                self.hu_turn1_model, sample, actions
            )
            if candidate_index is None:
                final_index = fallback_index
                no_override_reason = "fallback_to_self_board"
            else:
                final_index = candidate_index
                no_override_reason = (
                    "same_as_baseline" if candidate_index == fallback_index else "full_replacement"
                )
        else:
            predictions, predict_reason = self._safe_predict(
                self.hu_turn1_model,
                sample,
                len(actions),
            )
            if predictions is None:
                final_index = fallback_index
                no_override_reason = predict_reason
                legality_check_result = predict_reason
            else:
                candidate_index = self._argmax_index(predictions, actions)
                candidate_score = predictions[candidate_index]
                fallback_score = (
                    predictions[fallback_index] if fallback_index is not None else None
                )
                if fallback_index is None:
                    predicted_margin = self._prediction_margin(predictions, candidate_index)
                else:
                    predicted_margin = candidate_score - fallback_score

                if fallback_index is not None and candidate_index == fallback_index:
                    final_index = fallback_index
                    no_override_reason = "same_as_baseline"
                elif predicted_margin >= self.hu_turn1_min_margin:
                    legality_check_result = self._candidate_legality_result(
                        board,
                        actions,
                        candidate_index,
                    )
                    if legality_check_result == "legal":
                        final_index = candidate_index
                    else:
                        final_index = fallback_index
                        no_override_reason = "illegal_candidate"
                else:
                    final_index = fallback_index
                    no_override_reason = "below_hu_turn1_margin"

        if final_index is None:
            return None

        override_fired = fallback_index is not None and final_index != fallback_index
        self._log_hu_turn1_decision(
            self._hu_turn1_decision_record(
                board=board,
                opponent_board=opponent_board,
                dealt=dealt,
                dead_cards=dead_cards,
                actions=actions,
                hand_id=hand_id,
                game_id=game_id,
                decision_seed=decision_seed,
                street=street,
                fallback_index=fallback_index,
                candidate_index=candidate_index,
                final_index=final_index,
                override_fired=override_fired,
                no_override_reason=no_override_reason,
                predicted_margin=predicted_margin,
                candidate_score=candidate_score,
                fallback_score=fallback_score,
                legality_check_result=legality_check_result,
                latency_ms=self._elapsed_ms(started_at),
            )
        )
        return actions[final_index]

    def _choose_hu_turn3_action(
        self,
        board: Board,
        dealt: tuple[str, ...],
        *,
        dead_cards: tuple[str, ...],
        opponent_board: Board,
        hand_id: str | int | None,
        game_id: str | int | None,
        decision_seed: int | None,
        street: str | None,
    ) -> Action | None:
        actions = generate_turn_actions(board, dealt)
        if not actions:
            return None

        started_at = time.perf_counter()
        to_act_order = "second" if opponent_board.card_count() > board.card_count() else "first"

        fallback_sample = policy_sample(board, dealt, actions)
        fallback_index = self._safe_choose_index(
            self.turn3_model, fallback_sample, actions
        )
        if fallback_index is None:
            if self.hu_turn3_stage7_enabled and self._can_use_legacy_hu_turn3_direct():
                try:
                    sample = hu_policy_sample(
                        board,
                        dealt,
                        actions,
                        opponent_board=opponent_board,
                        dead_cards=dead_cards,
                        seat=self.seat,
                        to_act_order=to_act_order,
                    )
                    action_index = self._safe_choose_index(
                        self.hu_turn3_model, sample, actions
                    )
                    return actions[action_index] if action_index is not None else None
                except Exception:
                    return None
            return None

        if not self.hu_turn3_stage7_enabled:
            stage3_index = fallback_index
            reference_margin = None
            no_override_reason = "stage7_disabled"
            try:
                sample = hu_policy_sample(
                    board,
                    dealt,
                    actions,
                    opponent_board=opponent_board,
                    dead_cards=dead_cards,
                    seat=self.seat,
                    to_act_order=to_act_order,
                )
            except Exception:
                sample = None
                no_override_reason = "feature_failed"
            if sample is not None and self.hu_turn3_reference_model is not None:
                reference_predictions, reference_reason = self._safe_predict(
                    self.hu_turn3_reference_model,
                    sample,
                    len(actions),
                )
                if reference_predictions is None:
                    no_override_reason = reference_reason
                else:
                    reference_index = self._argmax_index(reference_predictions, actions)
                    reference_margin = self._prediction_margin(reference_predictions, reference_index)
                    stage3_index = reference_index
            self._log_hu_turn3_decision(
                self._stage7_decision_record(
                    board=board,
                    opponent_board=opponent_board,
                    dealt=dealt,
                    dead_cards=dead_cards,
                    actions=actions,
                    hand_id=hand_id,
                    game_id=game_id,
                    decision_seed=decision_seed,
                    street=street,
                    fallback_index=fallback_index,
                    stage3_index=stage3_index,
                    stage7_index=None,
                    final_index=stage3_index,
                    override_fired=False,
                    no_override_reason=no_override_reason,
                    reference_margin=reference_margin,
                    stage7_predicted_margin=None,
                    model_score=None,
                    legality_check_result="not_evaluated",
                    latency_ms=self._elapsed_ms(started_at),
                )
            )
            return actions[stage3_index]

        try:
            sample = hu_policy_sample(
                board,
                dealt,
                actions,
                opponent_board=opponent_board,
                dead_cards=dead_cards,
                seat=self.seat,
                to_act_order=to_act_order,
            )
        except Exception:
            self._log_hu_turn3_decision(
                self._stage7_decision_record(
                    board=board,
                    opponent_board=opponent_board,
                    dealt=dealt,
                    dead_cards=dead_cards,
                    actions=actions,
                    hand_id=hand_id,
                    game_id=game_id,
                    decision_seed=decision_seed,
                    street=street,
                    fallback_index=fallback_index,
                    stage3_index=fallback_index,
                    stage7_index=None,
                    final_index=fallback_index,
                    override_fired=False,
                    no_override_reason="feature_failed",
                    reference_margin=None,
                    stage7_predicted_margin=None,
                    model_score=None,
                    legality_check_result="not_evaluated",
                    latency_ms=self._elapsed_ms(started_at),
                )
            )
            return actions[fallback_index]

        reference_margin: float | None = None
        stage3_index = fallback_index
        reference_gate_available = self.hu_turn3_reference_model is not None
        reference_reason = ""
        if reference_gate_available:
            reference_predictions, reference_reason = self._safe_predict(
                self.hu_turn3_reference_model,
                sample,
                len(actions),
            )
            if reference_predictions is None:
                reference_gate_available = False
            else:
                reference_index = self._argmax_index(reference_predictions, actions)
                reference_margin = self._prediction_margin(reference_predictions, reference_index)
                stage3_index = reference_index

        if not reference_gate_available and self.hu_turn3_reference_min_margin > 0.0:
            self._log_hu_turn3_decision(
                self._stage7_decision_record(
                    board=board,
                    opponent_board=opponent_board,
                    dealt=dealt,
                    dead_cards=dead_cards,
                    actions=actions,
                    hand_id=hand_id,
                    game_id=game_id,
                    decision_seed=decision_seed,
                    street=street,
                    fallback_index=fallback_index,
                    stage3_index=stage3_index,
                    stage7_index=None,
                    final_index=stage3_index,
                    override_fired=False,
                    no_override_reason=reference_reason or "model_load_failed",
                    reference_margin=reference_margin,
                    stage7_predicted_margin=None,
                    model_score=None,
                    legality_check_result="not_evaluated",
                    latency_ms=self._elapsed_ms(started_at),
                )
            )
            return actions[stage3_index]

        if (reference_margin or 0.0) < self.hu_turn3_reference_min_margin:
            self._log_hu_turn3_decision(
                self._stage7_decision_record(
                    board=board,
                    opponent_board=opponent_board,
                    dealt=dealt,
                    dead_cards=dead_cards,
                    actions=actions,
                    hand_id=hand_id,
                    game_id=game_id,
                    decision_seed=decision_seed,
                    street=street,
                    fallback_index=fallback_index,
                    stage3_index=stage3_index,
                    stage7_index=None,
                    final_index=stage3_index,
                    override_fired=False,
                    no_override_reason="below_reference_margin",
                    reference_margin=reference_margin,
                    stage7_predicted_margin=None,
                    model_score=None,
                    legality_check_result="not_evaluated",
                    latency_ms=self._elapsed_ms(started_at),
                )
            )
            return actions[stage3_index]

        if self.hu_turn3_allowed_seats is not None and self.seat not in self.hu_turn3_allowed_seats:
            self._log_hu_turn3_decision(
                self._stage7_decision_record(
                    board=board,
                    opponent_board=opponent_board,
                    dealt=dealt,
                    dead_cards=dead_cards,
                    actions=actions,
                    hand_id=hand_id,
                    game_id=game_id,
                    decision_seed=decision_seed,
                    street=street,
                    fallback_index=fallback_index,
                    stage3_index=stage3_index,
                    stage7_index=None,
                    final_index=stage3_index,
                    override_fired=False,
                    no_override_reason="seat_not_allowed",
                    reference_margin=reference_margin,
                    stage7_predicted_margin=None,
                    model_score=None,
                    legality_check_result="not_evaluated",
                    latency_ms=self._elapsed_ms(started_at),
                )
            )
            return actions[stage3_index]

        predictions, prediction_reason = self._safe_predict(self.hu_turn3_model, sample, len(actions))
        if predictions is None:
            self._log_hu_turn3_decision(
                self._stage7_decision_record(
                    board=board,
                    opponent_board=opponent_board,
                    dealt=dealt,
                    dead_cards=dead_cards,
                    actions=actions,
                    hand_id=hand_id,
                    game_id=game_id,
                    decision_seed=decision_seed,
                    street=street,
                    fallback_index=fallback_index,
                    stage3_index=stage3_index,
                    stage7_index=None,
                    final_index=stage3_index,
                    override_fired=False,
                    no_override_reason=prediction_reason,
                    reference_margin=reference_margin,
                    stage7_predicted_margin=None,
                    model_score=None,
                    legality_check_result="not_evaluated",
                    latency_ms=self._elapsed_ms(started_at),
                )
            )
            return actions[stage3_index]

        action_index = self._argmax_index(predictions, actions)
        legality_check_result = self._candidate_legality_result(board, actions, action_index)
        stage7_predicted_margin = float(predictions[action_index]) - float(predictions[stage3_index])
        model_score = float(predictions[action_index])

        no_override_reason = ""
        if legality_check_result != "legal":
            no_override_reason = "illegal_candidate"
        elif action_index == stage3_index:
            no_override_reason = "same_as_stage3"
        elif stage7_predicted_margin < self.hu_turn3_min_margin:
            no_override_reason = "below_stage7_margin"
        elif (
            self.hu_turn3_min_model_score is not None
            and model_score < self.hu_turn3_min_model_score
        ):
            no_override_reason = "below_model_score"

        self_predictions = None
        if not no_override_reason and self.hu_turn3_support_model is not None and self.hu_turn3_min_support_margin > 0.0:
            support_predictions, support_reason = self._safe_predict(
                self.hu_turn3_support_model,
                sample,
                len(actions),
            )
            if support_predictions is None:
                no_override_reason = support_reason
            else:
                support_margin = float(support_predictions[action_index]) - float(
                    support_predictions[stage3_index]
                )
                if support_margin < self.hu_turn3_min_support_margin:
                    no_override_reason = "below_support_margin"
        if not no_override_reason and self.hu_turn3_max_self_regret is not None:
            self_predictions, self_reason = self._safe_predict(
                self.turn3_model,
                fallback_sample,
                len(actions),
            )
            if self_predictions is None:
                no_override_reason = self_reason
            else:
                self_regret = float(self_predictions[stage3_index]) - float(
                    self_predictions[action_index]
                )
                if self_regret > self.hu_turn3_max_self_regret:
                    no_override_reason = "fallback_to_stage3"
        if (
            not no_override_reason
            and self.hu_turn3_gate_model is not None
            and self.hu_turn3_min_gate_probability > 0.0
        ):
            if self_predictions is None:
                self_predictions, self_reason = self._safe_predict(
                    self.turn3_model,
                    fallback_sample,
                    len(actions),
                )
                if self_predictions is None:
                    no_override_reason = self_reason
            if not no_override_reason:
                gate_probability = self.hu_turn3_gate_model.predict_accept_probability(
                    sample,
                    chosen_index=action_index,
                    baseline_index=stage3_index,
                    hu_predictions=predictions,
                    self_predictions=self_predictions,
                )
                if not math.isfinite(float(gate_probability)):
                    no_override_reason = "nan_prediction"
                elif gate_probability < self.hu_turn3_min_gate_probability:
                    no_override_reason = "fallback_to_stage3"

        override_fired = not no_override_reason
        final_index = action_index if override_fired else stage3_index
        self._log_hu_turn3_decision(
            self._stage7_decision_record(
                board=board,
                opponent_board=opponent_board,
                dealt=dealt,
                dead_cards=dead_cards,
                actions=actions,
                hand_id=hand_id,
                game_id=game_id,
                decision_seed=decision_seed,
                street=street,
                fallback_index=fallback_index,
                stage3_index=stage3_index,
                stage7_index=action_index,
                final_index=final_index,
                override_fired=override_fired,
                no_override_reason=no_override_reason,
                reference_margin=reference_margin,
                stage7_predicted_margin=stage7_predicted_margin,
                model_score=model_score,
                legality_check_result=legality_check_result,
                latency_ms=self._elapsed_ms(started_at),
            )
        )
        return actions[final_index]

    def _safe_choose_index(
        self,
        model: object | None,
        sample: dict,
        actions: list[Action],
    ) -> int | None:
        if model is None:
            return None
        predictions, _reason = self._safe_predict(model, sample, len(actions))
        if predictions is not None:
            return self._argmax_index(predictions, actions)
        try:
            index = int(model.choose_action_index(sample))
        except Exception:
            return None
        if index < 0 or index >= len(actions):
            return None
        return index

    def _can_use_legacy_hu_turn3_direct(self) -> bool:
        return (
            self.hu_turn3_min_margin <= 0.0
            and self.hu_turn3_reference_model is None
            and self.hu_turn3_reference_min_margin <= 0.0
            and self.hu_turn3_support_model is None
            and self.hu_turn3_min_support_margin <= 0.0
            and self.hu_turn3_min_model_score is None
            and self.hu_turn3_allowed_seats is None
            and self.hu_turn3_gate_model is None
            and self.hu_turn3_min_gate_probability <= 0.0
            and self.hu_turn3_max_self_regret is None
        )

    def _safe_predict(
        self,
        model: object | None,
        sample: dict,
        expected_len: int,
    ) -> tuple[list[float] | None, str]:
        if model is None:
            return None, "model_load_failed"
        try:
            raw_predictions = model.predict_sample(sample)
            predictions = [float(value) for value in raw_predictions]
        except Exception:
            return None, "fallback_to_stage3"
        if len(predictions) != expected_len:
            return None, "illegal_candidate"
        if any(not math.isfinite(value) for value in predictions):
            return None, "nan_prediction"
        return predictions, ""

    def _argmax_index(
        self,
        predictions: list[float],
        actions: list[Action] | None = None,
    ) -> int:
        best = max(predictions)
        tied = [index for index, value in enumerate(predictions) if value == best]
        if actions is None:
            return tied[0]
        return min(tied, key=lambda index: action_key(actions[index]).sort_key())

    def _prediction_margin(self, predictions: list[float], best_index: int) -> float:
        if len(predictions) <= 1:
            return 0.0
        best = float(predictions[best_index])
        second = max(float(value) for index, value in enumerate(predictions) if index != best_index)
        return best - second

    def _candidate_legality_result(self, board: Board, actions: list[Action], action_index: int) -> str:
        if action_index < 0 or action_index >= len(actions):
            return "illegal_candidate"
        try:
            board.place(actions[action_index].placements)
        except Exception:
            return "illegal_candidate"
        return "legal"

    def _hu_turn0_decision_record(
        self,
        *,
        board: Board,
        opponent_board: Board,
        dealt: tuple[str, ...],
        dead_cards: tuple[str, ...],
        actions: list[Action],
        candidate_indices: list[int],
        hand_id: str | int | None,
        game_id: str | int | None,
        decision_seed: int | None,
        street: str | None,
        fallback_index: int,
        candidate_index: int | None,
        final_index: int,
        override_fired: bool,
        no_override_reason: str,
        threshold: float,
        predicted_margin: float | None,
        candidate_score: float | None,
        fallback_score: float | None,
        safe_selector_score: float | None,
        safe_selector_threshold: float,
        legality_check_result: str,
        latency_ms: float,
    ) -> dict[str, Any]:
        context = self.decision_context or {}
        visible_dead_cards = list(dead_cards)
        opponent_public = set(opponent_board.all_cards())
        hero_private_discards = [
            card for card in dead_cards if card not in opponent_public
        ]
        return {
            "hand_id": hand_id,
            "game_id": game_id,
            "seed": decision_seed if decision_seed is not None else self.seed,
            "street": street or "T0",
            "turn": street or "T0",
            "seat": self.seat,
            "hero_board": board_to_json(board),
            "opponent_board": board_to_json(opponent_board),
            "cards_to_place": list(dealt),
            "dead_cards": list(dead_cards),
            "visibility_model": "actor_observation_v1",
            "discard_visibility": "own_private_only",
            "true_dead_cards": [],
            "visible_dead_cards": visible_dead_cards,
            "hero_private_discards": hero_private_discards,
            "replay_ready": False,
            "action_count": len(actions),
            "candidate_pool_count": len(candidate_indices),
            "candidate_original_indices": list(candidate_indices),
            "candidate_action_keys": [
                action_key(actions[index]).to_token() for index in candidate_indices
            ],
            "action_key_schema": ACTION_KEY_SCHEMA,
            "legal_action_set_digest": legal_action_set_digest(actions),
            "legal_action_order_digest": ordered_action_mapping_digest(actions),
            "fallback_action_index": fallback_index,
            "candidate_action_index": candidate_index,
            "final_action_index": final_index,
            "fallback_action_key": action_key(actions[fallback_index]).to_token(),
            "candidate_action_key": (
                action_key(actions[candidate_index]).to_token()
                if candidate_index is not None
                else None
            ),
            "final_action_key": action_key(actions[final_index]).to_token(),
            "baseline_action": action_to_json(board, actions[fallback_index]),
            "hu_turn0_action": (
                action_to_json(board, actions[candidate_index])
                if candidate_index is not None
                else None
            ),
            "fallback_action": action_to_json(board, actions[fallback_index]),
            "final_action": action_to_json(board, actions[final_index]),
            "override_fired": override_fired,
            "no_override_reason": no_override_reason,
            "hu_turn0_candidate_topk": self.hu_turn0_candidate_topk,
            "hu_turn0_min_margin": threshold,
            "hu_turn0_predicted_margin": predicted_margin,
            "candidate_score": candidate_score,
            "fallback_score": fallback_score,
            "hu_turn0_safe_selector_enabled": self.hu_turn0_safe_selector_enabled,
            "hu_turn0_safe_selector_score": safe_selector_score,
            "hu_turn0_safe_selector_threshold": safe_selector_threshold,
            "legality_check_result": legality_check_result,
            "runtime_latency_ms": latency_ms,
            "runtime_profile": context.get("runtime_profile"),
            "runtime_status": context.get("runtime_status"),
            "selective_override_only": context.get("selective_override_only"),
            "full_replacement_enabled": context.get("full_replacement_enabled"),
            "fallback_policy": context.get("fallback_policy"),
            "t1_continuation": context.get("t1_continuation"),
            "t2_continuation": context.get("t2_continuation"),
            "t3_continuation": context.get("t3_continuation"),
        }

    def _log_hu_turn0_decision(self, record: dict[str, Any]) -> None:
        if self.hu_turn0_decision_log is not None:
            self.hu_turn0_decision_log.append(record)
        if self.hu_turn0_decision_log_path is None:
            return
        path = Path(self.hu_turn0_decision_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    def _hu_turn1_decision_record(
        self,
        *,
        board: Board,
        opponent_board: Board,
        dealt: tuple[str, ...],
        dead_cards: tuple[str, ...],
        actions: list[Action],
        hand_id: str | int | None,
        game_id: str | int | None,
        decision_seed: int | None,
        street: str | None,
        fallback_index: int | None,
        candidate_index: int | None,
        final_index: int,
        override_fired: bool,
        no_override_reason: str,
        predicted_margin: float | None,
        candidate_score: float | None,
        fallback_score: float | None,
        legality_check_result: str,
        latency_ms: float,
    ) -> dict[str, Any]:
        context = self.decision_context or {}
        visible_dead_cards = list(dead_cards)
        opponent_public = set(opponent_board.all_cards())
        hero_private_discards = [
            card for card in dead_cards if card not in opponent_public
        ]
        return {
            "hand_id": hand_id,
            "game_id": game_id,
            "seed": decision_seed if decision_seed is not None else self.seed,
            "street": street or "T1",
            "turn": street or "T1",
            "seat": self.seat,
            "hero_board": board_to_json(board),
            "opponent_board": board_to_json(opponent_board),
            "cards_to_place": list(dealt),
            "dead_cards": list(dead_cards),
            "visibility_model": "actor_observation_v1",
            "discard_visibility": "own_private_only",
            "true_dead_cards": [],
            "visible_dead_cards": visible_dead_cards,
            "hero_private_discards": hero_private_discards,
            "replay_ready": False,
            "action_count": len(actions),
            "action_key_schema": ACTION_KEY_SCHEMA,
            "legal_action_set_digest": legal_action_set_digest(actions),
            "legal_action_order_digest": ordered_action_mapping_digest(actions),
            "fallback_action_index": fallback_index,
            "candidate_action_index": candidate_index,
            "final_action_index": final_index,
            "fallback_action_key": (
                action_key(actions[fallback_index]).to_token()
                if fallback_index is not None
                else None
            ),
            "candidate_action_key": (
                action_key(actions[candidate_index]).to_token()
                if candidate_index is not None
                else None
            ),
            "final_action_key": action_key(actions[final_index]).to_token(),
            "baseline_action": (
                action_to_json(board, actions[fallback_index])
                if fallback_index is not None
                else None
            ),
            "hu_turn1_action": (
                action_to_json(board, actions[candidate_index])
                if candidate_index is not None
                else None
            ),
            "fallback_action": (
                action_to_json(board, actions[fallback_index])
                if fallback_index is not None
                else None
            ),
            "final_action": action_to_json(board, actions[final_index]),
            "override_fired": override_fired,
            "no_override_reason": no_override_reason or "",
            "hu_turn1_min_margin": self.hu_turn1_min_margin,
            "hu_turn1_predicted_margin": predicted_margin,
            "candidate_score": candidate_score,
            "fallback_score": fallback_score,
            "model_score": candidate_score,
            "legality_check_result": legality_check_result,
            "runtime_latency_ms": latency_ms,
        }

    def _log_hu_turn1_decision(self, record: dict[str, Any]) -> None:
        if self.hu_turn1_decision_log is not None:
            self.hu_turn1_decision_log.append(record)
        if self.hu_turn1_decision_log_path is None:
            return
        path = Path(self.hu_turn1_decision_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    def _stage7_decision_record(
        self,
        *,
        board: Board,
        opponent_board: Board,
        dealt: tuple[str, ...],
        dead_cards: tuple[str, ...],
        actions: list[Action],
        hand_id: str | int | None,
        game_id: str | int | None,
        decision_seed: int | None,
        street: str | None,
        fallback_index: int,
        stage3_index: int,
        stage7_index: int | None,
        final_index: int,
        override_fired: bool,
        no_override_reason: str,
        reference_margin: float | None,
        stage7_predicted_margin: float | None,
        model_score: float | None,
        legality_check_result: str,
        latency_ms: float,
    ) -> dict[str, Any]:
        opponent_public = set(opponent_board.all_cards())
        hero_private_discards = [
            card for card in dead_cards if card not in opponent_public
        ]
        return {
            "hand_id": hand_id,
            "game_id": game_id,
            "seed": decision_seed if decision_seed is not None else self.seed,
            "street": street or "T3",
            "turn": street or "T3",
            "seat": self.seat,
            "hero_board": board_to_json(board),
            "opponent_board": board_to_json(opponent_board),
            "cards_to_place": list(dealt),
            "dead_cards": list(dead_cards),
            "visibility_model": "actor_observation_v1",
            "discard_visibility": "own_private_only",
            "true_dead_cards": [],
            "visible_dead_cards": list(dead_cards),
            "hero_private_discards": hero_private_discards,
            "replay_ready": False,
            "action_count": len(actions),
            "action_key_schema": ACTION_KEY_SCHEMA,
            "legal_action_set_digest": legal_action_set_digest(actions),
            "legal_action_order_digest": ordered_action_mapping_digest(actions),
            "fallback_action_index": fallback_index,
            "stage3_action_index": stage3_index,
            "stage7_action_index": stage7_index,
            "final_action_index": final_index,
            "fallback_action_key": action_key(actions[fallback_index]).to_token(),
            "stage3_action_key": action_key(actions[stage3_index]).to_token(),
            "stage7_action_key": (
                action_key(actions[stage7_index]).to_token()
                if stage7_index is not None
                else None
            ),
            "final_action_key": action_key(actions[final_index]).to_token(),
            "stage3_action": action_to_json(board, actions[stage3_index]),
            "stage7_action": (
                action_to_json(board, actions[stage7_index])
                if stage7_index is not None
                else None
            ),
            "fallback_action": action_to_json(board, actions[fallback_index]),
            "final_action": action_to_json(board, actions[final_index]),
            "override_fired": override_fired,
            "no_override_reason": no_override_reason or "",
            "hu_turn3_min_margin": self.hu_turn3_min_margin,
            "hu_turn3_reference_min_margin": self.hu_turn3_reference_min_margin,
            "hu_turn3_min_model_score": self.hu_turn3_min_model_score,
            "hu_turn3_allowed_seats": (
                list(self.hu_turn3_allowed_seats)
                if self.hu_turn3_allowed_seats is not None
                else None
            ),
            "stage7_predicted_margin": stage7_predicted_margin,
            "reference_margin": reference_margin,
            "model_score": model_score,
            "ev": model_score,
            "rank_score": model_score,
            "legality_check_result": legality_check_result,
            "foul_risk": None,
            "fl": None,
            "royalty": None,
            "runtime_latency_ms": latency_ms,
        }

    def _log_hu_turn3_decision(self, record: dict[str, Any]) -> None:
        if self.hu_turn3_decision_log is not None:
            self.hu_turn3_decision_log.append(record)
        if self.hu_turn3_decision_log_path is None:
            return
        path = Path(self.hu_turn3_decision_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    def _elapsed_ms(self, started_at: float) -> float:
        return (time.perf_counter() - started_at) * 1000.0


def policy_sample(board: Board, dealt_cards: Iterable[str], actions: list[Action]) -> dict:
    """Build an action-value model sample from legal actions."""
    return {
        "rule_set": "regular",
        "phase": phase_for_card_count(board.card_count()),
        "board": board_to_json(board),
        "dealt": list(dealt_cards),
        "best_action": 0,
        "score_gap": 0.0,
        "actions": [action_to_json(board, action) for action in actions],
    }


def choose_opening_action_by_turn1_lookahead(
    board: Board,
    dealt_cards: Iterable[str],
    actions: list[Action],
    *,
    downstream_model: Turn3RidgeModel,
    future_samples: int,
    rng: random.Random,
    dead_cards: Iterable[str] = (),
) -> Action:
    """Choose the best exact opening placement by bootstrapping into Turn1.

    The opening placements themselves are fully enumerated by ``actions``.
    ``future_samples == 0`` enumerates all possible next 3-card deals from the
    locally visible deck; otherwise it samples that many next deals.
    """
    if future_samples < 0:
        raise ValueError("future_samples must be non-negative")
    if not actions:
        raise ValueError("no legal opening actions")
    dealt = tuple(dealt_cards)
    dead = tuple(dead_cards)
    used = set(board.all_cards()) | set(dealt) | set(dead)
    validate_cards((*board.all_cards(), *dealt, *dead))
    remaining = tuple(card for card in ALL_CARDS if card not in used)
    futures = select_future_deals(remaining, future_samples=future_samples, rng=rng)

    best_action = actions[0]
    best_score = float("-inf")
    for action in actions:
        next_board = board.place(action.placements)
        total = 0.0
        count = 0
        for future in futures:
            next_actions = generate_turn_actions(next_board, future)
            if not next_actions:
                continue
            sample = policy_sample(next_board, future, next_actions)
            predictions = downstream_model.predict_sample(sample)
            total += float(predictions.max())
            count += 1
        if count == 0:
            continue
        score = total / count
        if score > best_score:
            best_score = score
            best_action = action
    return best_action


def select_future_deals(
    remaining_cards: Iterable[str],
    *,
    future_samples: int,
    rng: random.Random,
) -> tuple[tuple[str, str, str], ...]:
    remaining = tuple(remaining_cards)
    if future_samples < 0:
        raise ValueError("future_samples must be non-negative")
    if future_samples == 0:
        return tuple(combinations(remaining, 3))
    combo_count = len(remaining) * (len(remaining) - 1) * (len(remaining) - 2) // 6
    if future_samples >= combo_count:
        return tuple(combinations(remaining, 3))
    return tuple(tuple(rng.sample(remaining, 3)) for _ in range(future_samples))


def phase_for_card_count(card_count: int) -> str:
    return {
        0: "opening_0card",
        5: "turn1_5card",
        7: "turn2_7card",
        9: "turn3_9card",
        11: "final_turn",
    }.get(card_count, f"turn_{card_count}card")


def board_to_json(board: Board) -> dict[str, list[str]]:
    return {
        "top": list(board.top),
        "middle": list(board.middle),
        "bottom": list(board.bottom),
    }


def action_to_json(board: Board, action: Action) -> dict:
    next_board = board.place(action.placements)
    return {
        "placements": [list(placement) for placement in action.placements],
        "discards": list(action.discards),
        "score": 0.0,
        "future_count": 0,
        "non_bust_future_count": 0,
        "next_board": board_to_json(next_board),
    }
