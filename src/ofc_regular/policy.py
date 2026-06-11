"""Playable policies for regular OFC Pineapple."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from .action_space import Action, generate_actions, generate_turn_actions
from .cards import ALL_CARDS, validate_cards
from .hu_turn3_gate_model import HuTurn3GateModel
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
    turn1_model: Turn3RidgeModel | None = None
    turn2_model: Turn3RidgeModel | None = None
    turn3_model: Turn3RidgeModel | None = None
    hu_turn3_model: object | None = None
    hu_turn3_reference_model: object | None = None
    hu_turn3_support_model: object | None = None
    hu_turn3_gate_model: HuTurn3GateModel | None = None
    hu_turn3_min_margin: float = 0.0
    hu_turn3_reference_min_margin: float = 0.0
    hu_turn3_min_support_margin: float = 0.0
    hu_turn3_min_gate_probability: float = 0.0
    hu_turn3_max_self_regret: float | None = None
    hu_turn3_stage7_enabled: bool = True
    hu_turn3_decision_log_path: str | Path | None = None
    hu_turn3_decision_log: list[dict[str, Any]] | None = None
    seat: str = "first"
    seed: int = 42
    fl_ev: dict[int, float] | None = None
    opening_lookahead_samples: int = 64

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        if self.fl_ev is None:
            self.fl_ev = dict(DEFAULT_FL_EV)
        if self.opening_lookahead_samples < 0:
            raise ValueError("opening_lookahead_samples must be non-negative")

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
        if board.card_count() == 0:
            actions = generate_actions(board, dealt)
            if actions and self.opening_model is not None:
                sample = policy_sample(board, dealt, actions)
                action_index = self.opening_model.choose_action_index(sample)
                return actions[action_index]
            if actions and self.turn1_model is not None:
                return choose_opening_action_by_turn1_lookahead(
                    board,
                    dealt,
                    actions,
                    downstream_model=self.turn1_model,
                    dead_cards=dead,
                    future_samples=self.opening_lookahead_samples,
                    rng=self.rng,
                )
            if actions:
                return self.rng.choice(actions)

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
                action_index = self.turn3_model.choose_action_index(sample)
                return actions[action_index]

        if board.card_count() == 7 and self.turn2_model is not None:
            actions = generate_turn_actions(board, dealt)
            if actions:
                sample = policy_sample(board, dealt, actions)
                action_index = self.turn2_model.choose_action_index(sample)
                return actions[action_index]

        if board.card_count() == 5 and self.turn1_model is not None:
            actions = generate_turn_actions(board, dealt)
            if actions:
                sample = policy_sample(board, dealt, actions)
                action_index = self.turn1_model.choose_action_index(sample)
                return actions[action_index]

        actions = generate_actions(board, dealt)
        if not actions:
            raise ValueError("no legal actions")
        return self.rng.choice(actions)

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
        fallback_index = self._safe_choose_index(self.turn3_model, fallback_sample, len(actions))
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
                    action_index = self._safe_choose_index(self.hu_turn3_model, sample, len(actions))
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
                    reference_index = self._argmax_index(reference_predictions)
                    reference_margin = self._prediction_margin(reference_predictions, reference_index)
                    stage3_index = reference_index
            self._log_hu_turn3_decision(
                self._stage7_decision_record(
                    board=board,
                    opponent_board=opponent_board,
                    dealt=dealt,
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
                reference_index = self._argmax_index(reference_predictions)
                reference_margin = self._prediction_margin(reference_predictions, reference_index)
                stage3_index = reference_index

        if not reference_gate_available and self.hu_turn3_reference_min_margin > 0.0:
            self._log_hu_turn3_decision(
                self._stage7_decision_record(
                    board=board,
                    opponent_board=opponent_board,
                    dealt=dealt,
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

        predictions, prediction_reason = self._safe_predict(self.hu_turn3_model, sample, len(actions))
        if predictions is None:
            self._log_hu_turn3_decision(
                self._stage7_decision_record(
                    board=board,
                    opponent_board=opponent_board,
                    dealt=dealt,
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

        action_index = self._argmax_index(predictions)
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
        expected_len: int,
    ) -> int | None:
        if model is None:
            return None
        try:
            index = int(model.choose_action_index(sample))
        except Exception:
            return None
        if index < 0 or index >= expected_len:
            return None
        return index

    def _can_use_legacy_hu_turn3_direct(self) -> bool:
        return (
            self.hu_turn3_min_margin <= 0.0
            and self.hu_turn3_reference_model is None
            and self.hu_turn3_reference_min_margin <= 0.0
            and self.hu_turn3_support_model is None
            and self.hu_turn3_min_support_margin <= 0.0
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

    def _argmax_index(self, predictions: list[float]) -> int:
        return max(range(len(predictions)), key=lambda index: predictions[index])

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

    def _stage7_decision_record(
        self,
        *,
        board: Board,
        opponent_board: Board,
        dealt: tuple[str, ...],
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
