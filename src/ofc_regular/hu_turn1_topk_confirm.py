"""Validation-only HU T1 TopK candidate generator with independent confirm MC."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .action_key import (
    ACTION_KEY_SCHEMA,
    action_key,
    action_key_from_payload,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from .action_space import Action, generate_turn_actions
from .evaluate_hu_turn2_stage8b_topk_mc_rerank import HuTurn2Stage8bTopKMcRerankPolicy
from .hu_belief import sample_hidden_card_particles
from .hu_infoset import ActorObservation
from .hu_turn1_safe_selector import score_hu_turn1_safe_selector
from .hu_turn1_teacher_pilot import evaluate_turn1_action_subset
from .hu_turn3_model import hu_policy_sample
from .policy import RegularAiPolicy, action_to_json, policy_sample
from .state import Board


RolloutPolicyFactory = Callable[..., RegularAiPolicy]


@dataclass(frozen=True)
class HuTurn1TopKConfirmConfig:
    top_k: int
    mc_samples: int
    min_delta: float
    confirm_mc_samples: int = 0
    confirm_se_multiplier: float = 0.0
    max_confirm_se: float | None = None
    min_predicted_delta: float | None = None
    safe_selector_threshold: float | None = None
    safe_selector_threshold_first: float | None = None
    safe_selector_threshold_second: float | None = None
    allowed_seats: tuple[str, ...] = ()
    candidate_topk: int = 0
    candidate_union_cap: int = 0
    candidate_union_mode: str = "single_model"

    def safe_selector_threshold_for_seat(self, seat: str) -> float | None:
        if seat == "first" and self.safe_selector_threshold_first is not None:
            return self.safe_selector_threshold_first
        if seat == "second" and self.safe_selector_threshold_second is not None:
            return self.safe_selector_threshold_second
        return self.safe_selector_threshold

    @property
    def config_id(self) -> str:
        parts = [
            f"k{self.top_k:g}",
            f"mc{self.mc_samples:g}",
            f"d{self.min_delta:g}",
        ]
        if self.confirm_mc_samples > 0:
            parts.append(f"confirm{self.confirm_mc_samples:g}")
            parts.append(f"cse{self.confirm_se_multiplier:g}")
        if self.max_confirm_se is not None:
            parts.append(f"csemax{self.max_confirm_se:g}")
        if self.min_predicted_delta is not None:
            parts.append(f"pd{self.min_predicted_delta:g}")
        if self.safe_selector_threshold is not None:
            parts.append(f"safe{self.safe_selector_threshold:g}")
        if self.safe_selector_threshold_first is not None:
            parts.append(f"safefirst{self.safe_selector_threshold_first:g}")
        if self.safe_selector_threshold_second is not None:
            parts.append(f"safesecond{self.safe_selector_threshold_second:g}")
        if self.allowed_seats:
            parts.append(f"seat_{'+'.join(self.allowed_seats)}")
        if self.candidate_topk > 0:
            parts.append(f"ctopk{self.candidate_topk:g}")
        if self.candidate_union_cap > 0:
            parts.append(f"cap{self.candidate_union_cap:g}")
        if self.candidate_union_mode != "single_model":
            parts.append(f"union_{self.candidate_union_mode}")
        return "_".join(parts)


def parse_hu_turn1_topk_confirm_configs(value: str) -> list[HuTurn1TopKConfirmConfig]:
    configs: list[HuTurn1TopKConfirmConfig] = []
    for item in [part.strip() for part in value.split(",") if part.strip()]:
        top_k: int | None = None
        mc_samples: int | None = None
        min_delta: float | None = None
        confirm_mc_samples = 0
        confirm_se_multiplier = 0.0
        max_confirm_se: float | None = None
        min_predicted_delta: float | None = None
        safe_selector_threshold: float | None = None
        safe_selector_threshold_first: float | None = None
        safe_selector_threshold_second: float | None = None
        allowed_seats: tuple[str, ...] = ()
        candidate_topk = 0
        candidate_union_cap = 0
        candidate_union_mode = "single_model"
        for token in item.split("/"):
            if not token:
                continue
            key, sep, raw_value = token.partition("=")
            if key.startswith("k") and not sep:
                top_k = int(float(key[1:]))
            elif key in {"k", "topk", "top_k"} and sep:
                top_k = int(float(raw_value))
            elif key.startswith("mc") and not sep:
                mc_samples = int(float(key[2:]))
            elif key in {"mc", "mc_samples", "stage_a_mc"} and sep:
                mc_samples = int(float(raw_value))
            elif key.startswith("d") and not sep:
                min_delta = float(key[1:])
            elif key in {"d", "delta", "min_delta"} and sep:
                min_delta = float(raw_value)
            elif key.startswith("confirm") and not sep:
                confirm_mc_samples = int(float(key[7:]))
            elif key.startswith("cmc") and not sep:
                confirm_mc_samples = int(float(key[3:]))
            elif key in {"confirm", "confirm_mc", "confirm_mc_samples", "stage_b_mc"} and sep:
                confirm_mc_samples = int(float(raw_value))
            elif key.startswith("csemax") and not sep:
                max_confirm_se = float(key[6:])
            elif key.startswith("cse") and not sep:
                confirm_se_multiplier = float(key[3:])
            elif key in {"cse", "confirm_se", "confirm_se_multiplier", "stage_b_se"} and sep:
                confirm_se_multiplier = float(raw_value)
            elif key in {"csemax", "confirm_se_max", "max_confirm_se", "stage_b_se_max"} and sep:
                max_confirm_se = float(raw_value)
            elif key.startswith("pd") and not sep:
                min_predicted_delta = float(key[2:])
            elif key in {"pd", "predicted_delta", "min_predicted_delta"} and sep:
                min_predicted_delta = float(raw_value)
            elif key.startswith("safefirst") and not sep:
                safe_selector_threshold_first = float(key[9:])
            elif key in {
                "safefirst",
                "safe_first",
                "safe_selector_threshold_first",
                "first_safe",
            } and sep:
                safe_selector_threshold_first = float(raw_value)
            elif key.startswith("safesecond") and not sep:
                safe_selector_threshold_second = float(key[10:])
            elif key in {
                "safesecond",
                "safe_second",
                "safe_selector_threshold_second",
                "second_safe",
            } and sep:
                safe_selector_threshold_second = float(raw_value)
            elif key.startswith("safe") and not sep:
                safe_selector_threshold = float(key[4:])
            elif key in {"safe", "safe_threshold", "safe_selector", "safe_selector_threshold"} and sep:
                safe_selector_threshold = float(raw_value)
            elif key.startswith("seat") and not sep:
                allowed_seats = tuple(part for part in key[4:].replace("_", "+").split("+") if part)
            elif key in {"seat", "seats", "allowed_seats"} and sep:
                allowed_seats = tuple(part for part in raw_value.replace("_", "+").split("+") if part)
            elif key.startswith("ctopk") and not sep:
                candidate_topk = int(float(key[5:]))
            elif key.startswith("mk") and not sep:
                candidate_topk = int(float(key[2:]))
            elif key in {"ctopk", "candidate_topk", "model_topk", "modeltopk", "mk"} and sep:
                candidate_topk = int(float(raw_value))
            elif key.startswith("cap") and not sep:
                candidate_union_cap = int(float(key[3:]))
            elif key in {"cap", "union_cap", "candidate_union_cap"} and sep:
                candidate_union_cap = int(float(raw_value))
            elif key in {"union", "union_mode", "candidate_union_mode"} and sep:
                candidate_union_mode = raw_value.strip()
            else:
                raise ValueError(f"unknown HU T1 TopK config token in {item}: {token}")
        if top_k is None or mc_samples is None or min_delta is None:
            raise ValueError(f"config must include k, mc, and d: {item}")
        if top_k <= 0 or mc_samples <= 0:
            raise ValueError(f"k and mc must be positive: {item}")
        if candidate_topk < 0 or candidate_union_cap < 0:
            raise ValueError(f"candidate_topk and candidate_union_cap must be non-negative: {item}")
        if candidate_union_mode not in {
            "single_model",
            "min_rank",
            "rank_sum",
            "reciprocal_rank_sum",
            "mean_score",
            "max_score",
            "mean_z_score",
            "max_z_score",
        }:
            raise ValueError(f"unknown HU T1 candidate union mode in {item}: {candidate_union_mode}")
        configs.append(
            HuTurn1TopKConfirmConfig(
                top_k=top_k,
                mc_samples=mc_samples,
                min_delta=min_delta,
                confirm_mc_samples=confirm_mc_samples,
                confirm_se_multiplier=confirm_se_multiplier,
                max_confirm_se=max_confirm_se,
                min_predicted_delta=min_predicted_delta,
                safe_selector_threshold=safe_selector_threshold,
                safe_selector_threshold_first=safe_selector_threshold_first,
                safe_selector_threshold_second=safe_selector_threshold_second,
                allowed_seats=allowed_seats,
                candidate_topk=candidate_topk,
                candidate_union_cap=candidate_union_cap,
                candidate_union_mode=candidate_union_mode,
            )
        )
    if not configs:
        raise ValueError("at least one HU T1 TopK config is required")
    return configs


class HuTurn1TopKConfirmPolicy(HuTurn2Stage8bTopKMcRerankPolicy):
    """Use HU T1 model as a candidate generator, then rerank with independent MC."""

    def __init__(
        self,
        *,
        hu_turn1_candidate_model: object | None,
        hu_turn1_candidate_models: Sequence[object] | None = None,
        hu_turn1_safe_selector_model: object | None = None,
        topk_confirm_config: HuTurn1TopKConfirmConfig,
        rollout_policy_factory: RolloutPolicyFactory | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("hu_turn1_model", None)
        kwargs.pop("hu_turn1_min_margin", None)
        super().__init__(hu_turn1_model=None, hu_turn1_min_margin=None, **kwargs)
        self.hu_turn1_candidate_model = hu_turn1_candidate_model
        candidate_models = tuple(hu_turn1_candidate_models or ())
        if not candidate_models and hu_turn1_candidate_model is not None:
            candidate_models = (hu_turn1_candidate_model,)
        self.hu_turn1_candidate_models = candidate_models
        self.hu_turn1_safe_selector_model = hu_turn1_safe_selector_model
        self.hu_turn1_topk_confirm_config = topk_confirm_config
        self.hu_turn1_rollout_policy_factory = rollout_policy_factory

    def choose_action_observation(
        self,
        observation: ActorObservation,
        *,
        hand_id: str | int | None = None,
        game_id: str | int | None = None,
        decision_seed: int | None = None,
    ) -> Action:
        """Preserve the validated policy observation through the T1 selector."""
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
            policy_observation=observation,
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
        policy_observation: ActorObservation | None = None,
    ) -> Action:
        dealt = tuple(dealt_cards)
        if (
            board.card_count() == 5
            and self.hu_turn1_candidate_models
            and opponent_board is not None
        ):
            action = self._choose_hu_turn1_topk_confirm_action(
                board,
                dealt,
                dead_cards=tuple(dead_cards),
                opponent_board=opponent_board,
                hand_id=hand_id,
                game_id=game_id,
                decision_seed=decision_seed,
                street=street,
                policy_observation=policy_observation,
            )
            if action is not None:
                return action
        return super().choose_action(
            board,
            dealt,
            dead_cards=dead_cards,
            opponent_board=opponent_board,
            hand_id=hand_id,
            game_id=game_id,
            decision_seed=decision_seed,
            street=street,
        )

    def _choose_hu_turn1_topk_confirm_action(
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
        policy_observation: ActorObservation | None,
    ) -> Action | None:
        started_at = time.perf_counter()
        config = self.hu_turn1_topk_confirm_config
        actions = generate_turn_actions(board, dealt)
        if not actions:
            return None

        fallback_sample = policy_sample(board, dealt, actions)
        fallback_index = self._safe_choose_index(
            self.turn1_model, fallback_sample, actions
        )
        final_index = fallback_index
        candidate_index: int | None = None
        stage_a_best_index: int | None = None
        no_override_reason = ""
        predicted_margin: float | None = None
        candidate_score: float | None = None
        fallback_score: float | None = None
        stage_a_delta: float | None = None
        stage_a_delta_se: float | None = None
        confirm_delta: float | None = None
        confirm_delta_se: float | None = None
        confirm_delta_count = 0
        safe_selector_score: float | None = None
        safe_selector_threshold = config.safe_selector_threshold_for_seat(self.seat)
        stage_a_latency_ms = 0.0
        confirm_latency_ms = 0.0
        evaluated_action_count = 0
        legality_check_result = "legal"
        predictions: list[float] | None = None
        candidate_model_count = len(self.hu_turn1_candidate_models)
        candidate_union_mode = config.candidate_union_mode
        candidate_topk = config.candidate_topk or config.top_k
        candidate_union_cap = config.candidate_union_cap or config.top_k
        candidate_union_size = 0
        candidate_predict_reason = ""

        to_act_order = "second" if opponent_board.card_count() > board.card_count() else "first"
        if policy_observation is None:
            opponent_public = set(opponent_board.all_cards())
            hero_private_discards = tuple(
                card for card in dead_cards if card not in opponent_public
            )
            try:
                policy_observation = ActorObservation(
                    hero_board=board,
                    opponent_public_board=opponent_board,
                    dealt_cards=dealt,
                    hero_private_discards=hero_private_discards,
                    seat=self.seat,
                    street="T1",
                    to_act_order=to_act_order,
                )
            except ValueError:
                # Normal runtime enters through choose_action_observation. Keep
                # incomplete legacy direct-call fixtures fail-closed.
                policy_observation = None
        elif (
            policy_observation.hero_board != board
            or policy_observation.opponent_public_board != opponent_board
            or policy_observation.dealt_cards != dealt
            or policy_observation.seat != self.seat
            or policy_observation.street != "T1"
        ):
            raise ValueError("policy observation disagrees with the T1 decision state")
        if policy_observation is not None:
            to_act_order = policy_observation.to_act_order
        sample = hu_policy_sample(
            board,
            dealt,
            actions,
            opponent_board=opponent_board,
            dead_cards=dead_cards,
            seat=self.seat,
            to_act_order=to_act_order,
        )
        if fallback_index is None:
            no_override_reason = "fallback_to_self_board"
        elif config.allowed_seats and self.seat not in config.allowed_seats:
            no_override_reason = "seat_not_allowed"
        else:
            selection = self._select_hu_turn1_candidate_indices(
                sample,
                action_count=len(actions),
                fallback_index=fallback_index,
            )
            predictions = selection["scores"]
            candidate_model_count = int(selection["candidate_model_count"])
            candidate_union_mode = str(selection["candidate_union_mode"])
            candidate_topk = int(selection["candidate_topk"])
            candidate_union_cap = int(selection["candidate_union_cap"])
            candidate_union_size = int(selection["candidate_union_size"])
            candidate_predict_reason = str(selection["predict_reason"])
            if predictions is None:
                no_override_reason = candidate_predict_reason
                legality_check_result = candidate_predict_reason
            else:
                selected = list(selection["selected"])
                if not selected:
                    no_override_reason = "topk_empty"
                else:
                    stage_a_started = time.perf_counter()
                    stage_a = self._rerank_turn1_sample(
                        board=board,
                        dealt=dealt,
                        opponent_board=opponent_board,
                        dead_cards=dead_cards,
                        action_indices=sorted({fallback_index, *selected}),
                        seed=self._hu_turn1_future_rollout_seed(
                            board=board,
                            opponent_board=opponent_board,
                            dealt=dealt,
                            decision_seed=decision_seed,
                            hand_id=hand_id,
                            game_id=game_id,
                            phase="stage_a",
                        ),
                        future_samples=config.mc_samples,
                        observation=policy_observation,
                    )
                    stage_a_latency_ms = (time.perf_counter() - stage_a_started) * 1000.0
                    evaluated_action_count = int(stage_a.get("evaluated_action_count", 0) if stage_a else 0)
                    if not stage_a:
                        no_override_reason = "mc_rerank_failed"
                    else:
                        stage_a_actions = {
                            int(action.get("original_index", action.get("action_index", -1))): action
                            for action in list(stage_a.get("actions") or ())
                        }
                        best_action = (stage_a.get("actions") or [None])[0]
                        if not best_action or fallback_index not in stage_a_actions:
                            no_override_reason = "mc_rerank_missing_baseline"
                        else:
                            stage_a_best_index = int(
                                best_action.get("original_index", best_action.get("action_index", -1))
                            )
                            candidate_index = stage_a_best_index
                            best_ev = float(best_action.get("score", best_action.get("ev", 0.0)))
                            baseline_ev = float(stage_a_actions[fallback_index].get("score", 0.0))
                            stage_a_delta = best_ev - baseline_ev
                            stage_a_delta_se = math.sqrt(
                                float(best_action.get("se", 0.0)) ** 2
                                + float(stage_a_actions[fallback_index].get("se", 0.0)) ** 2
                            )
                            candidate_score = (
                                float(predictions[candidate_index])
                                if predictions is not None and candidate_index is not None
                                else None
                            )
                            fallback_score = float(predictions[fallback_index]) if predictions is not None else None
                            predicted_margin = (
                                candidate_score - fallback_score
                                if candidate_score is not None and fallback_score is not None
                                else None
                            )
                            if (
                                safe_selector_threshold is not None
                                and self.hu_turn1_safe_selector_model is not None
                            ):
                                safe_selector_score = self._safe_selector_score(
                                    board=board,
                                    opponent_board=opponent_board,
                                    dead_cards=dead_cards,
                                    dealt=dealt,
                                    actions=actions,
                                    candidate_index=stage_a_best_index,
                                    fallback_index=fallback_index,
                                    predicted_margin=predicted_margin,
                                    candidate_score=candidate_score,
                                    fallback_score=fallback_score,
                                    stage_a_delta=stage_a_delta,
                                    stage_a_delta_se=stage_a_delta_se,
                                    observation=policy_observation,
                                )
                            if stage_a_best_index == fallback_index:
                                no_override_reason = "mc_best_is_baseline"
                            elif config.confirm_mc_samples > 0:
                                confirm_started = time.perf_counter()
                                confirm = self._rerank_turn1_sample(
                                    board=board,
                                    dealt=dealt,
                                    opponent_board=opponent_board,
                                    dead_cards=dead_cards,
                                    action_indices=sorted({fallback_index, stage_a_best_index}),
                                    seed=self._hu_turn1_future_rollout_seed(
                                        board=board,
                                        opponent_board=opponent_board,
                                        dealt=dealt,
                                        decision_seed=decision_seed,
                                        hand_id=hand_id,
                                        game_id=game_id,
                                        phase="stage_b_confirm",
                                    ),
                                    future_samples=config.confirm_mc_samples,
                                    paired_delta_candidate_index=stage_a_best_index,
                                    paired_delta_baseline_index=fallback_index,
                                    observation=policy_observation,
                                )
                                confirm_latency_ms = (time.perf_counter() - confirm_started) * 1000.0
                                if not confirm:
                                    no_override_reason = "confirm_mc_failed"
                                else:
                                    confirm_delta = float(confirm.get("paired_delta_mean", 0.0))
                                    confirm_delta_se = float(confirm.get("paired_delta_standard_error", 0.0))
                                    confirm_delta_count = int(confirm.get("paired_delta_count", 0) or 0)
                                    if confirm_delta < config.min_delta:
                                        no_override_reason = "below_confirm_delta"
                                    elif config.max_confirm_se is not None and confirm_delta_se > config.max_confirm_se:
                                        no_override_reason = "above_confirm_se"
                                    elif (
                                        config.confirm_se_multiplier > 0.0
                                        and confirm_delta < config.confirm_se_multiplier * confirm_delta_se
                                    ):
                                        no_override_reason = "below_confirm_se"
                                    else:
                                        if (
                                            safe_selector_threshold is not None
                                            and self.hu_turn1_safe_selector_model is None
                                        ):
                                            no_override_reason = "safe_selector_unavailable"
                                            legality_check_result = "safe_selector_unavailable"
                                        elif safe_selector_threshold is not None:
                                            safe_selector_score = self._safe_selector_score(
                                                board=board,
                                                opponent_board=opponent_board,
                                                dead_cards=dead_cards,
                                                dealt=dealt,
                                                actions=actions,
                                                candidate_index=stage_a_best_index,
                                                fallback_index=fallback_index,
                                                predicted_margin=predicted_margin,
                                                candidate_score=candidate_score,
                                                fallback_score=fallback_score,
                                                stage_a_delta=stage_a_delta,
                                                stage_a_delta_se=stage_a_delta_se,
                                                confirm_delta=confirm_delta,
                                                confirm_delta_se=confirm_delta_se,
                                                confirm_delta_count=confirm_delta_count,
                                                observation=policy_observation,
                                            )
                                            if safe_selector_score is None:
                                                no_override_reason = "safe_selector_failed"
                                                legality_check_result = "safe_selector_failed"
                                            elif safe_selector_score < safe_selector_threshold:
                                                no_override_reason = "below_safe_selector"
                                        if no_override_reason:
                                            pass
                                        else:
                                            legality_check_result = self._candidate_legality_result(
                                                board,
                                                actions,
                                                stage_a_best_index,
                                            )
                                            if legality_check_result == "legal":
                                                final_index = stage_a_best_index
                                            else:
                                                no_override_reason = "illegal_candidate"
                            elif stage_a_delta is not None and stage_a_delta >= config.min_delta:
                                if (
                                    safe_selector_threshold is not None
                                    and self.hu_turn1_safe_selector_model is None
                                ):
                                    no_override_reason = "safe_selector_unavailable"
                                    legality_check_result = "safe_selector_unavailable"
                                elif safe_selector_threshold is not None:
                                    if safe_selector_score is None:
                                        no_override_reason = "safe_selector_failed"
                                        legality_check_result = "safe_selector_failed"
                                    elif safe_selector_score < safe_selector_threshold:
                                        no_override_reason = "below_safe_selector"
                                if no_override_reason:
                                    pass
                                else:
                                    legality_check_result = self._candidate_legality_result(
                                        board,
                                        actions,
                                        stage_a_best_index,
                                    )
                                    if legality_check_result == "legal":
                                        final_index = stage_a_best_index
                                    else:
                                        no_override_reason = "illegal_candidate"
                            else:
                                no_override_reason = "below_rerank_delta"

        if final_index is None:
            return None

        override_fired = fallback_index is not None and final_index != fallback_index
        context = self.decision_context or {}
        visible_dead_cards = list(dead_cards)
        opponent_public = set(opponent_board.all_cards())
        hero_private_discards = [
            card for card in dead_cards if card not in opponent_public
        ]
        self._log_hu_turn1_decision(
            {
                "schema": "hu_turn1_topk_confirm_decision_v1",
                "phase": "hu_turn1_5card",
                "runtime_profile": context.get("runtime_profile", "hu_turn1_topk_confirm"),
                "runtime_status": context.get("runtime_status", "t1_topk_confirm_validation"),
                "hand_id": hand_id,
                "game_id": game_id,
                "decision_seed": decision_seed,
                "street": street,
                "seat": self.seat,
                "to_act_order": to_act_order,
                "config_id": config.config_id,
                "top_k": config.top_k,
                "mc_samples": config.mc_samples,
                "confirm_mc_samples": config.confirm_mc_samples,
                "confirm_se_multiplier": config.confirm_se_multiplier,
                "max_confirm_se": config.max_confirm_se,
                "min_predicted_delta": config.min_predicted_delta,
                "candidate_model_count": candidate_model_count,
                "candidate_union_mode": candidate_union_mode,
                "candidate_topk": candidate_topk,
                "candidate_union_cap": candidate_union_cap,
                "candidate_union_size": candidate_union_size,
                "candidate_predict_reason": candidate_predict_reason,
                "candidate_predicted_delta_filter_applied": candidate_union_mode == "single_model",
                "board": _board_to_json(board),
                "hero_board": _board_to_json(board),
                "opponent_board": _board_to_json(opponent_board),
                "dead_cards": list(dead_cards),
                "visibility_model": "actor_observation_v1",
                "discard_visibility": "own_private_only",
                "visible_dead_cards": visible_dead_cards,
                "true_dead_cards": [],
                "hero_private_discards": hero_private_discards,
                "policy_observation": (
                    policy_observation.to_dict()
                    if policy_observation is not None
                    else None
                ),
                "observation_fingerprint": (
                    policy_observation.fingerprint()
                    if policy_observation is not None
                    else None
                ),
                "replay_ready": False,
                "action_key_schema": ACTION_KEY_SCHEMA,
                "legal_action_set_digest": legal_action_set_digest(actions),
                "legal_action_order_digest": ordered_action_mapping_digest(actions),
                "cards_to_place": list(dealt),
                "baseline_action": action_to_json(board, actions[fallback_index]) if fallback_index is not None else None,
                "stage_a_best_action": (
                    action_to_json(board, actions[stage_a_best_index])
                    if stage_a_best_index is not None and 0 <= stage_a_best_index < len(actions)
                    else None
                ),
                "hu_turn1_action": (
                    action_to_json(board, actions[candidate_index])
                    if candidate_index is not None and 0 <= candidate_index < len(actions)
                    else None
                ),
                "fallback_action": action_to_json(board, actions[fallback_index]) if fallback_index is not None else None,
                "final_action": action_to_json(board, actions[final_index]),
                "baseline_action_index": fallback_index,
                "candidate_action_index": candidate_index,
                "stage_a_best_index": stage_a_best_index,
                "final_action_index": final_index,
                "selected_action_indices": selected if "selected" in locals() else [],
                "selected_action_keys": [
                    action_key(actions[index]).to_token()
                    for index in (selected if "selected" in locals() else [])
                ],
                "baseline_action_key": (
                    action_key(actions[fallback_index]).to_token()
                    if fallback_index is not None
                    else None
                ),
                "candidate_action_key": (
                    action_key(actions[candidate_index]).to_token()
                    if candidate_index is not None and 0 <= candidate_index < len(actions)
                    else None
                ),
                "stage_a_best_action_key": (
                    action_key(actions[stage_a_best_index]).to_token()
                    if stage_a_best_index is not None and 0 <= stage_a_best_index < len(actions)
                    else None
                ),
                "final_action_key": action_key(actions[final_index]).to_token(),
                "override_fired": override_fired,
                "no_override_reason": no_override_reason or ("override_fired" if override_fired else ""),
                "hu_turn1_min_margin": None,
                "hu_turn1_predicted_margin": predicted_margin,
                "candidate_score": candidate_score,
                "fallback_score": fallback_score,
                "model_score": candidate_score,
                "stage_a_delta": stage_a_delta,
                "stage_a_delta_se": stage_a_delta_se,
                "confirm_delta": confirm_delta,
                "confirm_delta_se": confirm_delta_se,
                "confirm_delta_count": confirm_delta_count,
                "safe_selector_score": safe_selector_score,
                "safe_selector_threshold": safe_selector_threshold,
                "safe_selector_threshold_global": config.safe_selector_threshold,
                "safe_selector_threshold_first": config.safe_selector_threshold_first,
                "safe_selector_threshold_second": config.safe_selector_threshold_second,
                "evaluated_action_count": evaluated_action_count,
                "legality_check_result": legality_check_result,
                "mc_rerank_latency_ms": stage_a_latency_ms,
                "confirm_mc_latency_ms": confirm_latency_ms,
                "runtime_latency_ms": self._elapsed_ms(started_at),
            }
        )
        return actions[final_index]

    def _select_hu_turn1_candidate_indices(
        self,
        sample: dict[str, Any],
        *,
        action_count: int,
        fallback_index: int,
    ) -> dict[str, Any]:
        config = self.hu_turn1_topk_confirm_config
        models = self.hu_turn1_candidate_models
        candidate_topk = config.candidate_topk or config.top_k
        candidate_union_cap = config.candidate_union_cap or config.top_k
        action_sort_keys = [
            action_key_from_payload(payload).sort_key() for payload in sample["actions"]
        ]
        if not models:
            return {
                "scores": None,
                "selected": [],
                "predict_reason": "model_load_failed",
                "candidate_model_count": 0,
                "candidate_union_mode": config.candidate_union_mode,
                "candidate_topk": candidate_topk,
                "candidate_union_cap": candidate_union_cap,
                "candidate_union_size": 0,
            }

        if len(models) == 1 and config.candidate_union_mode == "single_model":
            predictions, predict_reason = self._safe_predict(models[0], sample, action_count)
            if predictions is None:
                return {
                    "scores": None,
                    "selected": [],
                    "predict_reason": predict_reason,
                    "candidate_model_count": 1,
                    "candidate_union_mode": "single_model",
                    "candidate_topk": candidate_topk,
                    "candidate_union_cap": candidate_union_cap,
                    "candidate_union_size": 0,
                }
            order = sorted(
                range(action_count),
                key=lambda index: (-float(predictions[index]), action_sort_keys[index]),
            )
            selected = self._filter_candidate_order(
                order,
                scores=predictions,
                fallback_index=fallback_index,
                cap=config.top_k,
                apply_predicted_delta_filter=True,
            )
            return {
                "scores": predictions,
                "selected": selected,
                "predict_reason": "",
                "candidate_model_count": 1,
                "candidate_union_mode": "single_model",
                "candidate_topk": candidate_topk,
                "candidate_union_cap": candidate_union_cap,
                "candidate_union_size": len(order),
            }

        predictions_by_model: list[list[float]] = []
        rank_by_model: list[dict[int, int]] = []
        score_by_model: list[dict[int, float]] = []
        z_score_by_model: list[dict[int, float]] = []
        union_ranks: dict[int, int] = {}
        for model in models:
            predictions, predict_reason = self._safe_predict(model, sample, action_count)
            if predictions is None:
                return {
                    "scores": None,
                    "selected": [],
                    "predict_reason": predict_reason,
                    "candidate_model_count": len(models),
                    "candidate_union_mode": config.candidate_union_mode,
                    "candidate_topk": candidate_topk,
                    "candidate_union_cap": candidate_union_cap,
                    "candidate_union_size": 0,
                }
            predictions_by_model.append(predictions)
            order = sorted(
                range(action_count),
                key=lambda index: (-float(predictions[index]), action_sort_keys[index]),
            )
            rank_map = {index: rank for rank, index in enumerate(order)}
            score_map = {index: float(predictions[index]) for index in range(action_count)}
            z_values = _z_scores([float(value) for value in predictions])
            z_score_map = {index: float(z_values[index]) for index in range(action_count)}
            rank_by_model.append(rank_map)
            score_by_model.append(score_map)
            z_score_by_model.append(z_score_map)
            for rank, index in enumerate(order[: min(candidate_topk, action_count)]):
                previous = union_ranks.get(index)
                if previous is None or rank < previous:
                    union_ranks[index] = rank

        if not union_ranks:
            return {
                "scores": [0.0 for _index in range(action_count)],
                "selected": [],
                "predict_reason": "",
                "candidate_model_count": len(models),
                "candidate_union_mode": config.candidate_union_mode,
                "candidate_topk": candidate_topk,
                "candidate_union_cap": candidate_union_cap,
                "candidate_union_size": 0,
            }

        aggregate_scores = self._aggregate_candidate_scores(
            action_count=action_count,
            union_mode=config.candidate_union_mode,
            union_ranks=union_ranks,
            rank_by_model=rank_by_model,
            score_by_model=score_by_model,
            z_score_by_model=z_score_by_model,
        )
        order = self._candidate_union_order(
            union_mode=config.candidate_union_mode,
            union_ranks=union_ranks,
            rank_by_model=rank_by_model,
            score_by_model=score_by_model,
            z_score_by_model=z_score_by_model,
            action_count=action_count,
            action_sort_keys=action_sort_keys,
        )
        selected = self._filter_candidate_order(
            order,
            scores=aggregate_scores,
            fallback_index=fallback_index,
            cap=candidate_union_cap,
            apply_predicted_delta_filter=False,
        )
        return {
            "scores": aggregate_scores,
            "selected": selected,
            "predict_reason": "",
            "candidate_model_count": len(models),
            "candidate_union_mode": config.candidate_union_mode,
            "candidate_topk": candidate_topk,
            "candidate_union_cap": candidate_union_cap,
            "candidate_union_size": len(union_ranks),
        }

    def _filter_candidate_order(
        self,
        order: Sequence[int],
        *,
        scores: Sequence[float],
        fallback_index: int,
        cap: int,
        apply_predicted_delta_filter: bool,
    ) -> list[int]:
        config = self.hu_turn1_topk_confirm_config
        selected: list[int] = []
        fallback_score = float(scores[fallback_index])
        for index in order:
            index = int(index)
            if index == fallback_index:
                continue
            if apply_predicted_delta_filter and config.min_predicted_delta is not None:
                if float(scores[index]) - fallback_score < config.min_predicted_delta:
                    continue
            selected.append(index)
            if len(selected) >= cap:
                break
        return selected

    def _candidate_union_order(
        self,
        *,
        union_mode: str,
        union_ranks: dict[int, int],
        rank_by_model: list[dict[int, int]],
        score_by_model: list[dict[int, float]],
        z_score_by_model: list[dict[int, float]],
        action_count: int,
        action_sort_keys: Sequence[tuple[int, int, int, int]],
    ) -> list[int]:
        if union_mode in {"single_model", "min_rank"}:
            sort_key = lambda index: (union_ranks[index], action_sort_keys[index])
        elif union_mode == "rank_sum":
            penalty_rank = action_count
            sort_key = lambda index: (
                sum(rank_map.get(index, penalty_rank) for rank_map in rank_by_model),
                union_ranks[index],
                action_sort_keys[index],
            )
        elif union_mode == "reciprocal_rank_sum":
            sort_key = lambda index: (
                -sum(1.0 / (rank_map[index] + 1.0) for rank_map in rank_by_model if index in rank_map),
                union_ranks[index],
                action_sort_keys[index],
            )
        elif union_mode == "mean_score":
            sort_key = lambda index: (
                -_mean([score_map[index] for score_map in score_by_model]),
                union_ranks[index],
                action_sort_keys[index],
            )
        elif union_mode == "max_score":
            sort_key = lambda index: (
                -max(score_map[index] for score_map in score_by_model),
                union_ranks[index],
                action_sort_keys[index],
            )
        elif union_mode == "mean_z_score":
            sort_key = lambda index: (
                -_mean([score_map[index] for score_map in z_score_by_model]),
                union_ranks[index],
                action_sort_keys[index],
            )
        elif union_mode == "max_z_score":
            sort_key = lambda index: (
                -max(score_map[index] for score_map in z_score_by_model),
                union_ranks[index],
                action_sort_keys[index],
            )
        else:
            raise ValueError(f"unknown HU T1 candidate union mode: {union_mode}")
        return sorted(union_ranks, key=sort_key)

    def _aggregate_candidate_scores(
        self,
        *,
        action_count: int,
        union_mode: str,
        union_ranks: dict[int, int],
        rank_by_model: list[dict[int, int]],
        score_by_model: list[dict[int, float]],
        z_score_by_model: list[dict[int, float]],
    ) -> list[float]:
        penalty_rank = action_count
        scores: list[float] = []
        for index in range(action_count):
            if union_mode in {"single_model", "min_rank"}:
                scores.append(float(-union_ranks.get(index, penalty_rank)))
            elif union_mode == "rank_sum":
                scores.append(
                    float(-sum(rank_map.get(index, penalty_rank) for rank_map in rank_by_model))
                )
            elif union_mode == "reciprocal_rank_sum":
                scores.append(
                    float(sum(1.0 / (rank_map[index] + 1.0) for rank_map in rank_by_model))
                )
            elif union_mode == "mean_score":
                scores.append(_mean([score_map[index] for score_map in score_by_model]))
            elif union_mode == "max_score":
                scores.append(max(score_map[index] for score_map in score_by_model))
            elif union_mode == "mean_z_score":
                scores.append(_mean([score_map[index] for score_map in z_score_by_model]))
            elif union_mode == "max_z_score":
                scores.append(max(score_map[index] for score_map in z_score_by_model))
            else:
                raise ValueError(f"unknown HU T1 candidate union mode: {union_mode}")
        return scores

    def _safe_selector_score(
        self,
        *,
        board: Board,
        opponent_board: Board,
        dead_cards: tuple[str, ...],
        dealt: tuple[str, ...],
        actions: Sequence[Action],
        candidate_index: int,
        fallback_index: int,
        predicted_margin: float | None,
        candidate_score: float | None,
        fallback_score: float | None,
        stage_a_delta: float | None = None,
        stage_a_delta_se: float | None = None,
        confirm_delta: float | None = None,
        confirm_delta_se: float | None = None,
        confirm_delta_count: int = 0,
        observation: ActorObservation | None = None,
    ) -> float | None:
        if self.hu_turn1_safe_selector_model is None:
            return None
        try:
            row = {
                "seat": self.seat,
                "hero_board": _board_to_json(board),
                "opponent_board": _board_to_json(opponent_board),
                "dead_cards": list(dead_cards),
                "cards_to_place": list(dealt),
                "hu_turn1_action": action_to_json(board, actions[candidate_index]),
                "baseline_action": action_to_json(board, actions[fallback_index]),
                "hu_turn1_predicted_margin": predicted_margin,
                "candidate_score": candidate_score,
                "fallback_score": fallback_score,
                "action_count": len(actions),
                "stage_a_delta": stage_a_delta,
                "stage_a_delta_se": stage_a_delta_se,
                "confirm_delta": confirm_delta,
                "confirm_delta_se": confirm_delta_se,
                "confirm_delta_count": confirm_delta_count,
                "policy_observation": (
                    observation.to_dict() if observation is not None else None
                ),
                "observation_fingerprint": (
                    observation.fingerprint() if observation is not None else None
                ),
            }
            return score_hu_turn1_safe_selector(self.hu_turn1_safe_selector_model, row)
        except Exception:
            return None

    def _rerank_turn1_sample(
        self,
        *,
        board: Board,
        dealt: tuple[str, ...],
        opponent_board: Board,
        dead_cards: tuple[str, ...],
        action_indices: Sequence[int],
        seed: int,
        future_samples: int,
        paired_delta_candidate_index: int | None = None,
        paired_delta_baseline_index: int | None = None,
        observation: ActorObservation | None = None,
    ) -> dict[str, Any]:
        hero_player = 0 if self.seat == "first" else 1
        opponent_public = set(opponent_board.all_cards())
        hero_private_discards = tuple(
            card for card in dead_cards if card not in opponent_public
        )
        if observation is None:
            observation = ActorObservation(
                hero_board=board,
                opponent_public_board=opponent_board,
                dealt_cards=dealt,
                hero_private_discards=hero_private_discards,
                seat=self.seat,  # type: ignore[arg-type]
                street="T1",
                to_act_order=(
                    "second" if opponent_board.card_count() > board.card_count() else "first"
                ),
            )
        belief_batch = sample_hidden_card_particles(
            observation,
            base_seed=seed,
            run_id="hu_turn1_topk_confirm",
            sample_count=future_samples,
        )
        policies = [
            self._build_rollout_policy(seed=seed * 4 + 1, seat="first"),
            self._build_rollout_policy(seed=seed * 4 + 2, seat="second"),
        ]
        return evaluate_turn1_action_subset(
            board=board,
            opponent_board=opponent_board,
            dealt=dealt,
            hero_player=hero_player,
            policies=policies,
            hand_seed=_stable_int_seed(self.decision_context.get("hand_seed", seed)),
            sample_id=0,
            future_samples=future_samples,
            action_indices=action_indices,
            paired_delta_candidate_index=paired_delta_candidate_index,
            paired_delta_baseline_index=paired_delta_baseline_index,
            rng=random.Random(seed),
            observation=observation,
            belief_batch=belief_batch,
        )

    def _build_rollout_policy(self, *, seed: int, seat: str) -> RegularAiPolicy:
        if self.hu_turn1_rollout_policy_factory is not None:
            return self.hu_turn1_rollout_policy_factory(seed=seed, seat=seat)
        return RegularAiPolicy(
            opening_model=self.opening_model,
            turn1_model=self.turn1_model,
            turn2_model=self.turn2_model,
            turn3_model=self.turn3_model,
            seed=seed,
            seat=seat,
            opening_lookahead_samples=self.opening_lookahead_samples,
        )

    def _hu_turn1_future_rollout_seed(
        self,
        *,
        board: Board,
        opponent_board: Board,
        dealt: tuple[str, ...],
        decision_seed: int | None,
        hand_id: str | int | None,
        game_id: str | int | None,
        phase: str,
    ) -> int:
        payload = {
            "phase": phase,
            "decision_seed": decision_seed if decision_seed is not None else self.seed,
            "hand_id": hand_id,
            "game_id": game_id,
            "seat": self.seat,
            "board": _board_to_json(board),
            "opponent": _board_to_json(opponent_board),
            "dealt": list(dealt),
            "config_id": self.hu_turn1_topk_confirm_config.config_id,
        }
        data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


def _stable_int_seed(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        data = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
        return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _z_scores(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    mean = _mean(values)
    variance = sum((float(value) - mean) ** 2 for value in values) / len(values)
    if variance <= 0.0:
        return [0.0 for _value in values]
    std = math.sqrt(variance)
    return [(float(value) - mean) / std for value in values]


def _board_to_json(board: Board) -> dict[str, list[str]]:
    return {
        "top": list(board.top),
        "middle": list(board.middle),
        "bottom": list(board.bottom),
    }
