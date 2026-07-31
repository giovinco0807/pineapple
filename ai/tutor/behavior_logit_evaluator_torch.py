"""Direct pre-temperature logit extraction for frozen HU policy-value models.

The behavior calibration contract must observe the checkpoint logits before
temperature scaling and before Q32 probability quantization.  This module is
kept separate from :mod:`frozen_behavior_torch` so adding the calibration
reader does not silently change the adapter source hash already embedded in
existing range and smoke artifacts.

Only the four exact-hash T1/T2 HU PolicyValueNet assets are exposed by the
convenience builder.  Every extraction regenerates the complete legal action
set, binds the caller's action order to the ``BehaviorInfoSet``, and selects
the corresponding semantic 27-way logits directly from the CPU model.
"""
from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ai.engine.action_space import get_semantic_action_index, get_turn_actions
from ai.engine.encoding import Board, Observation, encode_state
from ai.tutor.exact_late import action_key
from ai.tutor.frozen_behavior_torch import (
    DEFAULT_QUANTIZATION_DENOMINATOR,
    KNOWN_HU_POLICY_VALUE_ASSETS,
    TorchPolicyValueBehaviorModel,
    _sha256_file,
)
from ai.tutor.t3_hu_full_card_range import BehaviorInfoSet


EVALUATOR_VERSION = "torch_policyvalue_pre_temperature_legal_logits_v1"


def _board(
    rows: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]],
) -> Board:
    return Board(top=list(rows[0]), middle=list(rows[1]), bottom=list(rows[2]))


class TorchPolicyValueLegalLogitEvaluator(TorchPolicyValueBehaviorModel):
    """Identity-temperature PolicyValueNet with direct legal-logit access.

    The inherited behavior distribution remains available for an exact
    end-to-end cross-check.  Calibration itself calls
    :meth:`pre_temperature_legal_logits` and therefore never attempts to
    recover logits from the quantized distribution.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        model_id: str,
        supported_turns: Sequence[int],
        supported_actors: Sequence[str],
        training_scope_id: str,
        quantization_denominator: int = DEFAULT_QUANTIZATION_DENOMINATOR,
        expected_checkpoint_sha256: str | None = None,
    ) -> None:
        super().__init__(
            checkpoint_path,
            model_id=model_id,
            supported_turns=supported_turns,
            supported_actors=supported_actors,
            training_scope_id=training_scope_id,
            temperature=Fraction(1, 1),
            quantization_denominator=quantization_denominator,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
        )
        self._row_extractor_sha256 = _sha256_file(Path(__file__).resolve())

    @property
    def row_extractor_sha256(self) -> str:
        return self._row_extractor_sha256

    @property
    def adapter_source_sha256(self) -> str:
        value = self.model_manifest.get("adapter_source_sha256")
        if not isinstance(value, str):  # pragma: no cover - constructor invariant
            raise AssertionError("frozen behavior manifest lost its adapter hash")
        return value

    @property
    def evaluator_manifest(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": "ofc_behavior_logit_evaluator/v1",
                "evaluator_version": EVALUATOR_VERSION,
                "checkpoint_sha256": self.checkpoint_sha256,
                "model_sha256": self.model_sha256,
                "row_extractor_sha256": self.row_extractor_sha256,
                "adapter_source_sha256": self.adapter_source_sha256,
                "temperature": "1/1",
                "device": "cpu",
                "logit_contract": (
                    "checkpoint_policy_head_pre_temperature_pre_quantization"
                ),
            }
        )

    def pre_temperature_legal_logits(
        self,
        information: BehaviorInfoSet,
        legal_action_ids: tuple[str, ...],
    ) -> tuple[float, ...]:
        if not isinstance(information, BehaviorInfoSet):
            raise TypeError("information must be a BehaviorInfoSet")
        if not isinstance(legal_action_ids, tuple) or any(
            not isinstance(action_id, str) for action_id in legal_action_ids
        ):
            raise TypeError("legal_action_ids must be a tuple of strings")
        if legal_action_ids != information.legal_action_ids:
            raise ValueError(
                "legal_action_ids must exactly match BehaviorInfoSet order"
            )
        if information.turn not in self._supported_turns:
            raise ValueError(f"behavior model does not support turn T{information.turn}")
        if information.actor not in self._supported_actors:
            raise ValueError(
                f"behavior model does not support actor {information.actor!r}"
            )
        if information.fantasy_state is not None:
            raise ValueError("Torch behavior evaluator supports normal hands only")

        board_self_rows = (
            information.board_bb
            if information.actor == "bb"
            else information.board_btn
        )
        board_opponent_rows = (
            information.board_btn
            if information.actor == "bb"
            else information.board_bb
        )
        board_self = _board(board_self_rows)
        current_draw = list(information.current_draw)
        legal_actions = get_turn_actions(current_draw, board_self)
        legal_by_id = {action_key(action): action for action in legal_actions}
        if set(legal_by_id) != set(legal_action_ids):
            raise ValueError("behavior information legal actions do not match regeneration")

        semantic_indices: dict[str, int] = {}
        for action_id in legal_action_ids:
            index = get_semantic_action_index(legal_by_id[action_id], current_draw)
            if not 0 <= index < self._output_dim:
                raise ValueError("behavior action semantic index exceeds checkpoint output")
            semantic_indices[action_id] = index
        if len(set(semantic_indices.values())) != len(semantic_indices):
            raise ValueError("behavior legal actions collide in semantic action space")

        observation = Observation(
            board_self=board_self,
            board_opponent=_board(board_opponent_rows),
            dealt_cards=current_draw,
            known_discards_self=[
                card for _turn, card in information.own_recall_before.discards_by_turn
            ],
            turn=information.turn,
            is_btn=information.actor == "btn",
            is_fl=False,
            opp_is_fl=False,
        )
        state = encode_state(observation)
        if not isinstance(state, np.ndarray) or state.ndim != 1:
            raise ValueError("behavior state encoder returned an invalid vector")
        if state.shape[0] < self._input_dim:
            raise ValueError("behavior state encoder is shorter than checkpoint input")
        state_tensor = torch.from_numpy(
            state[: self._input_dim].astype(np.float32, copy=False)
        ).unsqueeze(0)
        with torch.no_grad():
            logits, _value = self._model(state_tensor, masks=None)
            selected = tuple(
                float(logits[0, semantic_indices[action_id]].item())
                for action_id in legal_action_ids
            )
        if any(not math.isfinite(value) for value in selected):
            raise ValueError("checkpoint produced non-finite legal logits")
        return selected


def build_known_hu_policy_value_logit_evaluators(
    workspace_root: str | Path | None = None,
    *,
    quantization_denominator: int = DEFAULT_QUANTIZATION_DENOMINATOR,
) -> Mapping[tuple[int, str], TorchPolicyValueLegalLogitEvaluator]:
    """Load the exact-hash T1/T2 BB/BTN calibration evaluators."""
    root = (
        Path(workspace_root).resolve()
        if workspace_root is not None
        else Path(__file__).resolve().parents[2]
    )
    routes: dict[tuple[int, str], TorchPolicyValueLegalLogitEvaluator] = {}
    for (turn, actor), asset in sorted(KNOWN_HU_POLICY_VALUE_ASSETS.items()):
        routes[(turn, actor)] = TorchPolicyValueLegalLogitEvaluator(
            root / str(asset["relative_path"]),
            model_id=f"hu_t{turn}_{actor}_2m_policyvalue_prior",
            supported_turns=(turn,),
            supported_actors=(actor,),
            training_scope_id=f"hu_t{turn}_{actor}_visible_opponent_board_2m_v1",
            quantization_denominator=quantization_denominator,
            expected_checkpoint_sha256=str(asset["checkpoint_sha256"]),
        )
    return MappingProxyType(routes)


__all__ = [
    "EVALUATOR_VERSION",
    "TorchPolicyValueLegalLogitEvaluator",
    "build_known_hu_policy_value_logit_evaluators",
]
