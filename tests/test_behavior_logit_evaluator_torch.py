from fractions import Fraction
from pathlib import Path

import pytest
import torch

from ai.engine.action_space import get_turn_actions
from ai.engine.encoding import Board
from ai.tutor.behavior_logit_evaluator_torch import (
    TorchPolicyValueLegalLogitEvaluator,
)
from ai.tutor.behavior_temperature_calibration import (
    PreTemperatureLegalLogitEvaluator,
)
from ai.tutor.exact_late import action_key
from ai.tutor.frozen_behavior_torch import _quantize_largest_remainder
from ai.tutor.t3_hu_full_card_range import BehaviorInfoSet
from ai.tutor.t3_hu_public_cfr import PrivateRecall


BB_BOARD = (("2c", "3d", "4h"), ("7c", "8h"), ("Qc", "Kd"))
BTN_BOARD = (("5c", "6d", "8c"), ("9c",), ("As",))
DRAW = ("Ad", "Jd", "Qd")


def _information() -> BehaviorInfoSet:
    board = Board(
        top=list(BTN_BOARD[0]),
        middle=list(BTN_BOARD[1]),
        bottom=list(BTN_BOARD[2]),
    )
    legal = tuple(action_key(action) for action in get_turn_actions(list(DRAW), board))
    return BehaviorInfoSet(
        actor="btn",
        turn=1,
        board_bb=BB_BOARD,
        board_btn=BTN_BOARD,
        public_action_history=(),
        own_recall_before=PrivateRecall(),
        current_draw=DRAW,
        legal_action_ids=legal,
    )


def _real_evaluator() -> TorchPolicyValueLegalLogitEvaluator:
    checkpoint = Path("ai/data/t1_hu_btn_gcp_2m_model/t2_policyvalue_model.pt")
    if not checkpoint.is_file():
        pytest.skip("HU T1 BTN policy-value checkpoint is not present")
    return TorchPolicyValueLegalLogitEvaluator(
        checkpoint,
        model_id="hu_t1_btn_2m_policyvalue_prior",
        supported_turns=(1,),
        supported_actors=("btn",),
        training_scope_id="hu_t1_btn_visible_opponent_board_2m_v1",
        expected_checkpoint_sha256=(
            "064ab29967d4294f76f4ac844000b7fad97553eb341cfb54065f023bbfe7a32f"
        ),
    )


def test_direct_logits_reproduce_identity_temperature_distribution():
    evaluator = _real_evaluator()
    information = _information()

    assert isinstance(evaluator, PreTemperatureLegalLogitEvaluator)
    logits = evaluator.pre_temperature_legal_logits(
        information, information.legal_action_ids
    )
    probabilities = torch.softmax(torch.tensor(logits, dtype=torch.float64), dim=0)
    expected = _quantize_largest_remainder(
        {
            action_id: float(probability)
            for action_id, probability in zip(
                information.legal_action_ids, probabilities.tolist()
            )
        },
        denominator=evaluator.model_manifest["quantization_denominator"],
    )
    actual = evaluator.action_distribution(information)

    assert len(logits) == len(information.legal_action_ids)
    assert actual.probabilities == expected
    assert sum(actual.probabilities.values(), Fraction(0, 1)) == 1
    assert evaluator.adapter_source_sha256 == evaluator.model_manifest[
        "adapter_source_sha256"
    ]
    assert evaluator.evaluator_manifest["temperature"] == "1/1"
    assert len(evaluator.row_extractor_sha256) == 64


def test_direct_logits_reject_different_legal_action_order():
    evaluator = _real_evaluator()
    information = _information()
    reversed_ids = tuple(reversed(information.legal_action_ids))

    with pytest.raises(ValueError, match="exactly match"):
        evaluator.pre_temperature_legal_logits(information, reversed_ids)
