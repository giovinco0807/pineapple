from fractions import Fraction
from pathlib import Path

import pytest
import torch

from ai.engine.action_space import get_turn_actions
from ai.engine.encoding import Board
from ai.models.networks import PolicyNetwork
from ai.tutor.exact_late import action_key
from ai.tutor.frozen_behavior_torch import (
    DEFAULT_QUANTIZATION_DENOMINATOR,
    TorchPolicyBehaviorModel,
    TorchPolicyValueBehaviorModel,
    TurnActorBehaviorDispatch,
)
from ai.tutor.t3_hu_full_card_range import BehaviorInfoSet
from ai.tutor.t3_hu_public_cfr import PrivateRecall


BB_BOARD_T1_AFTER_FIRST = (
    ("2c", "3d", "4h"),
    ("7c", "8h"),
    ("Qc", "Kd"),
)
BTN_BOARD_T1 = (
    ("5c", "6d", "8c"),
    ("9c",),
    ("As",),
)
CURRENT_DRAW = ("Ad", "Jd", "Qd")


def _board(rows):
    return Board(top=list(rows[0]), middle=list(rows[1]), bottom=list(rows[2]))


def _information(*, actor="btn", turn=1, legal_action_ids=None):
    actor_rows = BTN_BOARD_T1 if actor == "btn" else BB_BOARD_T1_AFTER_FIRST
    actions = get_turn_actions(list(CURRENT_DRAW), _board(actor_rows))
    action_ids = tuple(action_key(action) for action in actions)
    return BehaviorInfoSet(
        actor=actor,
        turn=turn,
        board_bb=BB_BOARD_T1_AFTER_FIRST,
        board_btn=BTN_BOARD_T1,
        public_action_history=(),
        own_recall_before=PrivateRecall(),
        current_draw=CURRENT_DRAW,
        legal_action_ids=(action_ids if legal_action_ids is None else legal_action_ids),
    )


def _write_zero_checkpoint(path: Path, *, input_dim=520, output_dim=250):
    model = PolicyNetwork(input_dim=input_dim, max_actions=output_dim, dropout=0.0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    torch.save(model.state_dict(), path)


def _adapter(path: Path, **overrides):
    kwargs = {
        "model_id": "synthetic_t1_btn_behavior",
        "supported_turns": (1,),
        "supported_actors": ("btn",),
        "training_scope_id": "synthetic_complete_state_test_v1",
        "promotion_eligible": False,
    }
    kwargs.update(overrides)
    return TorchPolicyBehaviorModel(path, **kwargs)


def test_zero_logit_checkpoint_becomes_exact_q32_complete_distribution(tmp_path):
    checkpoint = tmp_path / "policy.pt"
    _write_zero_checkpoint(checkpoint)
    model = _adapter(checkpoint)
    information = _information()

    first = model.action_distribution(information)
    replay = model.action_distribution(information)

    assert first == replay
    assert first.information_digest == information.digest()
    assert first.source == "model"
    assert first.used_fallback is False
    assert set(first.probabilities) == set(information.legal_action_ids)
    assert sum(first.probabilities.values(), Fraction(0)) == 1
    assert all(
        probability.denominator <= DEFAULT_QUANTIZATION_DENOMINATOR
        for probability in first.probabilities.values()
    )
    q32_units = [
        probability * DEFAULT_QUANTIZATION_DENOMINATOR
        for probability in first.probabilities.values()
    ]
    assert all(units.denominator == 1 for units in q32_units)
    assert max(q32_units) - min(q32_units) <= 1
    assert model.model_manifest["input_dim"] == 520
    assert model.model_manifest["output_dim"] == 250
    assert model.model_manifest["promotion_eligible"] is False
    assert len(model.model_sha256) == 64


def test_adapter_scope_and_legal_action_contract_fail_closed(tmp_path):
    checkpoint = tmp_path / "policy.pt"
    _write_zero_checkpoint(checkpoint)
    model = _adapter(checkpoint)

    with pytest.raises(ValueError, match="does not support actor"):
        model.action_distribution(_information(actor="bb"))
    with pytest.raises(ValueError, match="does not support turn"):
        model.action_distribution(_information(turn=2))

    valid = _information()
    missing = BehaviorInfoSet(
        actor=valid.actor,
        turn=valid.turn,
        board_bb=valid.board_bb,
        board_btn=valid.board_btn,
        public_action_history=valid.public_action_history,
        own_recall_before=valid.own_recall_before,
        current_draw=valid.current_draw,
        legal_action_ids=valid.legal_action_ids[:-1],
    )
    with pytest.raises(ValueError, match="legal actions do not match"):
        model.action_distribution(missing)


def test_expected_checkpoint_hash_and_exact_temperature_are_enforced(tmp_path):
    checkpoint = tmp_path / "policy.pt"
    _write_zero_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="SHA256"):
        _adapter(checkpoint, expected_checkpoint_sha256="0" * 64)
    with pytest.raises(TypeError, match="exact"):
        _adapter(checkpoint, temperature=0.5)
    with pytest.raises(ValueError, match="positive"):
        _adapter(checkpoint, temperature=0)
    with pytest.raises(ValueError, match="legacy.*cannot be promotion eligible"):
        _adapter(
            checkpoint,
            training_scope_id="legacy_empty_opponent_board_btn_only",
            promotion_eligible=True,
        )


def test_turn_actor_dispatch_is_content_addressed_and_has_no_default_route(tmp_path):
    checkpoint = tmp_path / "policy.pt"
    _write_zero_checkpoint(checkpoint)
    child = _adapter(checkpoint)
    dispatch = TurnActorBehaviorDispatch({(1, "btn"): child})

    assert dispatch.action_distribution(_information()) == child.action_distribution(
        _information()
    )
    assert dispatch.model_manifest["routes"][0]["child_model_sha256"] == (
        child.model_sha256
    )
    assert dispatch.model_manifest["promotion_eligible"] is False
    assert len(dispatch.model_sha256) == 64
    with pytest.raises(KeyError, match="no frozen behavior model"):
        dispatch.action_distribution(_information(actor="bb"))


def test_real_legacy_t1_checkpoint_loads_with_explicit_non_promoted_scope():
    checkpoint = Path("ai/models/bottomup_t1/bc_policy_best.pt")
    if not checkpoint.is_file():
        pytest.skip("legacy T1 checkpoint is not present")
    model = TorchPolicyBehaviorModel(
        checkpoint,
        model_id="legacy_bottomup_t1_btn_behavior",
        supported_turns=(1,),
        supported_actors=("btn",),
        training_scope_id="legacy_empty_opponent_board_btn_only",
        promotion_eligible=False,
        expected_checkpoint_sha256=(
            "bb770ebc00ae100728786d416790cc1c32ada2316d4629ada0ad2f672d7f2ef8"
        ),
    )

    distribution = model.action_distribution(_information())

    assert model.model_manifest["input_dim"] == 520
    assert model.model_manifest["training_scope_id"] == (
        "legacy_empty_opponent_board_btn_only"
    )
    assert model.model_manifest["promotion_eligible"] is False
    assert sum(distribution.probabilities.values(), Fraction(0)) == 1
    assert distribution.source == "model"


def test_real_hu_t1_btn_policyvalue_checkpoint_is_content_addressed_prior():
    checkpoint = Path(
        "ai/data/t1_hu_btn_gcp_2m_model/t2_policyvalue_model.pt"
    )
    if not checkpoint.is_file():
        pytest.skip("HU T1 BTN policy-value checkpoint is not present")
    model = TorchPolicyValueBehaviorModel(
        checkpoint,
        model_id="hu_t1_btn_2m_policyvalue_prior",
        supported_turns=(1,),
        supported_actors=("btn",),
        training_scope_id="hu_t1_btn_visible_opponent_board_2m_v1",
        expected_checkpoint_sha256=(
            "064ab29967d4294f76f4ac844000b7fad97553eb341cfb54065f023bbfe7a32f"
        ),
    )

    first = model.action_distribution(_information(actor="btn", turn=1))
    replay = model.action_distribution(_information(actor="btn", turn=1))

    assert first == replay
    assert first.source == "model"
    assert first.used_fallback is False
    assert sum(first.probabilities.values(), Fraction(0)) == 1
    assert model.model_manifest["input_dim"] == 522
    assert model.model_manifest["output_dim"] == 27
    assert model.model_manifest["supported_actors"] == ["btn"]
    assert model.model_manifest["calibration_status"] == (
        "uncalibrated_policy_ranking_logits"
    )
    assert model.model_manifest["promotion_eligible"] is False
    assert len(model.model_sha256) == 64
