from __future__ import annotations

import hashlib
import pickle
import sys
import types
from pathlib import Path

import pytest

from ofc_regular.action_key import action_key
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m43_attempt10_distilled_model import (
    HU_M43_ATTEMPT10_DISTILLED_ARTIFACT_SCHEMA,
    HU_M43_ATTEMPT10_DISTILLED_MODEL_SCHEMA,
)
from ofc_regular.hu_m43_joint_model_loader import load_hu_m43_joint_action_model
from ofc_regular.hu_m4_t1_policy import HuM4T1SelectiveOverridePolicy
from ofc_regular.state import Board


class _Baseline:
    def __init__(self) -> None:
        self.calls: list[ActorObservation] = []

    def choose_action_observation(self, observation: ActorObservation, **_kwargs):
        self.calls.append(observation)
        actions = generate_turn_actions(observation.hero_board, observation.dealt_cards)
        return min(actions, key=lambda action: action_key(action).sort_key())


class _RawAttempt10Model:
    schema = HU_M43_ATTEMPT10_DISTILLED_MODEL_SCHEMA
    safety_enabled = True

    def predict_sample_with_baseline(self, *_args, **_kwargs):
        raise AssertionError("raw Attempt10 inference must not execute")

    def predict_safety_probability(self, *_args, **_kwargs):
        raise AssertionError("raw Attempt10 safety inference must not execute")


def _observation() -> ActorObservation:
    return ActorObservation(
        hero_board=Board.from_rows(
            top=("Ah",), middle=("Kd",), bottom=("2s", "3s", "9s")
        ),
        opponent_public_board=Board.from_rows(
            top=("Qh",),
            middle=("Jd", "Td"),
            bottom=("4s", "5s", "6s", "7s"),
        ),
        dealt_cards=("8c", "9c", "Tc"),
        hero_private_discards=(),
        seat="second",
        street="T1",
        to_act_order="second",
    )


def _raw_artifact(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "attempt10.pkl"
    encoded = pickle.dumps(
        {
            "artifact_schema": HU_M43_ATTEMPT10_DISTILLED_ARTIFACT_SCHEMA,
            "model_schema": HU_M43_ATTEMPT10_DISTILLED_MODEL_SCHEMA,
            "model": object(),
        },
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    path.write_bytes(encoded)
    return path, hashlib.sha256(encoded).hexdigest()


def test_attempt10_loader_rejects_raw_and_partial_runtime_bindings(
    tmp_path: Path,
) -> None:
    path, digest = _raw_artifact(tmp_path)
    with pytest.raises(ValueError, match="raw distilled artifact is not runtime-loadable"):
        load_hu_m43_joint_action_model(path)
    with pytest.raises(ValueError, match="requires expected SHA"):
        load_hu_m43_joint_action_model(path, expected_sha256=digest)
    with pytest.raises(ValueError, match="no threshold-lock"):
        load_hu_m43_joint_action_model(path, threshold_lock_path="unused.json")


def test_attempt10_loader_forwards_only_complete_binding_to_acceptance_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, digest = _raw_artifact(tmp_path)
    module_name = "ofc_regular.validate_hu_m43_attempt10_acceptance"
    fake_module = types.ModuleType(module_name)
    sentinel = object()
    captured: dict[str, object] = {}

    def _load_bound(model_path, **kwargs):
        captured["model_path"] = model_path
        captured.update(kwargs)
        return sentinel

    fake_module.load_bound_attempt10_distilled_model = _load_bound
    monkeypatch.setitem(sys.modules, module_name, fake_module)
    result = load_hu_m43_joint_action_model(
        path,
        expected_sha256=digest,
        freeze_manifest="freeze.json",
        training_manifest_path="training.json",
        runtime_source_manifest_path="source.json",
        runtime_source_root="source-tree",
        runtime_dependency_root="dependencies",
    )
    assert result is sentinel
    assert captured == {
        "model_path": path,
        "expected_sha256": digest,
        "runtime_freeze": "freeze.json",
        "training_manifest_path": "training.json",
        "runtime_source_manifest_path": "source.json",
        "runtime_source_root": "source-tree",
        "runtime_dependency_root": "dependencies",
    }


def test_policy_direct_raw_attempt10_model_is_nonfire_and_counterfactual_baseline(
) -> None:
    observation = _observation()
    baseline = _Baseline()
    raw = _RawAttempt10Model()
    decisions: list[dict] = []
    policy = HuM4T1SelectiveOverridePolicy(
        baseline,
        action_value_model=raw,
        safety_model=raw,
        safety_probability_threshold=0.5,
        runtime_binding_verified=True,
        decision_log=decisions,
    )

    selected = policy.choose_action_observation(
        observation, hand_id=11, game_id="attempt10", decision_seed=17
    )
    expected = min(
        generate_turn_actions(observation.hero_board, observation.dealt_cards),
        key=lambda action: action_key(action).sort_key(),
    )
    assert action_key(selected) == action_key(expected)
    assert baseline.calls == [observation]
    assert len(decisions) == 1
    assert decisions[0]["override_fired"] is False
    assert decisions[0]["nonfire_reason"] == "runtime_binding_unverified"
    assert decisions[0]["baseline_action_key"] == action_key(expected).to_token()
    assert decisions[0]["final_action_key"] == action_key(expected).to_token()
