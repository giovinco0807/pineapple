from __future__ import annotations

import hashlib
import pickle
import sys
import types
from pathlib import Path

import pytest

from ofc_regular.action_key import action_key
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.evaluate_hu_m4_population import requires_bound_population_runtime
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m43_attempt13_distilled_model import (
    BoundHuM43Attempt13DistilledModel,
    HU_M43_ATTEMPT13_DISTILLED_ARTIFACT_SCHEMA,
    HU_M43_ATTEMPT13_DISTILLED_MODEL_SCHEMA,
    is_bound_attempt13_distilled_model,
)
from ofc_regular.hu_m43_attempt13_distilled_runtime import (
    ATTEMPT13_BOUND_EXECUTION_MODULES,
    ATTEMPT13_RUNTIME_ACTIVATED,
    ATTEMPT13_SHARED_RUNTIME_DISPATCH_ENABLED,
    canonical_json_bytes,
    load_and_validate_distilled_runtime_manifest,
)
from ofc_regular.hu_m43_joint_model_loader import load_hu_m43_joint_action_model
from ofc_regular.hu_m4_t1_policy import HuM4T1SelectiveOverridePolicy
from ofc_regular.state import Board


class _Baseline:
    def __init__(self) -> None:
        self.calls: list[ActorObservation] = []

    def choose_action_observation(
        self, observation: ActorObservation, **_kwargs: object
    ) -> object:
        self.calls.append(observation)
        actions = generate_turn_actions(observation.hero_board, observation.dealt_cards)
        return min(actions, key=lambda action: action_key(action).sort_key())


class _RawAttempt13Model:
    schema = HU_M43_ATTEMPT13_DISTILLED_MODEL_SCHEMA
    runtime_binding_verified = False
    safety_enabled = True

    def predict_sample_with_baseline(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("raw Attempt13 inference must not execute")

    def predict_safety_probability(self, *_args: object, **_kwargs: object) -> float:
        raise AssertionError("raw Attempt13 safety inference must not execute")


def test_attempt13_raw_or_schema_spoof_is_never_a_bound_runtime() -> None:
    assert not is_bound_attempt13_distilled_model(_RawAttempt13Model())
    assert not is_bound_attempt13_distilled_model(object())


def test_attempt13_bound_wrapper_is_loader_issued_only() -> None:
    with pytest.raises(TypeError, match="loader-issued"):
        BoundHuM43Attempt13DistilledModel(object(), object())  # type: ignore[arg-type]


def test_attempt13_runtime_is_opt_in_and_binds_only_attempt13_modules() -> None:
    assert ATTEMPT13_RUNTIME_ACTIVATED is False
    assert ATTEMPT13_SHARED_RUNTIME_DISPATCH_ENABLED is True
    assert "ofc_regular.hu_m43_attempt13_distilled_model" in (
        ATTEMPT13_BOUND_EXECUTION_MODULES
    )
    assert "ofc_regular.train_hu_m43_attempt13_distilled" in (
        ATTEMPT13_BOUND_EXECUTION_MODULES
    )
    assert "ofc_regular_promotion.attempt13_profile" in (
        ATTEMPT13_BOUND_EXECUTION_MODULES
    )
    assert "ofc_regular_promotion.attempt13_population" in (
        ATTEMPT13_BOUND_EXECUTION_MODULES
    )
    assert all("attempt12" not in name for name in ATTEMPT13_BOUND_EXECUTION_MODULES)


def test_attempt13_population_runtime_is_always_binding_required() -> None:
    assert requires_bound_population_runtime(_RawAttempt13Model())


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
    path = tmp_path / "attempt13.pkl"
    encoded = pickle.dumps(
        {
            "artifact_schema": HU_M43_ATTEMPT13_DISTILLED_ARTIFACT_SCHEMA,
            "model_schema": HU_M43_ATTEMPT13_DISTILLED_MODEL_SCHEMA,
            "model": object(),
        },
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    path.write_bytes(encoded)
    return path, hashlib.sha256(encoded).hexdigest()


def test_attempt13_loader_rejects_raw_and_partial_runtime_bindings(
    tmp_path: Path,
) -> None:
    path, digest = _raw_artifact(tmp_path)
    with pytest.raises(ValueError, match="raw distilled artifact is not runtime-loadable"):
        load_hu_m43_joint_action_model(path)
    with pytest.raises(ValueError, match="Attempt13 distilled runtime requires"):
        load_hu_m43_joint_action_model(path, expected_sha256=digest)
    with pytest.raises(ValueError, match="has no threshold-lock input"):
        load_hu_m43_joint_action_model(path, threshold_lock_path="unused.json")


def test_attempt13_loader_forwards_only_complete_binding_to_acceptance_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, digest = _raw_artifact(tmp_path)
    module_name = "ofc_regular.validate_hu_m43_attempt13_acceptance"
    fake_module = types.ModuleType(module_name)
    sentinel = object()
    captured: dict[str, object] = {}

    def _load_bound(model_path: Path, **kwargs: object) -> object:
        captured["model_path"] = model_path
        captured.update(kwargs)
        return sentinel

    fake_module.load_bound_attempt13_distilled_model = _load_bound  # type: ignore[attr-defined]
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


def test_policy_direct_raw_attempt13_model_is_nonfire_and_counterfactual_baseline(
) -> None:
    observation = _observation()
    baseline = _Baseline()
    raw = _RawAttempt13Model()
    decisions: list[dict[str, object]] = []
    policy = HuM4T1SelectiveOverridePolicy(
        baseline,
        action_value_model=raw,
        safety_model=raw,
        safety_probability_threshold=0.5,
        runtime_binding_verified=True,
        decision_log=decisions,
    )

    selected = policy.choose_action_observation(
        observation, hand_id=11, game_id="attempt13", decision_seed=17
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


def test_attempt13_manifest_parser_is_canonical_and_fail_closed(
    tmp_path: Path,
) -> None:
    payload = {"b": 2, "a": 1}
    assert canonical_json_bytes(payload) == b'{"a":1,"b":2}\n'
    with pytest.raises(ValueError, match="keys changed"):
        load_and_validate_distilled_runtime_manifest(payload)
    noncanonical = tmp_path / "manifest.json"
    noncanonical.write_bytes(b'{ "a": 1, "b": 2 }\n')
    with pytest.raises(ValueError, match="not canonical"):
        load_and_validate_distilled_runtime_manifest(noncanonical)
