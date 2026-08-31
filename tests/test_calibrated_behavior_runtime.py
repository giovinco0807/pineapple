from __future__ import annotations

import copy
from dataclasses import replace
from fractions import Fraction
from types import MappingProxyType

import pytest

import ai.tutor.calibrated_behavior_runtime as runtime_module
from ai.tutor.behavior_temperature_calibration import (
    build_behavior_temperature_calibration,
)
from ai.tutor.calibrated_behavior_runtime import (
    EXPECTED_ROUTES,
    RUNTIME_MODEL_TYPE,
    RUNTIME_ROUTE_SCOPE,
    RUNTIME_SCHEMA,
    VerifiedCalibratedHuT1T2BehaviorDispatch,
    build_verified_calibrated_hu_t1_t2_behavior_dispatch,
)
from ai.tutor.promotion_gate_m3_full_card_strength import canonical_sha256
from ai.tutor.t3_hu_full_card_range import (
    BehaviorDistribution,
    _validated_behavior_identity,
)
from test_behavior_temperature_calibration import (
    _dataset,
    _small_gate,
    _t1_bb_information,
)
from test_calibrated_behavior_manifest_bridge import (
    TRAINING_ROOTS,
    _bridge,
    _t3_binding,
)


class _FakeIdentityEvaluator:
    def __init__(self, binding):
        self.checkpoint_sha256 = binding["checkpoint_sha256"]
        self.model_sha256 = binding["model_sha256"]
        self.row_extractor_sha256 = binding["row_extractor_sha256"]
        self.adapter_source_sha256 = binding["adapter_source_sha256"]

    @property
    def evaluator_manifest(self):
        return {
            "schema": "ofc_behavior_logit_evaluator/v1",
            "temperature": "1/1",
            "checkpoint_sha256": self.checkpoint_sha256,
            "model_sha256": self.model_sha256,
            "row_extractor_sha256": self.row_extractor_sha256,
            "adapter_source_sha256": self.adapter_source_sha256,
        }


def _install_fake_loaders(
    monkeypatch,
    bridge,
    *,
    evaluator_mutation=None,
    child_manifest_mutation=None,
    fallback=False,
):
    bindings = bridge["calibrated_behavior_manifest"]["role_model_bindings"]
    evaluators = {}
    assets = {}
    for turn, actor in EXPECTED_ROUTES:
        role = f"t{turn}_{actor}"
        evaluators[(turn, actor)] = _FakeIdentityEvaluator(bindings[role])
        assets[(turn, actor)] = {
            "relative_path": f"synthetic/{role}.pt",
            "checkpoint_sha256": bindings[role]["checkpoint_sha256"],
        }
    if evaluator_mutation is not None:
        evaluator_mutation(evaluators)

    calls = []

    class _FakeRuntimeChild:
        def __init__(
            self,
            checkpoint_path,
            *,
            model_id,
            supported_turns,
            supported_actors,
            training_scope_id,
            temperature,
            quantization_denominator,
            expected_checkpoint_sha256,
        ):
            del checkpoint_path, training_scope_id
            self._model_id = model_id
            turn = tuple(supported_turns)[0]
            actor = tuple(supported_actors)[0]
            role = f"t{turn}_{actor}"
            calls.append((role, temperature))
            manifest = {
                "schema": RUNTIME_SCHEMA,
                "model_id": model_id,
                "model_type": "torch_policyvalue_boltzmann_q32",
                "position_contract_version": "bb_first_v1",
                "rules_version": runtime_module.RULES_VERSION,
                "checkpoint_sha256": expected_checkpoint_sha256,
                "adapter_source_sha256": bindings[role]["adapter_source_sha256"],
                "temperature": str(temperature),
                "quantization_denominator": quantization_denominator,
                "supported_turns": [turn],
                "supported_actors": [actor],
                "promotion_eligible": False,
            }
            if child_manifest_mutation is not None:
                child_manifest_mutation(role, manifest)
            self._manifest = manifest
            self._sha256 = canonical_sha256(manifest)

        @property
        def model_id(self):
            return self._model_id

        @property
        def model_manifest(self):
            return MappingProxyType(copy.deepcopy(self._manifest))

        @property
        def model_sha256(self):
            return self._sha256

        def action_distribution(self, information):
            probability = Fraction(1, information.legal_action_count)
            return BehaviorDistribution(
                information_digest=information.digest(),
                probabilities=MappingProxyType(
                    {
                        action_id: probability
                        for action_id in information.legal_action_ids
                    }
                ),
                source="uniform_fallback" if fallback else "model",
                used_fallback=fallback,
            )

    monkeypatch.setattr(runtime_module, "KNOWN_HU_POLICY_VALUE_ASSETS", assets)
    monkeypatch.setattr(
        runtime_module,
        "APPROVED_HU_POLICY_VALUE_CHECKPOINTS",
        {
            route: asset["checkpoint_sha256"]
            for route, asset in assets.items()
        },
    )
    monkeypatch.setattr(
        runtime_module,
        "build_known_hu_policy_value_logit_evaluators",
        lambda _root, *, quantization_denominator: evaluators,
    )
    monkeypatch.setattr(
        runtime_module, "TorchPolicyValueBehaviorModel", _FakeRuntimeChild
    )
    return calls, evaluators


def _build_runtime(monkeypatch, *, fallback=False):
    records, rows, challenge_records, challenge_rows, artifact, bridge = _bridge()
    calls, evaluators = _install_fake_loaders(
        monkeypatch, bridge, fallback=fallback
    )
    model = build_verified_calibrated_hu_t1_t2_behavior_dispatch(
        records,
        rows,
        artifact,
        bridge,
        training_root_ids=TRAINING_ROOTS,
        t3_bb_likelihood_binding=_t3_binding(),
        challenge_records=challenge_records,
        challenge_evaluation_rows=challenge_rows,
        workspace_root=".",
    )
    return (
        records,
        rows,
        challenge_records,
        challenge_rows,
        artifact,
        bridge,
        model,
        calls,
        evaluators,
    )


def test_promoted_runtime_is_content_addressed_direct_temperature_dispatch(monkeypatch):
    (
        records,
        rows,
        challenge_records,
        challenge_rows,
        artifact,
        bridge,
        model,
        calls,
        _evaluators,
    ) = _build_runtime(monkeypatch)

    assert isinstance(model, VerifiedCalibratedHuT1T2BehaviorDispatch)
    manifest = dict(model.model_manifest)
    assert manifest["schema"] == RUNTIME_SCHEMA
    assert manifest["model_type"] == RUNTIME_MODEL_TYPE
    assert manifest["position_contract_version"] == "bb_first_v1"
    assert manifest["bb_first"] is True
    assert manifest["promotion_eligible"] is True
    assert manifest["promotion_scope"] == RUNTIME_ROUTE_SCOPE
    assert manifest["standalone_full_m3_behavior_complete"] is False
    assert manifest["t3_bb_route_included"] is False
    assert manifest["strategic_strength_evaluated"] is False
    assert manifest["strategic_strength_claimed"] is False
    assert manifest["no_fallback"] is True
    assert manifest["calibrated_bridge_sha256"] == bridge["bridge_sha256"]
    assert (
        manifest["calibrated_behavior_manifest_sha256"]
        == bridge["calibrated_behavior_sha256"]
    )
    assert manifest["calibration_artifact_sha256"] == artifact["artifact_sha256"]
    assert model.model_sha256 == canonical_sha256(dict(model.model_manifest))
    assert [row["role"] for row in manifest["routes"]] == [
        "t1_bb",
        "t1_btn",
        "t2_bb",
        "t2_btn",
    ]
    expected_temperatures = bridge["calibrated_behavior_manifest"][
        "role_temperatures"
    ]
    assert dict(calls) == expected_temperatures
    assert all(
        row["temperature"] == expected_temperatures[row["role"]]
        for row in manifest["routes"]
    )
    for row in manifest["routes"]:
        binding = bridge["calibrated_behavior_manifest"]["role_model_bindings"][
            row["role"]
        ]
        assert row["calibration_source_model_sha256"] == binding["model_sha256"]
        assert row["checkpoint_sha256"] == binding["checkpoint_sha256"]
        assert row["adapter_source_sha256"] == binding["adapter_source_sha256"]
        assert row["row_extractor_sha256"] == binding["row_extractor_sha256"]

    identity = _validated_behavior_identity(model)
    assert identity[0] == model.model_id
    assert identity[1] == model.model_sha256
    distribution = model.action_distribution(_t1_bb_information())
    assert distribution.used_fallback is False
    assert distribution.source == "model"
    assert sum(distribution.probabilities.values(), Fraction(0, 1)) == 1

    # Rebuild from reordered raw rows: bridge and runtime identities remain exact.
    replay = build_verified_calibrated_hu_t1_t2_behavior_dispatch(
        list(reversed(records)),
        list(reversed(rows)),
        artifact,
        bridge,
        training_root_ids=TRAINING_ROOTS,
        t3_bb_likelihood_binding=_t3_binding(),
        challenge_records=list(reversed(challenge_records)),
        challenge_evaluation_rows=list(reversed(challenge_rows)),
        workspace_root=".",
    )
    assert replay.model_sha256 == model.model_sha256
    assert dict(replay.model_manifest) == dict(model.model_manifest)


def test_t3_is_explicitly_not_a_route_and_fallback_is_forbidden(monkeypatch):
    *_, model, _calls, _evaluators = _build_runtime(monkeypatch, fallback=True)
    with pytest.raises(RuntimeError, match="forbids fallback"):
        model.action_distribution(_t1_bb_information())

    t3_information = replace(_t1_bb_information(), turn=3)
    with pytest.raises(KeyError, match="separate endogenous fixed-point"):
        model.action_distribution(t3_information)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("checkpoint_sha256", "1" * 64),
        ("model_sha256", "2" * 64),
        ("row_extractor_sha256", "3" * 64),
        ("adapter_source_sha256", "4" * 64),
    ],
)
def test_loaded_calibration_identity_mismatch_fails_closed(
    monkeypatch, field, replacement
):
    records, rows, challenge_records, challenge_rows, artifact, bridge = _bridge()

    def mutate(evaluators):
        setattr(evaluators[(1, "bb")], field, replacement)

    _install_fake_loaders(monkeypatch, bridge, evaluator_mutation=mutate)
    with pytest.raises(ValueError, match=f"loaded t1_bb {field} does not match"):
        build_verified_calibrated_hu_t1_t2_behavior_dispatch(
            records,
            rows,
            artifact,
            bridge,
            training_root_ids=TRAINING_ROOTS,
            t3_bb_likelihood_binding=_t3_binding(),
            challenge_records=challenge_records,
            challenge_evaluation_rows=challenge_rows,
            workspace_root=".",
        )


def test_runtime_child_adapter_or_temperature_mismatch_fails_closed(monkeypatch):
    records, rows, challenge_records, challenge_rows, artifact, bridge = _bridge()

    def mutate(role, manifest):
        if role == "t2_btn":
            manifest["adapter_source_sha256"] = "5" * 64

    _install_fake_loaders(monkeypatch, bridge, child_manifest_mutation=mutate)
    with pytest.raises(ValueError, match="runtime child t2_btn changed the calibrated adapter"):
        build_verified_calibrated_hu_t1_t2_behavior_dispatch(
            records,
            rows,
            artifact,
            bridge,
            training_root_ids=TRAINING_ROOTS,
            t3_bb_likelihood_binding=_t3_binding(),
            challenge_records=challenge_records,
            challenge_evaluation_rows=challenge_rows,
            workspace_root=".",
        )


def test_tampered_bridge_and_unpromoted_calibration_fail_before_loading(monkeypatch):
    records, rows, challenge_records, challenge_rows, artifact, bridge = _bridge()
    _install_fake_loaders(monkeypatch, bridge)

    tampered = copy.deepcopy(bridge)
    tampered["calibrated_behavior_manifest"]["role_temperatures"]["t1_bb"] = "9/7"
    with pytest.raises(ValueError, match="bridge SHA-256 mismatch"):
        build_verified_calibrated_hu_t1_t2_behavior_dispatch(
            records,
            rows,
            artifact,
            tampered,
            training_root_ids=TRAINING_ROOTS,
            t3_bb_likelihood_binding=_t3_binding(),
            challenge_records=challenge_records,
            challenge_evaluation_rows=challenge_rows,
            workspace_root=".",
        )

    bad_records, bad_rows, bad_challenge, bad_challenge_rows = _dataset(
        include_joker2=False
    )
    unpromoted = build_behavior_temperature_calibration(
        bad_records,
        bad_rows,
        challenge_records=bad_challenge,
        challenge_evaluation_rows=bad_challenge_rows,
        gate_config=_small_gate(),
    )
    assert unpromoted["promotion_eligible"] is False
    with pytest.raises(ValueError, match="not promotion eligible"):
        build_verified_calibrated_hu_t1_t2_behavior_dispatch(
            bad_records,
            bad_rows,
            unpromoted,
            bridge,
            training_root_ids=TRAINING_ROOTS,
            t3_bb_likelihood_binding=_t3_binding(),
            challenge_records=bad_challenge,
            challenge_evaluation_rows=bad_challenge_rows,
            workspace_root=".",
        )


def test_direct_construction_is_rejected():
    with pytest.raises(TypeError, match="use build_verified"):
        VerifiedCalibratedHuT1T2BehaviorDispatch({}, {}, _construction_token=object())


def test_non_q32_runtime_is_rejected_before_evidence_or_checkpoint_io():
    with pytest.raises(ValueError, match="exact Q32"):
        build_verified_calibrated_hu_t1_t2_behavior_dispatch(
            (),
            (),
            {},
            {},
            training_root_ids=(),
            t3_bb_likelihood_binding={},
            quantization_denominator=1024,
        )
