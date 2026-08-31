from __future__ import annotations

import copy
import inspect
from dataclasses import replace
from fractions import Fraction
from types import MappingProxyType

import pytest

import ai.tutor.calibrated_behavior_bootstrap as bootstrap_module
from ai.tutor.behavior_temperature_calibration import (
    build_behavior_temperature_calibration,
)
from ai.tutor.calibrated_behavior_bootstrap import (
    BOOTSTRAP_MODEL_TYPE,
    BOOTSTRAP_ROUTE_SCOPE,
    FixedPointBootstrapCalibratedHuT1T2BehaviorDispatch,
    build_fixed_point_bootstrap_calibrated_hu_t1_t2_behavior_dispatch,
)
from ai.tutor.calibrated_behavior_runtime import EXPECTED_ROUTES, RUNTIME_SCHEMA
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


@pytest.fixture(scope="module")
def passed_evidence():
    records, rows, challenge_records, challenge_rows = _dataset()
    artifact = build_behavior_temperature_calibration(
        records,
        rows,
        challenge_records=challenge_records,
        challenge_evaluation_rows=challenge_rows,
        gate_config=_small_gate(),
    )
    assert artifact["promotion_eligible"] is True
    return records, rows, challenge_records, challenge_rows, artifact


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
    artifact,
    *,
    evaluator_mutation=None,
    child_manifest_mutation=None,
    fallback=False,
):
    bindings = artifact["role_model_bindings"]
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

    constructor_calls = []
    evaluator_loader_calls = []

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
            constructor_calls.append((role, str(temperature)))
            manifest = {
                "schema": RUNTIME_SCHEMA,
                "model_id": model_id,
                "model_type": "torch_policyvalue_boltzmann_q32",
                "position_contract_version": "bb_first_v1",
                "rules_version": bootstrap_module.RULES_VERSION,
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

    def load_evaluators(_root, *, quantization_denominator):
        evaluator_loader_calls.append(quantization_denominator)
        return evaluators

    monkeypatch.setattr(bootstrap_module, "KNOWN_HU_POLICY_VALUE_ASSETS", assets)
    monkeypatch.setattr(
        bootstrap_module,
        "APPROVED_HU_POLICY_VALUE_CHECKPOINTS",
        {
            route: asset["checkpoint_sha256"]
            for route, asset in assets.items()
        },
    )
    monkeypatch.setattr(
        bootstrap_module,
        "build_known_hu_policy_value_logit_evaluators",
        load_evaluators,
    )
    monkeypatch.setattr(
        bootstrap_module, "TorchPolicyValueBehaviorModel", _FakeRuntimeChild
    )
    return constructor_calls, evaluator_loader_calls, evaluators


def _build(monkeypatch, evidence, *, fallback=False):
    records, rows, challenge_records, challenge_rows, artifact = evidence
    calls, loader_calls, evaluators = _install_fake_loaders(
        monkeypatch, artifact, fallback=fallback
    )
    model = build_fixed_point_bootstrap_calibrated_hu_t1_t2_behavior_dispatch(
        records,
        rows,
        artifact,
        challenge_records=challenge_records,
        challenge_evaluation_rows=challenge_rows,
        workspace_root=".",
    )
    return model, calls, loader_calls, evaluators


def _expected_temperatures(artifact):
    result = {}
    for role, row in artifact["temperatures"].items():
        payload = row["final_temperature"]
        temperature = Fraction(payload["numerator"], payload["denominator"])
        result[role] = f"{temperature.numerator}/{temperature.denominator}"
    return result


def test_bootstrap_runtime_rebuilds_raw_calibration_and_is_explicitly_nonpromotable(
    monkeypatch, passed_evidence
):
    records, rows, challenge_records, challenge_rows, artifact = passed_evidence
    model, calls, loader_calls, _evaluators = _build(
        monkeypatch, passed_evidence
    )

    assert isinstance(
        model, FixedPointBootstrapCalibratedHuT1T2BehaviorDispatch
    )
    manifest = dict(model.model_manifest)
    assert manifest["schema"] == RUNTIME_SCHEMA
    assert manifest["model_type"] == BOOTSTRAP_MODEL_TYPE
    assert manifest["position_contract_version"] == "bb_first_v1"
    assert manifest["bb_first"] is True
    assert manifest["calibrated"] is True
    assert manifest["calibration_verified_from_raw"] is True
    assert manifest["source_calibration_promotion_eligible"] is True
    assert manifest["source_calibration_all_required_gates_passed"] is True
    assert manifest["source_calibration_gate_failure_count"] == 0
    assert manifest["source_promotion_required_at_construction"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["fixed_point_bootstrap_only"] is True
    assert manifest["strategic_strength"] is False
    assert manifest["strategic_strength_evaluated"] is False
    assert manifest["strategic_strength_claimed"] is False
    assert manifest["route_scope"] == BOOTSTRAP_ROUTE_SCOPE == "t1_t2_only"
    assert manifest["no_fallback"] is True
    assert manifest["t3_bb_route_included"] is False
    assert manifest["t3_bb_likelihood_binding_included"] is False
    assert manifest["calibrated_behavior_bridge_included"] is False
    assert "calibrated_bridge_sha256" not in manifest
    assert "t3_bb_likelihood_binding_sha256" not in manifest
    assert manifest["calibration_artifact_sha256"] == artifact["artifact_sha256"]
    assert manifest["raw_record_set_sha256"] == artifact["raw_record_set_sha256"]
    assert (
        manifest["challenge_raw_record_set_sha256"]
        == artifact["joker_challenge"]["raw_record_set_sha256"]
    )
    assert loader_calls == [bootstrap_module.DEFAULT_QUANTIZATION_DENOMINATOR]

    expected_temperatures = _expected_temperatures(artifact)
    assert dict(calls) == expected_temperatures
    assert [row["role"] for row in manifest["routes"]] == [
        "t1_bb",
        "t1_btn",
        "t2_bb",
        "t2_btn",
    ]
    for route in manifest["routes"]:
        role = route["role"]
        binding = artifact["role_model_bindings"][role]
        assert route["temperature"] == expected_temperatures[role]
        assert route["checkpoint_sha256"] == binding["checkpoint_sha256"]
        assert route["calibration_source_model_sha256"] == binding["model_sha256"]
        assert route["row_extractor_sha256"] == binding["row_extractor_sha256"]
        assert route["adapter_source_sha256"] == binding["adapter_source_sha256"]

    assert model.model_sha256 == canonical_sha256(dict(model.model_manifest))
    identity = _validated_behavior_identity(model)
    assert identity[0] == model.model_id
    assert identity[1] == model.model_sha256
    distribution = model.action_distribution(_t1_bb_information())
    assert distribution.source == "model"
    assert distribution.used_fallback is False
    assert sum(distribution.probabilities.values(), Fraction(0, 1)) == 1

    # Order is outside the evidence identity.  Rebuilding all raw rows gives
    # the same content-addressed bootstrap runtime without a bridge/T3 input.
    replay = build_fixed_point_bootstrap_calibrated_hu_t1_t2_behavior_dispatch(
        list(reversed(records)),
        list(reversed(rows)),
        artifact,
        challenge_records=list(reversed(challenge_records)),
        challenge_evaluation_rows=list(reversed(challenge_rows)),
        workspace_root=".",
    )
    assert replay.model_sha256 == model.model_sha256
    assert dict(replay.model_manifest) == dict(model.model_manifest)

    signature = inspect.signature(
        build_fixed_point_bootstrap_calibrated_hu_t1_t2_behavior_dispatch
    )
    assert "bridge" not in signature.parameters
    assert "training_root_ids" not in signature.parameters
    assert "t3_bb_likelihood_binding" not in signature.parameters


def test_manifest_is_a_detached_snapshot(monkeypatch, passed_evidence):
    model, *_ = _build(monkeypatch, passed_evidence)
    original_sha256 = model.model_sha256
    detached = model.model_manifest
    detached["routes"][0]["temperature"] = "19/7"
    assert model.model_manifest["routes"][0]["temperature"] != "19/7"
    assert model.model_sha256 == original_sha256


def test_t3_is_not_routed_and_any_child_fallback_fails_closed(
    monkeypatch, passed_evidence
):
    model, *_ = _build(monkeypatch, passed_evidence, fallback=True)
    with pytest.raises(RuntimeError, match="forbids fallback"):
        model.action_distribution(_t1_bb_information())

    t3_information = replace(_t1_bb_information(), turn=3)
    with pytest.raises(KeyError, match="route_scope=t1_t2_only"):
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
def test_all_four_loaded_evaluator_bindings_are_exact(
    monkeypatch, passed_evidence, field, replacement
):
    records, rows, challenge_records, challenge_rows, artifact = passed_evidence

    def mutate(evaluators):
        setattr(evaluators[(1, "bb")], field, replacement)

    _install_fake_loaders(monkeypatch, artifact, evaluator_mutation=mutate)
    with pytest.raises(ValueError, match=f"loaded t1_bb {field} does not match"):
        build_fixed_point_bootstrap_calibrated_hu_t1_t2_behavior_dispatch(
            records,
            rows,
            artifact,
            challenge_records=challenge_records,
            challenge_evaluation_rows=challenge_rows,
            workspace_root=".",
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        (
            "adapter_source_sha256",
            "5" * 64,
            "runtime child t2_btn changed the calibrated adapter",
        ),
        (
            "temperature",
            "7/3",
            "runtime child t2_btn did not apply the calibrated temperature",
        ),
    ],
)
def test_runtime_child_adapter_and_temperature_are_rechecked(
    monkeypatch, passed_evidence, field, replacement, message
):
    records, rows, challenge_records, challenge_rows, artifact = passed_evidence

    def mutate(role, manifest):
        if role == "t2_btn":
            manifest[field] = replacement

    _install_fake_loaders(monkeypatch, artifact, child_manifest_mutation=mutate)
    with pytest.raises(ValueError, match=message):
        build_fixed_point_bootstrap_calibrated_hu_t1_t2_behavior_dispatch(
            records,
            rows,
            artifact,
            challenge_records=challenge_records,
            challenge_evaluation_rows=challenge_rows,
            workspace_root=".",
        )


def test_tampered_raw_evidence_fails_before_checkpoint_loading(
    monkeypatch, passed_evidence
):
    records, rows, challenge_records, challenge_rows, artifact = passed_evidence
    _calls, loader_calls, _evaluators = _install_fake_loaders(
        monkeypatch, artifact
    )
    tampered_rows = copy.deepcopy(rows)
    tampered_rows[0]["logits_f64_hex"][0] = float(99).hex()
    with pytest.raises(ValueError, match="evaluation row SHA-256 mismatch"):
        build_fixed_point_bootstrap_calibrated_hu_t1_t2_behavior_dispatch(
            records,
            tampered_rows,
            artifact,
            challenge_records=challenge_records,
            challenge_evaluation_rows=challenge_rows,
            workspace_root=".",
        )
    assert loader_calls == []

    tampered_artifact = copy.deepcopy(artifact)
    tampered_artifact["temperatures"]["t1_bb"]["final_temperature"][
        "numerator"
    ] += 1
    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        build_fixed_point_bootstrap_calibrated_hu_t1_t2_behavior_dispatch(
            records,
            rows,
            tampered_artifact,
            challenge_records=challenge_records,
            challenge_evaluation_rows=challenge_rows,
            workspace_root=".",
        )
    assert loader_calls == []


def test_nonpromoted_source_is_diagnostic_only_and_production_mode_fails_closed(
    monkeypatch,
):
    records, rows, challenge_records, challenge_rows = _dataset(
        include_joker2=False
    )
    artifact = build_behavior_temperature_calibration(
        records,
        rows,
        challenge_records=challenge_records,
        challenge_evaluation_rows=challenge_rows,
        gate_config=_small_gate(),
    )
    assert artifact["promotion_eligible"] is False
    _calls, loader_calls, _evaluators = _install_fake_loaders(
        monkeypatch, artifact
    )
    model = build_fixed_point_bootstrap_calibrated_hu_t1_t2_behavior_dispatch(
        records,
        rows,
        artifact,
        challenge_records=challenge_records,
        challenge_evaluation_rows=challenge_rows,
        workspace_root=".",
    )
    manifest = model.model_manifest
    assert manifest["promotion_eligible"] is False
    assert manifest["fixed_point_bootstrap_only"] is True
    assert manifest["source_calibration_promotion_eligible"] is False
    assert manifest["source_calibration_all_required_gates_passed"] is False
    assert manifest["source_calibration_gate_failure_count"] > 0
    assert manifest["source_promotion_required_at_construction"] is False
    assert loader_calls == [bootstrap_module.DEFAULT_QUANTIZATION_DENOMINATOR]

    before = len(loader_calls)
    with pytest.raises(ValueError, match="production bootstrap requires"):
        build_fixed_point_bootstrap_calibrated_hu_t1_t2_behavior_dispatch(
            records,
            rows,
            artifact,
            challenge_records=challenge_records,
            challenge_evaluation_rows=challenge_rows,
            workspace_root=".",
            require_promoted_source=True,
        )
    assert len(loader_calls) == before


def test_production_mode_accepts_only_a_promoted_source(
    monkeypatch, passed_evidence
):
    records, rows, challenge_records, challenge_rows, artifact = passed_evidence
    _install_fake_loaders(monkeypatch, artifact)
    model = build_fixed_point_bootstrap_calibrated_hu_t1_t2_behavior_dispatch(
        records,
        rows,
        artifact,
        challenge_records=challenge_records,
        challenge_evaluation_rows=challenge_rows,
        workspace_root=".",
        require_promoted_source=True,
    )
    manifest = model.model_manifest
    assert manifest["promotion_eligible"] is False
    assert manifest["source_calibration_promotion_eligible"] is True
    assert manifest["source_calibration_all_required_gates_passed"] is True
    assert manifest["source_promotion_required_at_construction"] is True


def test_direct_construction_and_non_q32_runtime_are_rejected():
    with pytest.raises(TypeError, match="use build_fixed_point_bootstrap"):
        FixedPointBootstrapCalibratedHuT1T2BehaviorDispatch(
            {}, {}, _construction_token=object()
        )

    with pytest.raises(ValueError, match="exact Q32"):
        build_fixed_point_bootstrap_calibrated_hu_t1_t2_behavior_dispatch(
            (), (), {}, quantization_denominator=1024
        )

    with pytest.raises(TypeError, match="require_promoted_source must be boolean"):
        build_fixed_point_bootstrap_calibrated_hu_t1_t2_behavior_dispatch(
            (), (), {}, require_promoted_source=1
        )
