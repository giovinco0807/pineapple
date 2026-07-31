"""Verified calibrated T1/T2 behavior likelihoods for the full-card range.

The calibration artifact is not trusted by itself.  The public builder first
rebuilds the promoted calibration bridge from its raw main and Joker-challenge
records, their direct pre-temperature logit rows, and the external T3-BB
fixed-point binding.  It then loads the four exact known HU PolicyValueNet
checkpoints and requires their calibration-time evaluator identities to match
the bridge role-for-role before constructing temperature-scaled runtime
models.

Temperature is passed to :class:`TorchPolicyValueBehaviorModel` and is applied
to the checkpoint logits before softmax and exact Q32 quantization.  No
probability-to-logit inversion and no fallback route exists here.

This dispatch intentionally contains only the exogenous T1/T2 likelihood
routes.  Its promotion eligibility is a component-level claim backed by a
verified *full* bridge (including the endogenous T3-BB fixed-point binding),
not a claim that this object alone can answer BTN-root T3 posterior queries or
that strategic strength has been established.
"""
from __future__ import annotations

import copy
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ai.engine.turn_order import POSITION_CONTRACT_VERSION
from ai.tutor.behavior_logit_evaluator_torch import (
    build_known_hu_policy_value_logit_evaluators,
)
from ai.tutor.calibrated_behavior_manifest_bridge import (
    BRIDGE_SCHEMA,
    verify_calibrated_behavior_bridge,
)
from ai.tutor.frozen_behavior_torch import (
    DEFAULT_QUANTIZATION_DENOMINATOR,
    KNOWN_HU_POLICY_VALUE_ASSETS,
    RULES_VERSION,
    TorchPolicyValueBehaviorModel,
)
from ai.tutor.promotion_gate_m3_full_card_strength import (
    BEHAVIOR_SCHEMA,
    CALIBRATION_BINDING_KEYS,
    CALIBRATION_ROLE_KEYS,
    canonical_sha256,
)
from ai.tutor.t3_hu_full_card_range import (
    BehaviorDistribution,
    BehaviorInfoSet,
    FrozenBehaviorModel,
)


RUNTIME_SCHEMA = "ofc_frozen_behavior_model/v1"
RUNTIME_MODEL_TYPE = "verified_calibrated_t1_t2_policyvalue_q32_dispatch_v1"
RUNTIME_ROUTE_SCOPE = "t1_t2_exogenous_likelihood_routes_only"
EXPECTED_ROUTES = ((1, "bb"), (1, "btn"), (2, "bb"), (2, "btn"))
_EXPECTED_ROUTE_SET = frozenset(EXPECTED_ROUTES)
APPROVED_HU_POLICY_VALUE_CHECKPOINTS = MappingProxyType(
    {
        (1, "bb"): "27dab713a658a5ede5637d2c96ae0d5330464b96738f562ae84ce7326c2562c6",
        (1, "btn"): "064ab29967d4294f76f4ac844000b7fad97553eb341cfb54065f023bbfe7a32f",
        (2, "bb"): "fc47e5a7a02375c8d3fac32d04b0849d8e7b100fe5650007c8c7f8e6c7b69460",
        (2, "btn"): "a4d5dfcff1811515a6b3db972b7f06e071f00666db49f3cefe8a0079b7e6c83b",
    }
)
_SHA256_LENGTH = 64
_CONSTRUCTION_TOKEN = object()


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _role(turn: int, actor: str) -> str:
    return f"t{turn}_{actor}"


def _canonical_temperature(value: Any, *, role: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"role_temperatures.{role} must be a rational string")
    if value.strip() != value:
        raise ValueError(f"role_temperatures.{role} must be canonical")
    try:
        temperature = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(
            f"role_temperatures.{role} must be a positive rational"
        ) from exc
    if temperature <= 0:
        raise ValueError(f"role_temperatures.{role} must be positive")
    canonical = f"{temperature.numerator}/{temperature.denominator}"
    if value != canonical:
        raise ValueError(f"role_temperatures.{role} must be canonical")
    return canonical


def _validated_identity_evaluator(
    evaluator: Any,
    *,
    role: str,
    expected_binding: Mapping[str, Any],
    expected_checkpoint_sha256: str,
) -> None:
    """Bind the loaded identity-temperature evaluator to one bridge role."""

    if set(expected_binding) != set(CALIBRATION_BINDING_KEYS):
        raise ValueError(f"role_model_bindings.{role} has unexpected fields")
    binding = {
        key: _require_sha256(
            expected_binding[key], label=f"role_model_bindings.{role}.{key}"
        )
        for key in CALIBRATION_BINDING_KEYS
    }
    if binding["checkpoint_sha256"] != expected_checkpoint_sha256:
        raise ValueError(
            f"{role} bridge checkpoint does not match the approved known asset"
        )

    direct = {
        "checkpoint_sha256": getattr(evaluator, "checkpoint_sha256", None),
        "model_sha256": getattr(evaluator, "model_sha256", None),
        "row_extractor_sha256": getattr(evaluator, "row_extractor_sha256", None),
        "adapter_source_sha256": getattr(evaluator, "adapter_source_sha256", None),
    }
    for field, expected in binding.items():
        actual = _require_sha256(direct[field], label=f"loaded {role} {field}")
        if actual != expected:
            raise ValueError(f"loaded {role} {field} does not match calibration bridge")

    evaluator_manifest = _require_mapping(
        getattr(evaluator, "evaluator_manifest", None),
        label=f"loaded {role} evaluator_manifest",
    )
    if evaluator_manifest.get("schema") != "ofc_behavior_logit_evaluator/v1":
        raise ValueError(f"loaded {role} evaluator has an unsupported schema")
    if evaluator_manifest.get("temperature") != "1/1":
        raise ValueError(
            f"loaded {role} evaluator is not the pre-temperature identity route"
        )
    for field, expected in binding.items():
        if evaluator_manifest.get(field) != expected:
            raise ValueError(
                f"loaded {role} evaluator manifest {field} does not match bridge"
            )


def _validated_runtime_child(
    child: FrozenBehaviorModel,
    *,
    turn: int,
    actor: str,
    role: str,
    binding: Mapping[str, Any],
    temperature: str,
    quantization_denominator: int,
) -> tuple[str, Mapping[str, Any]]:
    """Verify the calibrated child still loads the bound checkpoint/adapter."""

    child_sha256 = _require_sha256(
        child.model_sha256, label=f"runtime child {role} model_sha256"
    )
    manifest = _require_mapping(
        child.model_manifest, label=f"runtime child {role} manifest"
    )
    snapshot = copy.deepcopy(dict(manifest))
    if canonical_sha256(snapshot) != child_sha256:
        raise ValueError(f"runtime child {role} manifest/hash mismatch")
    if snapshot.get("schema") != RUNTIME_SCHEMA:
        raise ValueError(f"runtime child {role} has an unsupported schema")
    child_model_id = getattr(child, "model_id", None)
    if not isinstance(child_model_id, str) or not child_model_id:
        raise ValueError(f"runtime child {role} has an invalid model_id")
    if snapshot.get("model_id") != child_model_id:
        raise ValueError(f"runtime child {role} model_id/manifest mismatch")
    if snapshot.get("model_type") != "torch_policyvalue_boltzmann_q32":
        raise ValueError(f"runtime child {role} has the wrong model type")
    if snapshot.get("position_contract_version") != POSITION_CONTRACT_VERSION:
        raise ValueError(f"runtime child {role} does not use bb_first_v1")
    if snapshot.get("rules_version") != RULES_VERSION:
        raise ValueError(f"runtime child {role} has the wrong rules version")
    if snapshot.get("checkpoint_sha256") != binding["checkpoint_sha256"]:
        raise ValueError(f"runtime child {role} changed the calibrated checkpoint")
    if snapshot.get("adapter_source_sha256") != binding["adapter_source_sha256"]:
        raise ValueError(f"runtime child {role} changed the calibrated adapter")
    if snapshot.get("temperature") != temperature:
        raise ValueError(f"runtime child {role} did not apply the calibrated temperature")
    if snapshot.get("quantization_denominator") != quantization_denominator:
        raise ValueError(f"runtime child {role} has the wrong Q32 denominator")
    if snapshot.get("supported_turns") != [turn]:
        raise ValueError(f"runtime child {role} has the wrong supported turn")
    if snapshot.get("supported_actors") != [actor]:
        raise ValueError(f"runtime child {role} has the wrong supported actor")
    # The raw child remains an unpromoted ranking-logit adapter.  Promotion is
    # conferred only by this wrapper after the full raw bridge is rebuilt.
    if snapshot.get("promotion_eligible") is not False:
        raise ValueError(f"runtime child {role} must not self-promote")
    return child_sha256, snapshot


class VerifiedCalibratedHuT1T2BehaviorDispatch(FrozenBehaviorModel):
    """Exact, no-fallback router for bridge-verified T1/T2 likelihoods."""

    def __init__(
        self,
        routes: Mapping[tuple[int, str], FrozenBehaviorModel],
        manifest: Mapping[str, Any],
        *,
        _construction_token: object,
    ) -> None:
        if _construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError(
                "use build_verified_calibrated_hu_t1_t2_behavior_dispatch"
            )
        if set(routes) != _EXPECTED_ROUTE_SET:
            raise ValueError("calibrated runtime requires exactly four T1/T2 routes")
        snapshot = copy.deepcopy(dict(manifest))
        if snapshot.get("schema") != RUNTIME_SCHEMA:
            raise ValueError("calibrated runtime manifest schema mismatch")
        if snapshot.get("promotion_eligible") is not True:
            raise ValueError("calibrated runtime requires a promoted bridge")
        self._routes = MappingProxyType(dict(routes))
        self._manifest_snapshot = snapshot
        self._model_id = str(snapshot["model_id"])
        self._model_sha256 = canonical_sha256(snapshot)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_manifest(self) -> Mapping[str, Any]:
        # Return a detached nested snapshot so a caller cannot mutate the
        # content addressed identity retained by this runtime object.
        return MappingProxyType(copy.deepcopy(self._manifest_snapshot))

    @property
    def model_sha256(self) -> str:
        return self._model_sha256

    def action_distribution(self, information: BehaviorInfoSet) -> BehaviorDistribution:
        if not isinstance(information, BehaviorInfoSet):
            raise TypeError("information must be a BehaviorInfoSet")
        child = self._routes.get((information.turn, information.actor))
        if child is None:
            raise KeyError(
                f"no calibrated T1/T2 behavior route for "
                f"T{information.turn} {information.actor}; T3-BB requires the "
                "separate endogenous fixed-point runtime binding"
            )
        distribution = child.action_distribution(information)
        if not isinstance(distribution, BehaviorDistribution):
            raise TypeError("calibrated child must return BehaviorDistribution")
        if distribution.used_fallback or distribution.source == "uniform_fallback":
            raise RuntimeError("calibrated behavior runtime forbids fallback")
        return distribution


def build_verified_calibrated_hu_t1_t2_behavior_dispatch(
    records: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    calibration_artifact: Mapping[str, Any],
    bridge: Mapping[str, Any],
    *,
    training_root_ids: Sequence[str],
    t3_bb_likelihood_binding: Mapping[str, Any],
    challenge_records: Sequence[Mapping[str, Any]] = (),
    challenge_evaluation_rows: Sequence[Mapping[str, Any]] = (),
    workspace_root: str | Path | None = None,
    quantization_denominator: int = DEFAULT_QUANTIZATION_DENOMINATOR,
    model_id: str = "verified_calibrated_hu_t1_t2_behavior_dispatch_v1",
) -> VerifiedCalibratedHuT1T2BehaviorDispatch:
    """Rebuild all calibration evidence and load the exact calibrated routes."""

    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a non-empty string")
    if (
        isinstance(quantization_denominator, bool)
        or not isinstance(quantization_denominator, int)
        or quantization_denominator != DEFAULT_QUANTIZATION_DENOMINATOR
    ):
        raise ValueError("calibrated runtime requires the exact Q32 denominator")

    verified_bridge = verify_calibrated_behavior_bridge(
        records,
        evaluation_rows,
        calibration_artifact,
        bridge,
        training_root_ids=training_root_ids,
        t3_bb_likelihood_binding=t3_bb_likelihood_binding,
        challenge_records=challenge_records,
        challenge_evaluation_rows=challenge_evaluation_rows,
    )
    if verified_bridge.get("schema") != BRIDGE_SCHEMA:
        raise ValueError("verified calibration bridge schema mismatch")
    if verified_bridge.get("strategic_strength_evaluated") is not False:
        raise ValueError("calibration bridge must not claim strategic strength")
    if verified_bridge.get("strategic_strength_claimed") is not False:
        raise ValueError("calibration bridge must not claim strategic strength")
    bridge_sha256 = _require_sha256(
        verified_bridge.get("bridge_sha256"), label="bridge_sha256"
    )

    behavior = _require_mapping(
        verified_bridge.get("calibrated_behavior_manifest"),
        label="calibrated_behavior_manifest",
    )
    if behavior.get("schema") != BEHAVIOR_SCHEMA:
        raise ValueError("verified bridge behavior schema mismatch")
    if behavior.get("promotion_eligible") is not True:
        raise ValueError("verified bridge behavior is not promotion eligible")
    behavior_sha256 = _require_sha256(
        verified_bridge.get("calibrated_behavior_sha256"),
        label="calibrated_behavior_sha256",
    )
    if canonical_sha256(behavior) != behavior_sha256:
        raise ValueError("verified behavior manifest/hash mismatch")

    role_temperatures = _require_mapping(
        behavior.get("role_temperatures"), label="role_temperatures"
    )
    role_bindings = _require_mapping(
        behavior.get("role_model_bindings"), label="role_model_bindings"
    )
    if set(role_temperatures) != set(CALIBRATION_ROLE_KEYS):
        raise ValueError("verified bridge has incomplete role temperatures")
    if set(role_bindings) != set(CALIBRATION_ROLE_KEYS):
        raise ValueError("verified bridge has incomplete role model bindings")

    asset_routes = set(KNOWN_HU_POLICY_VALUE_ASSETS)
    if asset_routes != _EXPECTED_ROUTE_SET:
        raise ValueError("known HU policy-value asset registry has drifted")
    if set(APPROVED_HU_POLICY_VALUE_CHECKPOINTS) != _EXPECTED_ROUTE_SET:
        raise AssertionError("approved HU checkpoint registry is incomplete")
    root = (
        Path(workspace_root).resolve()
        if workspace_root is not None
        else Path(__file__).resolve().parents[2]
    )
    evaluators = build_known_hu_policy_value_logit_evaluators(
        root, quantization_denominator=quantization_denominator
    )
    if set(evaluators) != _EXPECTED_ROUTE_SET:
        raise ValueError("known logit evaluator loader returned incomplete routes")

    routes: dict[tuple[int, str], FrozenBehaviorModel] = {}
    route_manifest: list[dict[str, Any]] = []
    artifact_sha256 = _require_sha256(
        behavior.get("calibration_artifact_sha256"),
        label="calibration_artifact_sha256",
    )
    for turn, actor in EXPECTED_ROUTES:
        role = _role(turn, actor)
        asset = _require_mapping(
            KNOWN_HU_POLICY_VALUE_ASSETS[(turn, actor)],
            label=f"known asset {role}",
        )
        if set(asset) != {"relative_path", "checkpoint_sha256"}:
            raise ValueError(f"known asset {role} has unexpected fields")
        relative_path = asset.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"known asset {role} path is invalid")
        known_checkpoint_sha256 = _require_sha256(
            asset.get("checkpoint_sha256"),
            label=f"known asset {role} checkpoint_sha256",
        )
        approved_checkpoint_sha256 = APPROVED_HU_POLICY_VALUE_CHECKPOINTS[
            (turn, actor)
        ]
        if known_checkpoint_sha256 != approved_checkpoint_sha256:
            raise ValueError(
                f"known asset {role} checkpoint is not the approved exact hash"
            )
        binding = _require_mapping(
            role_bindings[role], label=f"role_model_bindings.{role}"
        )
        _validated_identity_evaluator(
            evaluators[(turn, actor)],
            role=role,
            expected_binding=binding,
            expected_checkpoint_sha256=known_checkpoint_sha256,
        )
        temperature = _canonical_temperature(role_temperatures[role], role=role)
        child = TorchPolicyValueBehaviorModel(
            root / relative_path,
            model_id=(
                f"calibrated_hu_{role}_policyvalue_q32_"
                f"{artifact_sha256[:12]}"
            ),
            supported_turns=(turn,),
            supported_actors=(actor,),
            training_scope_id=f"hu_{role}_visible_opponent_board_2m_v1",
            temperature=temperature,
            quantization_denominator=quantization_denominator,
            expected_checkpoint_sha256=known_checkpoint_sha256,
        )
        child_sha256, child_manifest = _validated_runtime_child(
            child,
            turn=turn,
            actor=actor,
            role=role,
            binding=binding,
            temperature=temperature,
            quantization_denominator=quantization_denominator,
        )
        routes[(turn, actor)] = child
        route_manifest.append(
            {
                "turn": turn,
                "actor": actor,
                "role": role,
                "temperature": temperature,
                "checkpoint_sha256": binding["checkpoint_sha256"],
                "calibration_source_model_sha256": binding["model_sha256"],
                "row_extractor_sha256": binding["row_extractor_sha256"],
                "adapter_source_sha256": binding["adapter_source_sha256"],
                "child_model_id": child.model_id,
                "child_model_sha256": child_sha256,
                "child_model_type": child_manifest.get("model_type"),
            }
        )

    t3_binding = _require_mapping(
        behavior.get("t3_bb_likelihood_binding"),
        label="t3_bb_likelihood_binding",
    )
    t3_binding_sha256 = _require_sha256(
        t3_binding.get("binding_sha256"),
        label="t3_bb_likelihood_binding.binding_sha256",
    )
    manifest: dict[str, Any] = {
        "schema": RUNTIME_SCHEMA,
        "model_id": model_id,
        "model_type": RUNTIME_MODEL_TYPE,
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "bb_first": True,
        "rules_version": RULES_VERSION,
        "promotion_eligible": True,
        "promotion_scope": RUNTIME_ROUTE_SCOPE,
        "promotion_basis": "verified_full_calibration_bridge_including_t3_bb_binding",
        "strategic_strength_evaluated": False,
        "strategic_strength_claimed": False,
        "standalone_full_m3_behavior_complete": False,
        "t3_bb_route_included": False,
        "t3_bb_runtime_integration": "required_separate_endogenous_fixed_point_route",
        "no_fallback": True,
        "temperature_application": "checkpoint_logits_div_temperature_then_softmax",
        "probability_quantization": "largest_remainder_exact_q32",
        "quantization_denominator": quantization_denominator,
        "calibrated_bridge_sha256": bridge_sha256,
        "calibrated_behavior_manifest_sha256": behavior_sha256,
        "calibration_artifact_sha256": artifact_sha256,
        "calibration_gate_config_sha256": _require_sha256(
            behavior.get("calibration_gate_config_sha256"),
            label="calibration_gate_config_sha256",
        ),
        "calibration_gate_result_sha256": _require_sha256(
            behavior.get("calibration_gate_result_sha256"),
            label="calibration_gate_result_sha256",
        ),
        "t3_bb_likelihood_binding_sha256": t3_binding_sha256,
        "routes": sorted(route_manifest, key=lambda row: (row["turn"], row["actor"])),
    }
    return VerifiedCalibratedHuT1T2BehaviorDispatch(
        routes, manifest, _construction_token=_CONSTRUCTION_TOKEN
    )


__all__ = [
    "APPROVED_HU_POLICY_VALUE_CHECKPOINTS",
    "EXPECTED_ROUTES",
    "RUNTIME_MODEL_TYPE",
    "RUNTIME_ROUTE_SCOPE",
    "RUNTIME_SCHEMA",
    "VerifiedCalibratedHuT1T2BehaviorDispatch",
    "build_verified_calibrated_hu_t1_t2_behavior_dispatch",
]
