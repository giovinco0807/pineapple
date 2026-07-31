"""Non-promotable calibrated T1/T2 runtime for T3 fixed-point bootstrap.

The final T3-BB behavior likelihood is an output of the candidate-policy /
posterior fixed-point process, so requiring that binding in order to run the
process would be circular.  This module provides the deliberately narrower
bootstrap boundary: it rebuilds the complete T1/T2 temperature-calibration
artifact from raw main and Joker-challenge evidence, binds the four exact
checkpoint/evaluator identities, and exposes only those four no-fallback
routes.

Complete raw revalidation is required, but the source calibration gate need
not be promotion-passing: a small non-promoting smoke artifact is useful for
executing the fixed-point pipeline before production evidence exists.  The
source gate status is recorded exactly in the manifest.  The resulting
runtime is permanently marked ``promotion_eligible=false`` and
``fixed_point_bootstrap_only=true``; it is not a strategic-strength or M3
promotion claim and does not accept a bridge or provisional/fake T3 binding.
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
from ai.tutor.behavior_temperature_calibration import (
    CALIBRATION_SCHEMA,
    ROLE_KEYS,
    TEMPERATURE_DENOMINATOR,
    TEMPERATURE_MAX_NUMERATOR,
    TEMPERATURE_MIN_NUMERATOR,
    verify_behavior_temperature_calibration,
)
from ai.tutor.calibrated_behavior_runtime import (
    APPROVED_HU_POLICY_VALUE_CHECKPOINTS,
    EXPECTED_ROUTES,
    RUNTIME_SCHEMA,
    _require_mapping,
    _require_sha256,
    _validated_identity_evaluator,
    _validated_runtime_child,
)
from ai.tutor.frozen_behavior_torch import (
    DEFAULT_QUANTIZATION_DENOMINATOR,
    KNOWN_HU_POLICY_VALUE_ASSETS,
    RULES_VERSION,
    TorchPolicyValueBehaviorModel,
)
from ai.tutor.promotion_gate_m3_full_card_strength import (
    CALIBRATION_BINDING_KEYS,
    canonical_sha256,
)
from ai.tutor.t3_hu_full_card_range import (
    BehaviorDistribution,
    BehaviorInfoSet,
    FrozenBehaviorModel,
)


BOOTSTRAP_MODEL_TYPE = (
    "verified_calibrated_t1_t2_fixed_point_bootstrap_q32_dispatch_v1"
)
BOOTSTRAP_ROUTE_SCOPE = "t1_t2_only"
_EXPECTED_ROUTE_SET = frozenset(EXPECTED_ROUTES)
_EXPECTED_ROLE_SET = frozenset(ROLE_KEYS)
_CONSTRUCTION_TOKEN = object()


def _role(turn: int, actor: str) -> str:
    return f"t{turn}_{actor}"


def _source_calibration_status(artifact: Mapping[str, Any]) -> tuple[bool, bool, int]:
    """Validate and return source gate status without requiring it to pass."""

    if artifact.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError("verified artifact has an unsupported calibration schema")
    artifact_eligible = artifact.get("promotion_eligible")
    if not isinstance(artifact_eligible, bool):
        raise TypeError("temperature calibration promotion_eligible must be boolean")
    gate_result = _require_mapping(artifact.get("gate_result"), label="gate_result")
    gate_eligible = gate_result.get("promotion_eligible")
    all_passed = gate_result.get("all_required_gates_passed")
    if not isinstance(gate_eligible, bool) or not isinstance(all_passed, bool):
        raise TypeError("temperature calibration gate status must be boolean")
    failures = gate_result.get("failures")
    if not isinstance(failures, list) or any(
        not isinstance(failure, str) or not failure for failure in failures
    ):
        raise TypeError("temperature calibration gate failures must be strings")
    if artifact_eligible != gate_eligible or gate_eligible != all_passed:
        raise ValueError("temperature calibration promotion status is inconsistent")
    if all_passed != (failures == []):
        raise ValueError("temperature calibration failures/status is inconsistent")
    return artifact_eligible, all_passed, len(failures)


def _artifact_role_temperatures(artifact: Mapping[str, Any]) -> dict[str, str]:
    """Extract exact reduced rational temperatures from the rebuilt artifact."""

    temperatures = _require_mapping(artifact.get("temperatures"), label="temperatures")
    if set(temperatures) != _EXPECTED_ROLE_SET:
        raise ValueError("temperatures must contain exactly the four T1/T2 roles")
    result: dict[str, str] = {}
    for role in ROLE_KEYS:
        role_payload = _require_mapping(
            temperatures[role], label=f"temperatures.{role}"
        )
        payload = _require_mapping(
            role_payload.get("final_temperature"),
            label=f"temperatures.{role}.final_temperature",
        )
        if set(payload) != {"numerator", "denominator"}:
            raise ValueError(
                f"temperatures.{role}.final_temperature must contain exact "
                "numerator/denominator"
            )
        numerator = payload.get("numerator")
        denominator = payload.get("denominator")
        if isinstance(numerator, bool) or not isinstance(numerator, int):
            raise TypeError(
                f"temperatures.{role}.final_temperature.numerator must be an integer"
            )
        if denominator != TEMPERATURE_DENOMINATOR:
            raise ValueError(
                f"temperatures.{role}.final_temperature denominator must be "
                f"{TEMPERATURE_DENOMINATOR}"
            )
        if not TEMPERATURE_MIN_NUMERATOR <= numerator <= TEMPERATURE_MAX_NUMERATOR:
            raise ValueError(
                f"temperatures.{role}.final_temperature is outside [1/20,20]"
            )
        temperature = Fraction(numerator, denominator)
        result[role] = f"{temperature.numerator}/{temperature.denominator}"
    return result


def _artifact_role_bindings(
    artifact: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    bindings = _require_mapping(
        artifact.get("role_model_bindings"), label="role_model_bindings"
    )
    if set(bindings) != _EXPECTED_ROLE_SET:
        raise ValueError(
            "role_model_bindings must contain exactly the four T1/T2 roles"
        )
    result: dict[str, dict[str, str]] = {}
    for role in ROLE_KEYS:
        binding = _require_mapping(
            bindings[role], label=f"role_model_bindings.{role}"
        )
        if set(binding) != set(CALIBRATION_BINDING_KEYS):
            raise ValueError(
                f"role_model_bindings.{role} must contain exact checkpoint/model/"
                "extractor/adapter hashes"
            )
        result[role] = {
            field: _require_sha256(
                binding[field], label=f"role_model_bindings.{role}.{field}"
            )
            for field in CALIBRATION_BINDING_KEYS
        }
    return result


class FixedPointBootstrapCalibratedHuT1T2BehaviorDispatch(FrozenBehaviorModel):
    """Strict four-route likelihood component used only to produce a fixed point."""

    def __init__(
        self,
        routes: Mapping[tuple[int, str], FrozenBehaviorModel],
        manifest: Mapping[str, Any],
        *,
        _construction_token: object,
    ) -> None:
        if _construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError(
                "use build_fixed_point_bootstrap_calibrated_hu_t1_t2_behavior_dispatch"
            )
        if set(routes) != _EXPECTED_ROUTE_SET:
            raise ValueError("bootstrap runtime requires exactly four T1/T2 routes")
        snapshot = copy.deepcopy(dict(manifest))
        if snapshot.get("schema") != RUNTIME_SCHEMA:
            raise ValueError("bootstrap runtime manifest schema mismatch")
        if snapshot.get("promotion_eligible") is not False:
            raise ValueError("bootstrap runtime must be non-promotable")
        if snapshot.get("fixed_point_bootstrap_only") is not True:
            raise ValueError("bootstrap runtime must be fixed-point-only")
        if snapshot.get("strategic_strength") is not False:
            raise ValueError("bootstrap runtime must not claim strategic strength")
        if snapshot.get("route_scope") != BOOTSTRAP_ROUTE_SCOPE:
            raise ValueError("bootstrap runtime must expose only T1/T2 routes")
        if snapshot.get("no_fallback") is not True:
            raise ValueError("bootstrap runtime must forbid fallback")
        self._routes = MappingProxyType(dict(routes))
        self._manifest_snapshot = snapshot
        self._model_id = str(snapshot["model_id"])
        self._model_sha256 = canonical_sha256(snapshot)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_manifest(self) -> Mapping[str, Any]:
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
                f"fixed-point bootstrap has no route for T{information.turn} "
                f"{information.actor}; route_scope={BOOTSTRAP_ROUTE_SCOPE}"
            )
        distribution = child.action_distribution(information)
        if not isinstance(distribution, BehaviorDistribution):
            raise TypeError("bootstrap child must return BehaviorDistribution")
        if distribution.used_fallback or distribution.source == "uniform_fallback":
            raise RuntimeError("fixed-point bootstrap behavior runtime forbids fallback")
        return distribution


def build_fixed_point_bootstrap_calibrated_hu_t1_t2_behavior_dispatch(
    records: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    calibration_artifact: Mapping[str, Any],
    *,
    challenge_records: Sequence[Mapping[str, Any]] = (),
    challenge_evaluation_rows: Sequence[Mapping[str, Any]] = (),
    workspace_root: str | Path | None = None,
    quantization_denominator: int = DEFAULT_QUANTIZATION_DENOMINATOR,
    require_promoted_source: bool = False,
    model_id: str = "fixed_point_bootstrap_calibrated_hu_t1_t2_dispatch_v1",
) -> FixedPointBootstrapCalibratedHuT1T2BehaviorDispatch:
    """Rebuild raw calibration and load exact no-fallback T1/T2 routes.

    Deliberately absent inputs: a calibrated-behavior bridge, training root
    partition, and T3-BB likelihood binding.  Those are downstream products or
    promotion evidence and would create a circular bootstrap dependency here.
    """

    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a non-empty string")
    if not isinstance(require_promoted_source, bool):
        raise TypeError("require_promoted_source must be boolean")
    if (
        isinstance(quantization_denominator, bool)
        or not isinstance(quantization_denominator, int)
        or quantization_denominator != DEFAULT_QUANTIZATION_DENOMINATOR
    ):
        raise ValueError("bootstrap runtime requires the exact Q32 denominator")

    # This is the trust boundary: every record, direct pre-temperature logit,
    # split, metric, bootstrap interval, gate result, and self hash is rebuilt.
    verified_artifact = verify_behavior_temperature_calibration(
        records,
        evaluation_rows,
        calibration_artifact,
        challenge_records=challenge_records,
        challenge_evaluation_rows=challenge_evaluation_rows,
    )
    (
        source_calibration_promotion_eligible,
        calibration_gate_passed,
        calibration_gate_failure_count,
    ) = _source_calibration_status(verified_artifact)
    if require_promoted_source and not (
        source_calibration_promotion_eligible and calibration_gate_passed
    ):
        raise ValueError(
            "production bootstrap requires a promotion-eligible source calibration "
            "with all required gates passed"
        )
    artifact_sha256 = _require_sha256(
        verified_artifact.get("artifact_sha256"), label="artifact_sha256"
    )
    role_temperatures = _artifact_role_temperatures(verified_artifact)
    role_bindings = _artifact_role_bindings(verified_artifact)

    if set(KNOWN_HU_POLICY_VALUE_ASSETS) != _EXPECTED_ROUTE_SET:
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
        if (
            known_checkpoint_sha256
            != APPROVED_HU_POLICY_VALUE_CHECKPOINTS[(turn, actor)]
        ):
            raise ValueError(
                f"known asset {role} checkpoint is not the approved exact hash"
            )
        binding = role_bindings[role]
        _validated_identity_evaluator(
            evaluators[(turn, actor)],
            role=role,
            expected_binding=binding,
            expected_checkpoint_sha256=known_checkpoint_sha256,
        )
        temperature = role_temperatures[role]
        child = TorchPolicyValueBehaviorModel(
            root / relative_path,
            model_id=(
                f"fixed_point_bootstrap_{role}_policyvalue_q32_"
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

    gate_config = _require_mapping(
        verified_artifact.get("gate_config"), label="gate_config"
    )
    gate_result = _require_mapping(
        verified_artifact.get("gate_result"), label="gate_result"
    )
    challenge = _require_mapping(
        verified_artifact.get("joker_challenge"), label="joker_challenge"
    )
    manifest: dict[str, Any] = {
        "schema": RUNTIME_SCHEMA,
        "model_id": model_id,
        "model_type": BOOTSTRAP_MODEL_TYPE,
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "bb_first": True,
        "rules_version": RULES_VERSION,
        "calibrated": True,
        "calibration_verified_from_raw": True,
        "source_calibration_promotion_eligible": (
            source_calibration_promotion_eligible
        ),
        "source_calibration_all_required_gates_passed": calibration_gate_passed,
        "source_calibration_gate_failure_count": calibration_gate_failure_count,
        "source_promotion_required_at_construction": require_promoted_source,
        "promotion_eligible": False,
        "fixed_point_bootstrap_only": True,
        "strategic_strength": False,
        "strategic_strength_evaluated": False,
        "strategic_strength_claimed": False,
        "route_scope": BOOTSTRAP_ROUTE_SCOPE,
        "no_fallback": True,
        "t3_bb_route_included": False,
        "t3_bb_likelihood_binding_included": False,
        "calibrated_behavior_bridge_included": False,
        "temperature_application": (
            "checkpoint_logits_div_temperature_then_softmax"
        ),
        "probability_quantization": "largest_remainder_exact_q32",
        "quantization_denominator": quantization_denominator,
        "calibration_artifact_sha256": artifact_sha256,
        "calibration_gate_config_sha256": _require_sha256(
            gate_config.get("gate_config_sha256"), label="gate_config_sha256"
        ),
        "calibration_gate_result_sha256": _require_sha256(
            gate_result.get("gate_result_sha256"), label="gate_result_sha256"
        ),
        "raw_record_set_sha256": _require_sha256(
            verified_artifact.get("raw_record_set_sha256"),
            label="raw_record_set_sha256",
        ),
        "evaluation_row_set_sha256": _require_sha256(
            verified_artifact.get("evaluation_row_set_sha256"),
            label="evaluation_row_set_sha256",
        ),
        "challenge_raw_record_set_sha256": _require_sha256(
            challenge.get("raw_record_set_sha256"),
            label="joker_challenge.raw_record_set_sha256",
        ),
        "challenge_evaluation_row_set_sha256": _require_sha256(
            challenge.get("evaluation_row_set_sha256"),
            label="joker_challenge.evaluation_row_set_sha256",
        ),
        "routes": sorted(route_manifest, key=lambda row: (row["turn"], row["actor"])),
    }
    return FixedPointBootstrapCalibratedHuT1T2BehaviorDispatch(
        routes, manifest, _construction_token=_CONSTRUCTION_TOKEN
    )


__all__ = [
    "BOOTSTRAP_MODEL_TYPE",
    "BOOTSTRAP_ROUTE_SCOPE",
    "FixedPointBootstrapCalibratedHuT1T2BehaviorDispatch",
    "build_fixed_point_bootstrap_calibrated_hu_t1_t2_behavior_dispatch",
]
