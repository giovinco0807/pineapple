"""Strict bridge from behavior calibration to the M3 strength-gate contract.

The temperature-calibration artifact and the full-card strength gate are kept
separate deliberately: calibration establishes that a frozen behavior model's
action likelihoods are usable, while the strength gate evaluates a T3/T4
policy on an independent holdout.  This module translates only the former
contract into the exact behavior manifest consumed by the latter.

Every source calibration artifact is rebuilt from raw decision records and
content-bound pre-temperature evaluation rows before translation.  Root
partitions use the strength gate's split-independent root-identity commitment
domain.  The calibration partition contains *all* main fit/dev/test roots and
all targeted Joker-challenge roots, so none may later be reused as strength
holdout evidence.

Passing this bridge is not evidence of strategic strength.

The temperature artifact covers exactly T1/T2 x BB/BTN.  A BTN ``t3_second``
root additionally observes BB's T3 action, so its hidden-discard posterior
requires a distinct T3-BB likelihood.  The bridge therefore also requires a
content-addressed, promotion-eligible candidate-policy/posterior fixed-point
binding; a frozen ranking prior cannot satisfy that contract.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any, Mapping, Sequence

from ai.tutor.behavior_calibration_contract import verify_behavior_decision_log
from ai.tutor.behavior_temperature_calibration import (
    ROLE_KEYS,
    verify_behavior_temperature_calibration,
)
from ai.tutor.promotion_gate_m3_full_card_strength import (
    BEHAVIOR_MODEL_TYPE,
    BEHAVIOR_SCHEMA,
    CALIBRATED_BEHAVIOR_KEYS,
    CALIBRATION_BINDING_KEYS,
    ROOT_PARTITION_SCHEMA,
    canonical_json,
    canonical_sha256,
    root_identity_commitment_sha256,
    self_hash,
    verify_t3_bb_likelihood_binding,
)


BRIDGE_SCHEMA = "ofc_calibrated_behavior_bridge/v1"
MODEL_TYPE = BEHAVIOR_MODEL_TYPE
_PARTITION_PURPOSES = frozenset({"training", "calibration"})
_SHA256_LENGTH = 64
_BRIDGE_KEYS = frozenset(
    {
        "schema",
        "strategic_strength_evaluated",
        "strategic_strength_claimed",
        "calibrated_behavior_manifest",
        "calibrated_behavior_sha256",
        "root_partitions",
        "bridge_sha256",
    }
)


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], *, label: str
) -> None:
    actual = frozenset(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"{label} keys mismatch: missing={missing}, extra={extra}")


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _root_ids(
    records: Sequence[Mapping[str, Any]], *, label: str
) -> set[str]:
    roots: set[str] = set()
    for record in records:
        verified = verify_behavior_decision_log(record)
        roots.add(verified.root_id)
    if not roots:
        raise ValueError(f"{label} must contain at least one root")
    return roots


def build_root_partition_manifest(
    *, purpose: str, root_ids: Sequence[str]
) -> dict[str, Any]:
    """Build the exact root-partition manifest used by the strength gate.

    Root IDs are accepted only as builder inputs.  The returned artifact emits
    their domain-separated identity commitments, not the IDs themselves.
    Duplicate input IDs are rejected rather than silently coalesced.
    """

    if purpose not in _PARTITION_PURPOSES:
        raise ValueError("partition purpose must be 'training' or 'calibration'")
    if isinstance(root_ids, (str, bytes)):
        raise TypeError("root_ids must be a sequence of complete root ID strings")
    normalized: list[str] = []
    for index, root_id in enumerate(root_ids):
        if not isinstance(root_id, str) or not root_id:
            raise ValueError(f"root_ids[{index}] must be a non-empty string")
        normalized.append(root_id)
    if not normalized:
        raise ValueError(f"{purpose} root partition must not be empty")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{purpose} root partition contains duplicate root IDs")

    partition: dict[str, Any] = {
        "schema": ROOT_PARTITION_SCHEMA,
        "purpose": purpose,
        "root_commitments": sorted(
            root_identity_commitment_sha256(root_id) for root_id in normalized
        ),
    }
    partition["manifest_sha256"] = self_hash(partition, "manifest_sha256")
    return partition


def _canonical_temperature(value: Any, *, role: str) -> str:
    payload = _require_mapping(value, label=f"temperatures.{role}.final_temperature")
    if set(payload) != {"numerator", "denominator"}:
        raise ValueError(
            f"temperatures.{role}.final_temperature must have exact numerator/denominator"
        )
    numerator = payload.get("numerator")
    denominator = payload.get("denominator")
    if (
        not isinstance(numerator, int)
        or isinstance(numerator, bool)
        or not isinstance(denominator, int)
        or isinstance(denominator, bool)
        or numerator <= 0
        or denominator <= 0
    ):
        raise ValueError(f"temperatures.{role}.final_temperature must be positive integers")
    temperature = Fraction(numerator, denominator)
    if not Fraction(1, 20) <= temperature <= Fraction(20, 1):
        raise ValueError(f"temperatures.{role}.final_temperature is outside [1/20,20]")
    return f"{temperature.numerator}/{temperature.denominator}"


def _role_temperatures(artifact: Mapping[str, Any]) -> dict[str, str]:
    temperatures = _require_mapping(artifact.get("temperatures"), label="temperatures")
    if set(temperatures) != set(ROLE_KEYS):
        raise ValueError("temperatures must contain exactly t1_bb/t1_btn/t2_bb/t2_btn")
    return {
        role: _canonical_temperature(
            _require_mapping(temperatures[role], label=f"temperatures.{role}").get(
                "final_temperature"
            ),
            role=role,
        )
        for role in ROLE_KEYS
    }


def _role_model_bindings(artifact: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    bindings = _require_mapping(
        artifact.get("role_model_bindings"), label="role_model_bindings"
    )
    if set(bindings) != set(ROLE_KEYS):
        raise ValueError(
            "role_model_bindings must contain exactly t1_bb/t1_btn/t2_bb/t2_btn"
        )
    result: dict[str, dict[str, str]] = {}
    for role in ROLE_KEYS:
        binding = _require_mapping(bindings[role], label=f"role_model_bindings.{role}")
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


def _require_promoted_calibration(artifact: Mapping[str, Any]) -> None:
    if artifact.get("promotion_eligible") is not True:
        raise ValueError("temperature calibration artifact is not promotion eligible")
    gate_result = _require_mapping(artifact.get("gate_result"), label="gate_result")
    if gate_result.get("promotion_eligible") is not True:
        raise ValueError("temperature calibration gate result is not promotion eligible")
    if gate_result.get("all_required_gates_passed") is not True:
        raise ValueError("temperature calibration gate did not pass all required gates")
    failures = gate_result.get("failures")
    if failures != []:
        raise ValueError("promoted temperature calibration gate must have no failures")


def _build_behavior_manifest(
    artifact: Mapping[str, Any],
    *,
    training_partition: Mapping[str, Any],
    calibration_partition: Mapping[str, Any],
    t3_bb_likelihood_binding: Mapping[str, Any],
) -> dict[str, Any]:
    gate_config = _require_mapping(artifact.get("gate_config"), label="gate_config")
    gate_result = _require_mapping(artifact.get("gate_result"), label="gate_result")
    manifest: dict[str, Any] = {
        "schema": BEHAVIOR_SCHEMA,
        "model_type": MODEL_TYPE,
        "calibrated": True,
        "calibration_method": "t1_t2_observed_action_nll_temperature",
        "calibration_dataset_kind": "observed_full_trace_actions",
        "root_split": "root_disjoint_train_calibration_test",
        "promotion_eligible": True,
        "role_temperatures": _role_temperatures(artifact),
        "calibration_artifact_sha256": _require_sha256(
            artifact.get("artifact_sha256"), label="artifact_sha256"
        ),
        "calibration_gate_config_sha256": _require_sha256(
            gate_config.get("gate_config_sha256"), label="gate_config_sha256"
        ),
        "calibration_gate_result_sha256": _require_sha256(
            gate_result.get("gate_result_sha256"), label="gate_result_sha256"
        ),
        "role_model_bindings": _role_model_bindings(artifact),
        # BTN's T3 root observes BB's T3 action.  Its hidden-discard posterior
        # therefore needs a fifth likelihood route that cannot be supplied by
        # the exogenous T1/T2 temperature-calibration artifact.  Promotion
        # accepts only an explicitly converged endogenous candidate-policy
        # fixed-point binding here.
        "t3_bb_likelihood_binding": verify_t3_bb_likelihood_binding(
            t3_bb_likelihood_binding
        ),
        "training_root_partition_sha256": _require_sha256(
            training_partition.get("manifest_sha256"),
            label="training_root_partition.manifest_sha256",
        ),
        "calibration_root_partition_sha256": _require_sha256(
            calibration_partition.get("manifest_sha256"),
            label="calibration_root_partition.manifest_sha256",
        ),
    }
    if set(manifest) != set(CALIBRATED_BEHAVIOR_KEYS):
        raise AssertionError("bridge output drifted from the strength-gate field contract")
    return manifest


def build_calibrated_behavior_bridge(
    records: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    calibration_artifact: Mapping[str, Any],
    *,
    training_root_ids: Sequence[str],
    t3_bb_likelihood_binding: Mapping[str, Any],
    challenge_records: Sequence[Mapping[str, Any]] = (),
    challenge_evaluation_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Rebuild calibration and create a strength-gate-compatible bridge bundle."""

    verified_artifact = verify_behavior_temperature_calibration(
        records,
        evaluation_rows,
        calibration_artifact,
        challenge_records=challenge_records,
        challenge_evaluation_rows=challenge_evaluation_rows,
    )
    _require_promoted_calibration(verified_artifact)

    main_roots = _root_ids(records, label="main calibration records")
    challenge_roots = (
        _root_ids(challenge_records, label="Joker challenge records")
        if challenge_records
        else set()
    )
    calibration_roots = main_roots | challenge_roots
    normalized_training_roots = list(training_root_ids)
    training_root_set = set(normalized_training_roots)
    if training_root_set & calibration_roots:
        raise ValueError("training and calibration root partitions overlap")

    training_partition = build_root_partition_manifest(
        purpose="training", root_ids=normalized_training_roots
    )
    calibration_partition = build_root_partition_manifest(
        purpose="calibration", root_ids=sorted(calibration_roots)
    )
    behavior_manifest = _build_behavior_manifest(
        verified_artifact,
        training_partition=training_partition,
        calibration_partition=calibration_partition,
        t3_bb_likelihood_binding=t3_bb_likelihood_binding,
    )

    bridge: dict[str, Any] = {
        "schema": BRIDGE_SCHEMA,
        "strategic_strength_evaluated": False,
        "strategic_strength_claimed": False,
        "calibrated_behavior_manifest": behavior_manifest,
        "calibrated_behavior_sha256": canonical_sha256(behavior_manifest),
        "root_partitions": {
            "training": training_partition,
            "calibration": calibration_partition,
        },
    }
    bridge["bridge_sha256"] = self_hash(bridge, "bridge_sha256")
    return bridge


def verify_calibrated_behavior_bridge(
    records: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    calibration_artifact: Mapping[str, Any],
    bridge: Mapping[str, Any],
    *,
    training_root_ids: Sequence[str],
    t3_bb_likelihood_binding: Mapping[str, Any],
    challenge_records: Sequence[Mapping[str, Any]] = (),
    challenge_evaluation_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Rebuild the complete bridge and require exact byte-semantic equality."""

    raw = _require_mapping(bridge, label="calibrated behavior bridge")
    _require_exact_keys(raw, _BRIDGE_KEYS, label="calibrated behavior bridge")
    if raw.get("schema") != BRIDGE_SCHEMA:
        raise ValueError("unsupported calibrated behavior bridge schema")
    if raw.get("strategic_strength_evaluated") is not False:
        raise ValueError("bridge must not claim that strategic strength was evaluated")
    if raw.get("strategic_strength_claimed") is not False:
        raise ValueError("bridge must not claim strategic strength")
    claimed_hash = _require_sha256(raw.get("bridge_sha256"), label="bridge_sha256")
    if claimed_hash != self_hash(raw, "bridge_sha256"):
        raise ValueError("calibrated behavior bridge SHA-256 mismatch")

    rebuilt = build_calibrated_behavior_bridge(
        records,
        evaluation_rows,
        calibration_artifact,
        training_root_ids=training_root_ids,
        t3_bb_likelihood_binding=t3_bb_likelihood_binding,
        challenge_records=challenge_records,
        challenge_evaluation_rows=challenge_evaluation_rows,
    )
    if canonical_json(raw) != canonical_json(rebuilt):
        raise ValueError("calibrated behavior bridge does not match raw inputs")
    return rebuilt


__all__ = [
    "BRIDGE_SCHEMA",
    "MODEL_TYPE",
    "build_calibrated_behavior_bridge",
    "build_root_partition_manifest",
    "verify_calibrated_behavior_bridge",
]
