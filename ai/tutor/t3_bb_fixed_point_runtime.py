"""Runtime binding for a fixed-point-verified T3-BB behavior policy.

The T1/T2 calibrated dispatch intentionally cannot answer the fifth behavior
query used by a BTN ``t3_second`` root: the immediately preceding BB T3
action.  This module loads that route from a content-addressed exact policy
table, binds it to the passing fixed-point likelihood record, and composes it
with the verified T1/T2 runtime without adding a fallback.

The candidate table is not a strategic-strength claim by itself.  Its
``promotion_eligible`` field remains false; only the composite likelihood
runtime is eligible for building M3 T3 root posteriors, and only when the exact
candidate artifact/checkpoint/source hashes match the reverified fixed-point
binding.
"""
from __future__ import annotations

import copy
import json
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from ai.engine.turn_order import POSITION_CONTRACT_VERSION
from ai.tutor.calibrated_behavior_runtime import (
    RUNTIME_MODEL_TYPE as T1_T2_RUNTIME_MODEL_TYPE,
    VerifiedCalibratedHuT1T2BehaviorDispatch,
)
from ai.tutor.calibrated_behavior_bootstrap import (
    BOOTSTRAP_MODEL_TYPE,
    FixedPointBootstrapCalibratedHuT1T2BehaviorDispatch,
)
from ai.tutor.frozen_behavior_torch import RULES_VERSION
from ai.tutor.promotion_gate_m3_full_card_strength import (
    SOLVER_METHOD,
    T3_BB_LIKELIHOOD_METHOD,
    canonical_sha256,
    verify_t3_bb_likelihood_binding,
)
from ai.tutor.t3_hu_full_card_range import (
    BehaviorDistribution,
    BehaviorInfoSet,
    FrozenBehaviorModel,
)


ARTIFACT_SCHEMA = "ofc_t3_bb_candidate_policy_artifact/v1"
MODEL_SCHEMA = "ofc_frozen_behavior_model/v1"
MODEL_TYPE = "fixed_point_full_card_mccfr_exact_t3_bb_table_v1"
COMPOSITE_MODEL_TYPE = "verified_calibrated_t1_t2_plus_fixed_point_t3_bb_v1"
COMPOSITE_SCOPE = "m3_t3_root_behavior_likelihood_routes"
ITERATION_MODEL_TYPE = "nonpromoted_t1_t2_plus_candidate_t3_bb_iteration_v1"
ITERATION_SCOPE = "t3_bb_candidate_policy_posterior_fixed_point_iteration_only"
_SHA256_LENGTH = 64
_ARTIFACT_KEYS = frozenset({"schema", "model_manifest", "artifact_sha256"})
_MODEL_KEYS = frozenset(
    {
        "schema",
        "model_id",
        "model_type",
        "position_contract_version",
        "rules_version",
        "promotion_eligible",
        "promotion_basis",
        "supported_turns",
        "supported_actors",
        "no_fallback",
        "probability_encoding",
        "candidate_policy_method",
        "solver_manifest_sha256",
        "range_builder_source_sha256",
        "checkpoint_sha256",
        "probabilities",
    }
)
_COMPOSITE_TOKEN = object()
_CHILD_TOKEN = object()
_ITERATION_CHILD_TOKEN = object()
_ITERATION_DISPATCH_TOKEN = object()


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


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], *, label: str
) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} requires exact versioned fields; "
            f"missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _canonical_probability(value: Any, *, label: str) -> Fraction:
    if not isinstance(value, str) or value.strip() != value or "/" not in value:
        raise TypeError(f"{label} must be a canonical rational string")
    try:
        probability = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"{label} must be a canonical rational string") from exc
    canonical = f"{probability.numerator}/{probability.denominator}"
    if value != canonical:
        raise ValueError(f"{label} must be reduced canonical rational data")
    if not 0 <= probability <= 1:
        raise ValueError(f"{label} must be in [0, 1]")
    return probability


def _normalize_probabilities(
    raw: Any, *, label: str
) -> dict[str, Mapping[str, Fraction]]:
    table = _require_mapping(raw, label=label)
    if not table:
        raise ValueError(f"{label} must contain at least one information set")
    normalized: dict[str, Mapping[str, Fraction]] = {}
    for digest, raw_row in sorted(table.items()):
        information_digest = _require_sha256(
            digest, label=f"{label} information digest"
        )
        row = _require_mapping(raw_row, label=f"{label}.{information_digest}")
        if not row:
            raise ValueError(f"{label}.{information_digest} must not be empty")
        probabilities: dict[str, Fraction] = {}
        for action_id, raw_probability in sorted(row.items()):
            if not isinstance(action_id, str) or not action_id:
                raise ValueError(
                    f"{label}.{information_digest} requires non-empty action IDs"
                )
            probabilities[action_id] = _canonical_probability(
                raw_probability,
                label=f"{label}.{information_digest}.{action_id}",
            )
        if sum(probabilities.values(), Fraction(0, 1)) != 1:
            raise ValueError(
                f"{label}.{information_digest} probabilities must sum exactly to one"
            )
        normalized[information_digest] = MappingProxyType(probabilities)
    return normalized


def build_t3_bb_candidate_policy_artifact(
    probabilities_by_information_digest: Mapping[
        str, Mapping[str, Fraction | int | str]
    ],
    *,
    checkpoint_sha256: str,
    solver_manifest_sha256: str,
    range_builder_source_sha256: str,
    model_id: str = "fixed_point_t3_bb_candidate_policy_v1",
) -> dict[str, Any]:
    """Build a canonical exact-table artifact for one fixed-point candidate."""

    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a non-empty string")
    checkpoint = _require_sha256(checkpoint_sha256, label="checkpoint_sha256")
    solver = _require_sha256(
        solver_manifest_sha256, label="solver_manifest_sha256"
    )
    source = _require_sha256(
        range_builder_source_sha256, label="range_builder_source_sha256"
    )
    encoded: dict[str, dict[str, str]] = {}
    for digest, row in probabilities_by_information_digest.items():
        information_digest = _require_sha256(
            digest, label="probabilities information digest"
        )
        if not isinstance(row, Mapping) or not row:
            raise ValueError(
                f"probabilities.{information_digest} must be a non-empty object"
            )
        encoded_row: dict[str, str] = {}
        for action_id, raw_probability in row.items():
            if not isinstance(action_id, str) or not action_id:
                raise ValueError("probabilities require non-empty action IDs")
            if isinstance(raw_probability, bool) or isinstance(raw_probability, float):
                raise TypeError("candidate probabilities require exact rational inputs")
            probability = Fraction(raw_probability)
            encoded_row[action_id] = (
                f"{probability.numerator}/{probability.denominator}"
            )
        encoded[information_digest] = dict(sorted(encoded_row.items()))
    # Reuse the strict decoder before hashing so producers cannot create an
    # artifact the runtime itself would reject.
    _normalize_probabilities(encoded, label="probabilities")
    manifest: dict[str, Any] = {
        "schema": MODEL_SCHEMA,
        "model_id": model_id,
        "model_type": MODEL_TYPE,
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "rules_version": RULES_VERSION,
        "promotion_eligible": False,
        "promotion_basis": "requires_verified_t3_bb_fixed_point_binding",
        "supported_turns": [3],
        "supported_actors": ["bb"],
        "no_fallback": True,
        "probability_encoding": "exact_reduced_rational_table",
        "candidate_policy_method": SOLVER_METHOD,
        "solver_manifest_sha256": solver,
        "range_builder_source_sha256": source,
        "checkpoint_sha256": checkpoint,
        "probabilities": {
            digest: encoded[digest] for digest in sorted(encoded)
        },
    }
    artifact = {
        "schema": ARTIFACT_SCHEMA,
        "model_manifest": manifest,
        "artifact_sha256": canonical_sha256(manifest),
    }
    return verify_t3_bb_candidate_policy_artifact(artifact)


def verify_t3_bb_candidate_policy_artifact(
    artifact: Any,
    *,
    t3_bb_likelihood_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate an artifact and optionally bind it to a passing fixed point."""

    raw = _require_mapping(artifact, label="candidate policy artifact")
    _require_exact_keys(raw, _ARTIFACT_KEYS, label="candidate policy artifact")
    if raw.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError("candidate policy artifact schema mismatch")
    manifest = _require_mapping(
        raw.get("model_manifest"), label="candidate policy model_manifest"
    )
    _require_exact_keys(manifest, _MODEL_KEYS, label="candidate policy manifest")
    expected = {
        "schema": MODEL_SCHEMA,
        "model_type": MODEL_TYPE,
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "rules_version": RULES_VERSION,
        "promotion_eligible": False,
        "promotion_basis": "requires_verified_t3_bb_fixed_point_binding",
        "supported_turns": [3],
        "supported_actors": ["bb"],
        "no_fallback": True,
        "probability_encoding": "exact_reduced_rational_table",
        "candidate_policy_method": SOLVER_METHOD,
    }
    for field, wanted in expected.items():
        if manifest.get(field) != wanted:
            raise ValueError(f"candidate policy manifest {field} mismatch")
    if not isinstance(manifest.get("model_id"), str) or not manifest["model_id"].strip():
        raise ValueError("candidate policy model_id must be non-empty")
    for field in (
        "checkpoint_sha256",
        "solver_manifest_sha256",
        "range_builder_source_sha256",
    ):
        _require_sha256(manifest.get(field), label=f"candidate policy {field}")
    _normalize_probabilities(manifest.get("probabilities"), label="probabilities")
    artifact_sha256 = _require_sha256(
        raw.get("artifact_sha256"), label="artifact_sha256"
    )
    if canonical_sha256(manifest) != artifact_sha256:
        raise ValueError("candidate policy artifact content hash mismatch")

    if t3_bb_likelihood_binding is not None:
        binding = verify_t3_bb_likelihood_binding(t3_bb_likelihood_binding)
        if binding["method"] != T3_BB_LIKELIHOOD_METHOD:
            raise ValueError("T3-BB likelihood method mismatch")
        comparisons = {
            "candidate_policy_artifact_sha256": artifact_sha256,
            "candidate_policy_checkpoint_sha256": manifest["checkpoint_sha256"],
            "solver_manifest_sha256": manifest["solver_manifest_sha256"],
            "range_builder_source_sha256": manifest[
                "range_builder_source_sha256"
            ],
        }
        for field, actual in comparisons.items():
            if binding.get(field) != actual:
                raise ValueError(
                    f"candidate policy artifact does not match binding {field}"
                )
    return copy.deepcopy(dict(raw))


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"candidate policy JSON has duplicate key {key!r}")
        result[key] = value
    return result


def read_t3_bb_candidate_policy_artifact(
    path: str | Path,
    *,
    t3_bb_likelihood_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Read strict UTF-8 JSON and verify it before returning a detached copy."""

    artifact_path = Path(path)
    try:
        raw = json.loads(
            artifact_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read candidate policy artifact: {exc}") from exc
    return verify_t3_bb_candidate_policy_artifact(
        raw, t3_bb_likelihood_binding=t3_bb_likelihood_binding
    )


class VerifiedFixedPointT3BBBehaviorModel(FrozenBehaviorModel):
    """No-fallback exact table whose identity matches the fixed-point binding."""

    def __init__(
        self,
        artifact: Mapping[str, Any],
        *,
        _construction_token: object,
    ) -> None:
        if _construction_token is not _CHILD_TOKEN:
            raise TypeError("use build_verified_fixed_point_t3_bb_behavior_model")
        snapshot = copy.deepcopy(dict(artifact))
        manifest = snapshot["model_manifest"]
        self._artifact = snapshot
        self._manifest = copy.deepcopy(dict(manifest))
        self._probabilities = MappingProxyType(
            _normalize_probabilities(
                self._manifest["probabilities"], label="probabilities"
            )
        )

    @property
    def model_id(self) -> str:
        return str(self._manifest["model_id"])

    @property
    def model_sha256(self) -> str:
        return str(self._artifact["artifact_sha256"])

    @property
    def model_manifest(self) -> Mapping[str, Any]:
        return MappingProxyType(copy.deepcopy(self._manifest))

    @property
    def checkpoint_sha256(self) -> str:
        return str(self._manifest["checkpoint_sha256"])

    def action_distribution(self, information: BehaviorInfoSet) -> BehaviorDistribution:
        if not isinstance(information, BehaviorInfoSet):
            raise TypeError("information must be a BehaviorInfoSet")
        if information.turn != 3 or information.actor != "bb":
            raise KeyError("fixed-point child supports exactly the T3-BB route")
        digest = information.digest()
        row = self._probabilities.get(digest)
        if row is None:
            raise KeyError(f"fixed-point T3-BB table has no row for {digest}")
        if set(row) != set(information.legal_action_ids):
            missing = sorted(set(information.legal_action_ids) - set(row))
            extra = sorted(set(row) - set(information.legal_action_ids))
            raise ValueError(
                "fixed-point T3-BB action coverage mismatch: "
                f"missing={missing}, extra={extra}"
            )
        return BehaviorDistribution(
            information_digest=digest,
            probabilities=row,
            source="table",
            used_fallback=False,
        )


class UnpromotedT3BBCandidateBehaviorModel(FrozenBehaviorModel):
    """Exact candidate table used only inside an unfinished fixed-point loop."""

    def __init__(
        self,
        artifact: Mapping[str, Any],
        *,
        _construction_token: object,
    ) -> None:
        if _construction_token is not _ITERATION_CHILD_TOKEN:
            raise TypeError("use build_unpromoted_t3_bb_candidate_behavior_model")
        snapshot = copy.deepcopy(dict(artifact))
        manifest = snapshot["model_manifest"]
        if manifest.get("promotion_eligible") is not False:
            raise ValueError("iteration candidate child must remain non-promotable")
        self._artifact = snapshot
        self._manifest = copy.deepcopy(dict(manifest))
        self._probabilities = MappingProxyType(
            _normalize_probabilities(
                self._manifest["probabilities"], label="probabilities"
            )
        )

    @property
    def model_id(self) -> str:
        return str(self._manifest["model_id"])

    @property
    def model_sha256(self) -> str:
        return str(self._artifact["artifact_sha256"])

    @property
    def model_manifest(self) -> Mapping[str, Any]:
        return MappingProxyType(copy.deepcopy(self._manifest))

    def action_distribution(self, information: BehaviorInfoSet) -> BehaviorDistribution:
        if not isinstance(information, BehaviorInfoSet):
            raise TypeError("information must be a BehaviorInfoSet")
        if information.turn != 3 or information.actor != "bb":
            raise KeyError("iteration candidate supports exactly the T3-BB route")
        digest = information.digest()
        row = self._probabilities.get(digest)
        if row is None:
            raise KeyError(f"iteration T3-BB candidate table has no row for {digest}")
        if set(row) != set(information.legal_action_ids):
            missing = sorted(set(information.legal_action_ids) - set(row))
            extra = sorted(set(row) - set(information.legal_action_ids))
            raise ValueError(
                "iteration T3-BB action coverage mismatch: "
                f"missing={missing}, extra={extra}"
            )
        return BehaviorDistribution(
            information_digest=digest,
            probabilities=row,
            source="table",
            used_fallback=False,
        )


def build_unpromoted_t3_bb_candidate_behavior_model(
    artifact: Mapping[str, Any],
) -> UnpromotedT3BBCandidateBehaviorModel:
    """Load a content-valid candidate without pretending a fixed point passed."""

    verified = verify_t3_bb_candidate_policy_artifact(artifact)
    return UnpromotedT3BBCandidateBehaviorModel(
        verified, _construction_token=_ITERATION_CHILD_TOKEN
    )


class FixedPointIterationBehaviorDispatch(FrozenBehaviorModel):
    """No-fallback T1/T2 + provisional T3-BB router for one iteration."""

    def __init__(
        self,
        t1_t2: FixedPointBootstrapCalibratedHuT1T2BehaviorDispatch,
        t3_bb: UnpromotedT3BBCandidateBehaviorModel,
        manifest: Mapping[str, Any],
        *,
        _construction_token: object,
    ) -> None:
        if _construction_token is not _ITERATION_DISPATCH_TOKEN:
            raise TypeError("use build_fixed_point_iteration_behavior_dispatch")
        self._t1_t2 = t1_t2
        self._t3_bb = t3_bb
        self._manifest = copy.deepcopy(dict(manifest))
        if self._manifest.get("promotion_eligible") is not False:
            raise ValueError("iteration dispatch must remain non-promotable")
        if self._manifest.get("fixed_point_iteration_only") is not True:
            raise ValueError("iteration dispatch must be fixed-point-only")

    @property
    def model_id(self) -> str:
        return str(self._manifest["model_id"])

    @property
    def model_manifest(self) -> Mapping[str, Any]:
        return MappingProxyType(copy.deepcopy(self._manifest))

    @property
    def model_sha256(self) -> str:
        return canonical_sha256(self._manifest)

    def action_distribution(self, information: BehaviorInfoSet) -> BehaviorDistribution:
        if not isinstance(information, BehaviorInfoSet):
            raise TypeError("information must be a BehaviorInfoSet")
        if information.turn == 3 and information.actor == "bb":
            return self._t3_bb.action_distribution(information)
        if information.turn in (1, 2) and information.actor in ("bb", "btn"):
            return self._t1_t2.action_distribution(information)
        raise KeyError(
            f"fixed-point iteration has no route for T{information.turn} "
            f"{information.actor}"
        )


def build_fixed_point_iteration_behavior_dispatch(
    t1_t2_bootstrap: FixedPointBootstrapCalibratedHuT1T2BehaviorDispatch,
    candidate_policy_artifact: Mapping[str, Any],
    *,
    model_id: str = "t3_bb_fixed_point_iteration_behavior_dispatch_v1",
) -> FixedPointIterationBehaviorDispatch:
    """Compose one provisional candidate with the raw-verified T1/T2 bootstrap."""

    if not isinstance(
        t1_t2_bootstrap, FixedPointBootstrapCalibratedHuT1T2BehaviorDispatch
    ):
        raise TypeError("t1_t2_bootstrap must be the fixed-point bootstrap runtime")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a non-empty string")
    bootstrap_manifest = copy.deepcopy(dict(t1_t2_bootstrap.model_manifest))
    if bootstrap_manifest.get("model_type") != BOOTSTRAP_MODEL_TYPE:
        raise ValueError("T1/T2 bootstrap model type mismatch")
    if bootstrap_manifest.get("promotion_eligible") is not False:
        raise ValueError("T1/T2 bootstrap must remain non-promotable")
    if bootstrap_manifest.get("fixed_point_bootstrap_only") is not True:
        raise ValueError("T1/T2 component is not a fixed-point bootstrap")
    if bootstrap_manifest.get("no_fallback") is not True:
        raise ValueError("T1/T2 bootstrap must forbid fallback")
    if canonical_sha256(bootstrap_manifest) != t1_t2_bootstrap.model_sha256:
        raise ValueError("T1/T2 bootstrap manifest/hash mismatch")
    candidate = build_unpromoted_t3_bb_candidate_behavior_model(
        candidate_policy_artifact
    )
    manifest: dict[str, Any] = {
        "schema": MODEL_SCHEMA,
        "model_id": model_id,
        "model_type": ITERATION_MODEL_TYPE,
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "rules_version": RULES_VERSION,
        "promotion_eligible": False,
        "fixed_point_iteration_only": True,
        "fixed_point_converged": False,
        "promotion_basis": "none_unfinished_fixed_point_iteration",
        "route_scope": ITERATION_SCOPE,
        "strategic_strength_evaluated": False,
        "strategic_strength_claimed": False,
        "no_fallback": True,
        "source_calibration_promotion_eligible": bootstrap_manifest[
            "source_calibration_promotion_eligible"
        ],
        "source_calibration_all_required_gates_passed": bootstrap_manifest[
            "source_calibration_all_required_gates_passed"
        ],
        "t1_t2_bootstrap_model_sha256": t1_t2_bootstrap.model_sha256,
        "t3_bb_candidate_policy_artifact_sha256": candidate.model_sha256,
        "routes": [
            {"turn": 1, "actor": "bb", "source": "calibrated_bootstrap"},
            {"turn": 1, "actor": "btn", "source": "calibrated_bootstrap"},
            {"turn": 2, "actor": "bb", "source": "calibrated_bootstrap"},
            {"turn": 2, "actor": "btn", "source": "calibrated_bootstrap"},
            {"turn": 3, "actor": "bb", "source": "provisional_exact_table"},
        ],
    }
    return FixedPointIterationBehaviorDispatch(
        t1_t2_bootstrap,
        candidate,
        manifest,
        _construction_token=_ITERATION_DISPATCH_TOKEN,
    )


def build_verified_fixed_point_t3_bb_behavior_model(
    artifact: Mapping[str, Any],
    *,
    t3_bb_likelihood_binding: Mapping[str, Any],
) -> VerifiedFixedPointT3BBBehaviorModel:
    verified = verify_t3_bb_candidate_policy_artifact(
        artifact, t3_bb_likelihood_binding=t3_bb_likelihood_binding
    )
    return VerifiedFixedPointT3BBBehaviorModel(
        verified, _construction_token=_CHILD_TOKEN
    )


class VerifiedM3BehaviorLikelihoodDispatch(FrozenBehaviorModel):
    """Complete no-fallback behavior router for BB/BTN T3 root ranges."""

    def __init__(
        self,
        t1_t2: VerifiedCalibratedHuT1T2BehaviorDispatch,
        t3_bb: VerifiedFixedPointT3BBBehaviorModel,
        manifest: Mapping[str, Any],
        *,
        _construction_token: object,
    ) -> None:
        if _construction_token is not _COMPOSITE_TOKEN:
            raise TypeError("use build_verified_m3_behavior_likelihood_dispatch")
        self._t1_t2 = t1_t2
        self._t3_bb = t3_bb
        self._manifest = copy.deepcopy(dict(manifest))

    @property
    def model_id(self) -> str:
        return str(self._manifest["model_id"])

    @property
    def model_manifest(self) -> Mapping[str, Any]:
        return MappingProxyType(copy.deepcopy(self._manifest))

    @property
    def model_sha256(self) -> str:
        return canonical_sha256(self._manifest)

    def action_distribution(self, information: BehaviorInfoSet) -> BehaviorDistribution:
        if not isinstance(information, BehaviorInfoSet):
            raise TypeError("information must be a BehaviorInfoSet")
        if information.turn == 3 and information.actor == "bb":
            return self._t3_bb.action_distribution(information)
        if information.turn in (1, 2) and information.actor in ("bb", "btn"):
            return self._t1_t2.action_distribution(information)
        raise KeyError(
            f"no M3 root likelihood route for T{information.turn} "
            f"{information.actor}; T3-BTN/T4 posterior routes remain separate"
        )


def build_verified_m3_behavior_likelihood_dispatch(
    t1_t2_dispatch: VerifiedCalibratedHuT1T2BehaviorDispatch,
    candidate_policy_artifact: Mapping[str, Any],
    *,
    t3_bb_likelihood_binding: Mapping[str, Any],
    model_id: str = "verified_m3_t3_root_behavior_likelihood_dispatch_v1",
) -> VerifiedM3BehaviorLikelihoodDispatch:
    """Compose exact T1/T2 and fixed-point T3-BB routes for M3 root ranges."""

    if not isinstance(
        t1_t2_dispatch, VerifiedCalibratedHuT1T2BehaviorDispatch
    ):
        raise TypeError("t1_t2_dispatch must be the verified calibrated runtime")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a non-empty string")
    binding = verify_t3_bb_likelihood_binding(t3_bb_likelihood_binding)
    t1_manifest = copy.deepcopy(
        dict(
            _require_mapping(
                t1_t2_dispatch.model_manifest, label="T1/T2 runtime manifest"
            )
        )
    )
    if t1_manifest.get("model_type") != T1_T2_RUNTIME_MODEL_TYPE:
        raise ValueError("T1/T2 runtime model type mismatch")
    if t1_manifest.get("promotion_eligible") is not True:
        raise ValueError("T1/T2 runtime is not promotion eligible")
    if t1_manifest.get("t3_bb_route_included") is not False:
        raise ValueError("T1/T2 component unexpectedly contains a T3-BB route")
    if (
        t1_manifest.get("t3_bb_likelihood_binding_sha256")
        != binding["binding_sha256"]
    ):
        raise ValueError("T1/T2 runtime was built from a different T3-BB binding")
    if canonical_sha256(t1_manifest) != t1_t2_dispatch.model_sha256:
        raise ValueError("T1/T2 runtime manifest/hash mismatch")

    t3_bb = build_verified_fixed_point_t3_bb_behavior_model(
        candidate_policy_artifact,
        t3_bb_likelihood_binding=binding,
    )
    manifest: dict[str, Any] = {
        "schema": MODEL_SCHEMA,
        "model_id": model_id,
        "model_type": COMPOSITE_MODEL_TYPE,
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "rules_version": RULES_VERSION,
        "promotion_eligible": True,
        "promotion_scope": COMPOSITE_SCOPE,
        "promotion_basis": "verified_calibration_bridge_and_t3_bb_fixed_point_binding",
        "strategic_strength_evaluated": False,
        "strategic_strength_claimed": False,
        "no_fallback": True,
        "m3_t3_root_behavior_complete": True,
        "all_turn_behavior_complete": False,
        "t4_posterior_routes_complete": False,
        "routes": [
            {"turn": 1, "actor": "bb", "source": "calibrated_t1_t2"},
            {"turn": 1, "actor": "btn", "source": "calibrated_t1_t2"},
            {"turn": 2, "actor": "bb", "source": "calibrated_t1_t2"},
            {"turn": 2, "actor": "btn", "source": "calibrated_t1_t2"},
            {"turn": 3, "actor": "bb", "source": "fixed_point_exact_table"},
        ],
        "t1_t2_component_model_sha256": t1_t2_dispatch.model_sha256,
        "t3_bb_candidate_policy_artifact_sha256": t3_bb.model_sha256,
        "t3_bb_candidate_policy_checkpoint_sha256": t3_bb.checkpoint_sha256,
        "t3_bb_likelihood_binding_sha256": binding["binding_sha256"],
        "fixed_point_evidence_sha256": binding["fixed_point_evidence_sha256"],
        "fixed_point_gate_result_sha256": binding[
            "fixed_point_gate_result_sha256"
        ],
        "fixed_point_config_sha256": binding["fixed_point_config_sha256"],
        "solver_manifest_sha256": binding["solver_manifest_sha256"],
        "range_builder_source_sha256": binding[
            "range_builder_source_sha256"
        ],
    }
    return VerifiedM3BehaviorLikelihoodDispatch(
        t1_t2_dispatch,
        t3_bb,
        manifest,
        _construction_token=_COMPOSITE_TOKEN,
    )


__all__ = [
    "ARTIFACT_SCHEMA",
    "COMPOSITE_MODEL_TYPE",
    "COMPOSITE_SCOPE",
    "ITERATION_MODEL_TYPE",
    "ITERATION_SCOPE",
    "MODEL_TYPE",
    "VerifiedFixedPointT3BBBehaviorModel",
    "VerifiedM3BehaviorLikelihoodDispatch",
    "FixedPointIterationBehaviorDispatch",
    "UnpromotedT3BBCandidateBehaviorModel",
    "build_fixed_point_iteration_behavior_dispatch",
    "build_t3_bb_candidate_policy_artifact",
    "build_verified_fixed_point_t3_bb_behavior_model",
    "build_verified_m3_behavior_likelihood_dispatch",
    "build_unpromoted_t3_bb_candidate_behavior_model",
    "read_t3_bb_candidate_policy_artifact",
    "verify_t3_bb_candidate_policy_artifact",
]
